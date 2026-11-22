"""Bounded encoder lifetime, using isolated real children and scratch clips."""
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from alarm_recorder import AlarmRecorder
import platform_support as ps


class RecorderDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='recorder-deadline-')
        self.addCleanup(self.folder.cleanup)
        self.recorder = AlarmRecorder(self.folder.name, 'isolated-encoder')
        self.addCleanup(self.recorder.stop)
        self.children = []
        self.timers = []
        self.created = threading.Event()
        self.popen = subprocess.Popen
        self.timer = threading.Timer
        self.addCleanup(self.reap_children)

    def reap_children(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)

    @staticmethod
    def clip(payload=b'x' * (1024 * 1024), number=1):
        return {'frames': [(100. + i * .1, payload, None, None, None) for i in range(12)],
                'alarms': [{'id': f'deadline:{number}', 'kind': 'main', 'reason': 'scratch'}],
                'wall_start': 1700000000. + number}

    def launch(self, code):
        def factory(_command, **kwargs):
            # Use the actual interpreter, not Windows' venv launcher, so this
            # Popen handle owns the sleeper directly and cleanup is exact.
            child = self.popen([getattr(sys, '_base_executable', sys.executable), '-c', code], **kwargs)
            self.children.append(child)
            self.created.set()
            return child
        return factory

    def capture_timer(self, *args, **kwargs):
        timer = self.timer(*args, **kwargs)
        self.timers.append(timer)
        return timer

    def assert_reaped(self):
        self.assertTrue(self.children)
        self.assertTrue(all(child.poll() is not None for child in self.children))
        self.assertTrue(all(not timer.is_alive() for timer in self.timers))
        self.assertIsNone(self.recorder._process)
        self.assertIsNone(getattr(self.recorder, '_encoder_timer', None))
        self.assertEqual([], list(Path(self.folder.name).glob('*.part.*')))

    def test_blocked_pipe_never_reaches_communicate_timeout_without_feeding_deadline(self):
        """Reproduce the old order safely: kill this one child after observation."""
        child = self.popen([getattr(sys, '_base_executable', sys.executable), '-c',
                            'import time; time.sleep(20)'], stdin=subprocess.PIPE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           **ps.background_subprocess_kwargs())
        self.children.append(child)
        entered = threading.Event()
        reached_communicate = threading.Event()

        def old_order():
            try:
                entered.set()
                child.stdin.write(b'x' * (1024 * 1024))
                reached_communicate.set()
                child.communicate(timeout=.1)
            except (OSError, subprocess.TimeoutExpired):
                pass

        writer = threading.Thread(target=old_order, daemon=True)
        writer.start()
        try:
            self.assertTrue(entered.wait(1))
            writer.join(.3)
            self.assertTrue(writer.is_alive(), 'the input should still be blocked past .1 seconds')
            self.assertFalse(reached_communicate.is_set(), 'communicate timeout has not started')
        finally:
            child.kill()
            child.wait(timeout=5)
            writer.join(2)
            child.stdin.close()
        self.assertFalse(writer.is_alive())

    def test_deadline_ends_nonreading_encoder_and_cleans_partial_artifacts(self):
        self.recorder.ENCODER_TIMEOUT_S = .25
        started = time.monotonic()
        with mock.patch('alarm_recorder.subprocess.Popen', side_effect=self.launch('import time; time.sleep(20)')), \
             mock.patch('alarm_recorder.threading.Timer', side_effect=self.capture_timer):
            with self.assertRaisesRegex(RuntimeError, 'encoder took too long'):
                self.recorder._write(self.clip())
        self.assertLess(time.monotonic() - started, 4.)
        self.assert_reaped()
        self.assertTrue(self.timers, 'deadline starts before feeding stdin')
        self.assertEqual(0, self.recorder.saved)
        self.assertEqual([], list(Path(self.folder.name).iterdir()))

    def test_stop_unblocks_writer_and_leaves_no_worker_child_or_deadline(self):
        self.recorder.ENCODER_TIMEOUT_S = 20.
        with mock.patch('alarm_recorder.subprocess.Popen', side_effect=self.launch('import time; time.sleep(20)')), \
             mock.patch('alarm_recorder.threading.Timer', side_effect=self.capture_timer):
            self.recorder._submit(self.clip())
            self.assertTrue(self.created.wait(2))
            deadline = time.monotonic() + 2
            while getattr(self.recorder, '_encoder_timer', None) is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertIsNotNone(getattr(self.recorder, '_encoder_timer', None))
            started = time.monotonic()
            self.recorder.stop()
        self.assertLess(time.monotonic() - started, 3.)
        self.assertFalse(self.recorder._worker.is_alive())
        self.assert_reaped()
        self.assertEqual(0, self.recorder._jobs.unfinished_tasks)
        self.assertEqual([], list(Path(self.folder.name).iterdir()))

    def test_deadline_also_covers_encoding_after_all_input_was_read(self):
        self.recorder.ENCODER_TIMEOUT_S = .25
        code = 'import sys, time; sys.stdin.buffer.read(); time.sleep(20)'
        with mock.patch('alarm_recorder.subprocess.Popen', side_effect=self.launch(code)), \
             mock.patch('alarm_recorder.threading.Timer', side_effect=self.capture_timer):
            with self.assertRaisesRegex(RuntimeError, 'encoder took too long'):
                self.recorder._write(self.clip(b'small input'))
        self.assert_reaped()
        self.assertEqual([], list(Path(self.folder.name).iterdir()))

    def test_stop_during_process_creation_reaps_child_without_starting_deadline(self):
        factory = self.launch('import time; time.sleep(20)')

        def create_and_stop(*args, **kwargs):
            child = factory(*args, **kwargs)
            self.recorder.stop()
            return child

        with mock.patch('alarm_recorder.subprocess.Popen', side_effect=create_and_stop), \
             mock.patch('alarm_recorder.threading.Timer', side_effect=self.capture_timer):
            with self.assertRaisesRegex(RuntimeError, 'stopping'):
                self.recorder._write(self.clip())
        self.assert_reaped()
        self.assertTrue(all(timer.ident is None for timer in self.timers))
        self.assertEqual([], list(Path(self.folder.name).iterdir()))

    def test_encoder_error_cancels_deadline_and_saves_no_receipt(self):
        code = "import sys; sys.stderr.write('isolated encoder rejected input'); sys.exit(7)"
        with mock.patch('alarm_recorder.subprocess.Popen', side_effect=self.launch(code)), \
             mock.patch('alarm_recorder.threading.Timer', side_effect=self.capture_timer):
            with self.assertRaisesRegex(RuntimeError, 'isolated encoder rejected input'):
                self.recorder._write(self.clip())
        self.assert_reaped()
        self.assertEqual([], list(Path(self.folder.name).iterdir()))

    def test_real_ffmpeg_success_cancels_deadline_and_publishes_complete_receipt(self):
        ffmpeg = shutil.which('ffmpeg') or r'C:\c500-guard\tools\ffmpeg\bin\ffmpeg.exe'
        if not Path(ffmpeg).is_file():
            self.skipTest('ffmpeg not available')
        from PIL import Image
        image = BytesIO()
        Image.new('RGB', (160, 90), (20, 50, 80)).save(image, format='JPEG')
        self.recorder.ffmpeg = ffmpeg

        def real_encoder(*args, **kwargs):
            child = self.popen(*args, **kwargs)
            self.children.append(child)
            return child

        with mock.patch('alarm_recorder.subprocess.Popen', side_effect=real_encoder), \
             mock.patch('alarm_recorder.threading.Timer', side_effect=self.capture_timer):
            self.recorder._write(self.clip(image.getvalue()))
        self.assert_reaped()
        self.assertEqual(1, self.recorder.saved)
        receipts = list(Path(self.folder.name).glob('*.json'))
        self.assertEqual(1, len(receipts))
        meta = json.loads(receipts[0].read_text())
        self.assertEqual(12, meta['frames'])
        video = Path(self.folder.name) / meta['name']
        self.assertGreater(video.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
