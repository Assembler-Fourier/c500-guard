"""A short video of every alarm, the newest few kept, nothing left running."""
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from PIL import Image, ImageDraw

from test_watch import watch, watcher
from alarm_recorder import AlarmRecorder, CLIP_NAME
import platform_support as ps

FFMPEG = shutil.which("ffmpeg") or str(Path(r"C:\c500-guard\tools\ffmpeg\bin\ffmpeg.exe"))
HAVE_FFMPEG = Path(FFMPEG).is_file()


def jpeg(index):
    image = Image.new("RGB", (160, 90), (20, 30, 40))
    ImageDraw.Draw(image).rectangle([index % 150, 30, index % 150 + 10, 50], fill=(250, 200, 40))
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=80)
    return buffer.getvalue()


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.saved = []
        self.recorder = AlarmRecorder(self.folder.name, FFMPEG if HAVE_FFMPEG else None, before_s=3, after_s=5,
                                      keep=2, clock=self.clock, on_saved=lambda name, ids: self.saved.append((name, ids)))

    def tearDown(self):
        self.recorder.stop()
        self.folder.cleanup()

    def feed(self, seconds, start=0):
        for index in range(int(seconds * 10)):
            self.clock.now += .1
            self.recorder.add_frame(jpeg(start + index))

    def wait_saved(self, count, timeout=60):
        deadline = time.monotonic() + timeout
        while len(self.saved) < count and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertEqual(count, len(self.saved), self.recorder.last_error)

    def test_clip_holds_the_seconds_before_and_after_and_alarms_close_together_share_it(self):
        self.feed(6)
        self.recorder.alarm("s:1", "main", "Exit road: light")
        self.feed(2, 60)
        self.recorder.alarm("s:2", "early", "Top road")
        self.assertTrue(self.recorder.pending_for("s:2"))
        clip = self.recorder._clip
        self.assertAlmostEqual(3., clip["frames"][-1][0] - clip["frames"][0][0], delta=2.3)
        with mock.patch.object(self.recorder, "_submit") as submit:
            self.feed(6, 80)
        finished = submit.call_args.args[0]
        span = finished["frames"][-1][0] - finished["frames"][0][0]
        self.assertAlmostEqual(3 + 2 + 5, span, delta=.35)   # before + gap + after the second alarm
        self.assertEqual(["s:1", "s:2"], [item["id"] for item in finished["alarms"]])
        self.assertIsNone(self.recorder._clip)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    def test_video_is_written_indexed_and_only_the_newest_are_kept(self):
        for number in range(3):
            self.feed(4, number * 100)
            self.recorder.alarm(f"s:{number}", "main", "test")
            self.feed(6, number * 100 + 50)
            self.wait_saved(number + 1)
            self.clock.now += 1
            time.sleep(1.1)                       # distinct file times and names
        clips = sorted(Path(self.folder.name).glob("*.mp4"))
        self.assertEqual(2, len(clips))
        self.assertTrue(all(CLIP_NAME.match(path.name) for path in clips))
        self.assertIsNone(self.recorder.recording_for("s:0"))
        name = self.recorder.recording_for("s:2")
        self.assertTrue(name and name.endswith("s_2.mp4"))
        meta = json.loads((Path(self.folder.name) / name).with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual("s:2", meta["alarms"][0]["id"])
        self.assertGreater((Path(self.folder.name) / name).stat().st_size, 1000)
        again = AlarmRecorder(self.folder.name, FFMPEG, clock=self.clock)
        self.assertEqual(name, again.recording_for("s:2"))
        self.assertIsNone(self.recorder._process, "no encoder left running")

    def test_disabled_short_or_stopped_recordings_write_nothing(self):
        self.recorder.configure(enabled=False)
        self.recorder.alarm("s:1", "main", "x")
        self.assertIsNone(self.recorder._clip)
        self.recorder.configure(enabled=True)
        self.recorder.alarm("s:2", "main", "no pictures before it")
        with mock.patch("subprocess.Popen") as popen:
            self.clock.now += 5
            self.recorder.add_frame(jpeg(1))                # one picture: too short to keep
            popen.assert_not_called()
        self.recorder.alarm("s:3", "main", "x")
        self.recorder.stop()
        self.assertIsNone(self.recorder._clip)
        self.recorder.alarm("s:4", "main", "after stop")
        self.assertIsNone(self.recorder._clip)

    def test_settings_are_clamped(self):
        self.recorder.configure(before_s=99, after_s=1, keep=0)
        status = self.recorder.status()
        self.assertEqual((30., 5., 1), (status["before_s"], status["after_s"], status["keep"]))


class WatcherRecordingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        mock.patch.object(watch, 'RECORDINGS_DIR', Path(self.folder.name)).start()
        mock.patch.object(watch, 'notify').start()
        self.obj = watcher()
        self.obj.recorder = AlarmRecorder(self.folder.name, None)
        self.obj.armed = True

    def tearDown(self):
        mock.patch.stopall()
        self.obj.recorder.stop()
        self.folder.cleanup()

    def test_alarm_starts_a_clip_and_the_alert_lists_its_video(self):
        self.obj.raise_alarm('main', 'Exit road: light')
        alarm_id = self.obj.alarm_id
        alert = self.obj.alerts()['alerts'][0]
        self.assertTrue(alert['recording_pending'])
        self.assertIsNone(alert['recording'])
        name = '20261114-041831-' + alarm_id.replace(':', '_')[-40:] + '.mp4'
        (Path(self.folder.name) / name).write_bytes(b'video')
        self.obj.recorder._index[alarm_id] = name
        version = self.obj.alerts_version
        self.obj._recording_saved(name, [alarm_id])
        self.assertGreater(self.obj.alerts_version, version)
        self.assertEqual(name, self.obj.alerts()['alerts'][0]['recording'])
        self.assertEqual('recording', self.obj.log_event.call_args_list[-1].args[0])

    def test_recording_settings_load_with_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zones.json'
            path.write_text(json.dumps({'recordings': {'keep': 500, 'before_s': -3, 'after_s': 25, 'enabled': False}}))
            self.assertEqual({'enabled': False, 'before_s': 0., 'after_s': 25, 'keep': 200},
                             watch.Watcher._load_recording_settings(path))
            path.write_text('{}')
            self.assertEqual({'enabled': True, 'before_s': 10., 'after_s': 20., 'keep': 25},
                             watch.Watcher._load_recording_settings(path))

    def test_videos_are_served_with_byte_ranges_and_names_are_checked(self):
        name = '20261114-041831-abc_1.mp4'
        data = bytes(range(256)) * 40
        (Path(self.folder.name) / name).write_bytes(data)
        server = watch.start_web(self.obj, 0)
        url = 'http://127.0.0.1:' + str(server.server_address[1])
        try:
            with urllib.request.urlopen(url + '/recordings/' + name, timeout=2) as response:
                self.assertEqual((200, data), (response.status, response.read()))
            request = urllib.request.Request(url + '/recordings/' + name, headers={'Range': 'bytes=100-199'})
            with urllib.request.urlopen(request, timeout=2) as response:
                self.assertEqual(206, response.status)
                self.assertEqual(data[100:200], response.read())
                self.assertEqual(f'bytes 100-199/{len(data)}', response.headers['Content-Range'])
            for bad in ('..%5Czones.json', 'x.mp4', '20261114-041831-abc_1.json'):
                with self.subTest(bad), self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(url + '/recordings/' + bad, timeout=2)
                self.assertEqual(404, caught.exception.code)
        finally:
            server.shutdown()
            server.server_close()


@unittest.skipUnless(ps.WINDOWS, "Windows job objects")
class TieTests(unittest.TestCase):
    def test_child_tied_to_a_process_ends_when_that_process_ends(self):
        script = (
            "import subprocess, sys, time\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import platform_support as ps\n"
            # The venv's python.exe is only a launcher; tie the real interpreter, as ffmpeg is tied.
            "child = subprocess.Popen([getattr(sys, '_base_executable', sys.executable), '-c',\n"
            "                          'import time; time.sleep(20)'],\n"
            "                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "print(child.pid, ps.tie_to_this_process(child), flush=True)\n"
            "time.sleep(.5)\n")
        src = str(Path(__file__).resolve().parents[1] / 'src')
        parent = subprocess.run([sys.executable, '-c', script, src], capture_output=True, text=True, timeout=30)
        pid_text, tied = parent.stdout.split()
        pid = int(pid_text)
        self.assertEqual('True', tied)
        deadline = time.monotonic() + 5
        while self._alive(pid):
            if time.monotonic() > deadline:
                # Seen alive just now, so this is still the test's own sleeper: end it, then fail.
                subprocess.run(['taskkill', '/PID', str(pid), '/F'], capture_output=True)
                self.fail('the tied child outlived its parent')
            time.sleep(.1)

    @staticmethod
    def _alive(pid):
        result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'], capture_output=True, text=True)
        return str(pid) in result.stdout


if __name__ == '__main__':
    unittest.main()
