"""Paired clean incident evidence; scratch files and a fake encoder only."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from alarm_recorder import AlarmRecorder


class FakeEncoder:
    def __init__(self, command, **kwargs):
        self.target = Path(command[-1])
        self.parts = []
        self.returncode = None
        self.stdin = type('Sink', (), {'write': lambda _s, value: self.parts.append(value),
                                      'close': lambda _s: None})()

    def communicate(self, timeout=None):
        self.target.write_bytes(b'fake-mp4:' + b''.join(self.parts))
        self.returncode = 0
        return None, b''

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class CleanRecordingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.saved = []
        self.tied = []
        self.recorder = AlarmRecorder(self.folder.name, 'fake-ffmpeg', before_s=1., after_s=5., keep=2,
            on_saved=lambda name, ids: self.saved.append((name, ids)), tie_child=self.tied.append)
        self.addCleanup(self.recorder.stop)

    def clip(self, *, clean=True, number=1):
        # Exact deliberately irregular receipt times; no image decoding occurs.
        for index in range(12):
            at = 100.+index*.07
            self.recorder.add_frame(f'overlay{index}'.encode(), at,
                clean_jpeg=f'clean{index}'.encode() if clean else None,
                source_at=at-(.03 if index % 2 else .08) if clean else None)
        self.recorder.alarm(f'test:{number}', 'main', 'isolated incident', 100.8)
        result = self.recorder._clip
        result['wall_start'] = 1700000000.+number
        self.recorder._clip = None
        return result

    @staticmethod
    def name(clip):
        stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(clip['wall_start']))
        return f"{stamp}-{clip['alarms'][0]['id'].replace(':', '_')}.mp4"

    def write(self, clip):
        with patch('alarm_recorder.subprocess.Popen', side_effect=FakeEncoder):
            self.recorder._write(clip)
        target = Path(self.folder.name)/self.name(clip)
        return target, json.loads(target.with_suffix('.json').read_text())

    def test_clean_archive_pairs_every_frame_and_retains_irregular_source_timing(self):
        clip = self.clip()
        target, meta = self.write(clip)
        self.assertEqual(12, meta['clean_frames'])
        self.assertEqual(target.with_suffix('.clean.zip').name, meta['clean_archive'])
        self.assertEqual('aligned_before_app_overlays', meta['clean_frame_space'])
        timing = meta['frame_timing']
        self.assertEqual('decoder_receipt', timing['source_meaning'])
        self.assertEqual(clip['frames'][0][3], timing['source_start_at'])
        self.assertNotEqual(timing['frames'][1]['source_offset_s'],
                            timing['frames'][1]['preview_offset_s'])
        with zipfile.ZipFile(target.with_suffix('.clean.zip')) as bundle:
            self.assertEqual(meta, json.loads(bundle.read('manifest.json')))
            self.assertEqual(13, len(bundle.infolist()))
            for index, item in enumerate(clip['frames']):
                row = timing['frames'][index]
                self.assertEqual(index, row['index'])
                self.assertEqual(item[0], row['preview_at'])
                self.assertEqual(item[3], row['source_at'])
                self.assertEqual(item[3]-clip['frames'][0][3], row['source_offset_s'])
                self.assertEqual(item[2], bundle.read(row['clean']))
            self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in bundle.infolist()))
        self.assertEqual(b'fake-mp4:' + b''.join(i[1] for i in clip['frames']), target.read_bytes())
        self.assertEqual([(target.name, ['test:1'])], self.saved)
        self.assertEqual(1, len(self.tied))
        self.assertIsNone(self.recorder._process)

    def test_legacy_call_writes_annotated_video_without_archive_or_invented_source_times(self):
        target, meta = self.write(self.clip(clean=False))
        self.assertIsNone(meta['clean_archive'])
        self.assertFalse(target.with_suffix('.clean.zip').exists())
        self.assertIsNone(meta['frame_timing']['source_start_at'])
        self.assertTrue(all(row['source_at'] is None and row['source_offset_s'] is None
                            and row['clean'] is None for row in meta['frame_timing']['frames']))

    def test_pair_buffering_does_no_filesystem_work_or_encoding(self):
        with patch('alarm_recorder.zipfile.ZipFile') as archive, \
             patch('alarm_recorder.subprocess.Popen') as encoder, \
             patch.object(Path, 'open') as open_file:
            self.clip()
            archive.assert_not_called()
            encoder.assert_not_called()
            open_file.assert_not_called()
            self.assertIsNone(self.recorder._worker)

    def test_mixed_missing_pairs_keep_indexes_and_explicit_omission(self):
        self.recorder.add_frame(b'a', 100., clean_jpeg=b'raw-a', source_at=99.9)
        self.recorder.add_frame(b'b', 100.1)  # Default remains compatible.
        self.recorder.add_frame(b'c', 100.2, clean_jpeg=b'raw-c', source_at=float('nan'))
        self.recorder.add_frame(b'd', 100.3, clean_jpeg=b'raw-d', source_at=101.)
        self.recorder.alarm('test:4', 'main', 'mixed', 100.3)
        clip = self.recorder._clip
        clip['wall_start'] = 1700000004.
        self.recorder._clip = None
        target, meta = self.write(clip)
        self.assertEqual(1, meta['clean_frames'])
        self.assertEqual(2, meta['clean_frames_omitted'])
        rows = meta['frame_timing']['frames']
        self.assertEqual('source_timestamp_unavailable', rows[2]['clean_omitted'])
        self.assertEqual([0, 1, 2, 3], [row['index'] for row in rows])
        with zipfile.ZipFile(target.with_suffix('.clean.zip')) as bundle:
            self.assertEqual(['manifest.json', 'clean/000000.jpg'], bundle.namelist())

    def test_clean_byte_budgets_and_frame_count_bound_memory(self):
        recorder = self.recorder
        recorder.configure(before_s=0)
        recorder.MAX_CLEAN_FRAME_BYTES = 5
        recorder.MAX_CLEAN_BUFFER_BYTES = 8
        recorder.MAX_CLEAN_CLIP_BYTES = 12
        recorder.MAX_FRAMES = 12
        recorder.add_frame(b'a', 100., clean_jpeg=b'123456', source_at=99.9)
        self.assertIsNone(recorder._buffer[-1][2])
        self.assertEqual('clean_frame_byte_limit', recorder._buffer[-1][4])
        recorder.add_frame(b'a', 100.1, clean_jpeg=b'1234', source_at=100.)
        recorder.alarm('test:5', 'main', 'budget', 100.1)
        with patch.object(recorder, '_submit') as submit:
            for index in range(20):
                at = 100.2+index*.01
                recorder.add_frame(b'a', at, clean_jpeg=b'1234', source_at=at-.001)
        self.assertLessEqual(recorder._buffer_clean_bytes, 8)
        self.assertLessEqual(len(recorder._buffer), 2)
        finished = submit.call_args.args[0]
        self.assertEqual(12, len(finished['frames']))
        self.assertLessEqual(sum(len(i[2]) for i in finished['frames'] if i[2]), 12)
        self.assertTrue(any(i[4] == 'clean_clip_byte_limit' for i in finished['frames']))

    def test_maximum_duration_also_caps_first_alarm_with_long_before_and_after(self):
        recorder = self.recorder
        recorder.configure(before_s=30, after_s=60)
        for index in range(300):
            at = 100.+index*.1
            recorder.add_frame(b'overlay', at, clean_jpeg=b'clean', source_at=at-.02)
        recorder.alarm('test:6', 'main', 'bounded', 130.)
        self.assertEqual(160., recorder._clip['end_at'])
        with patch.object(recorder, '_submit') as submit:
            recorder.add_frame(b'late', 165., clean_jpeg=b'late', source_at=164.9)
        frames = submit.call_args.args[0]['frames']
        self.assertLessEqual(frames[-1][0]-frames[0][0], recorder.MAX_S)
        self.assertNotEqual(b'late', frames[-1][1])

    def test_archive_failure_preserves_existing_artifacts_and_removes_temporary_files(self):
        clip = self.clip()
        target = Path(self.folder.name)/self.name(clip)
        paths = (target, target.with_suffix('.json'), target.with_suffix('.clean.zip'))
        for path in paths:
            path.write_bytes(b'previous-complete-artifact')
        with patch('alarm_recorder.subprocess.Popen', side_effect=FakeEncoder), \
             patch('alarm_recorder.zipfile.ZipFile.writestr', side_effect=OSError('scratch disk failure')):
            with self.assertRaisesRegex(OSError, 'scratch disk failure'):
                self.recorder._write(clip)
        self.assertEqual([], self.saved)
        self.assertTrue(all(path.read_bytes() == b'previous-complete-artifact' for path in paths))
        self.assertEqual([], list(Path(self.folder.name).glob('*.part.*')))

    def test_metadata_preparation_failure_preserves_existing_artifacts(self):
        clip = self.clip()
        target = Path(self.folder.name)/self.name(clip)
        paths = (target, target.with_suffix('.json'), target.with_suffix('.clean.zip'))
        for path in paths:
            path.write_bytes(b'old')
        write_bytes = Path.write_bytes

        def fail_receipt(path, data):
            if path.name.endswith('.part.json'):
                raise OSError('scratch receipt failure')
            return write_bytes(path, data)

        with patch('alarm_recorder.subprocess.Popen', side_effect=FakeEncoder), \
             patch.object(Path, 'write_bytes', fail_receipt):
            with self.assertRaisesRegex(OSError, 'scratch receipt failure'):
                self.recorder._write(clip)
        self.assertTrue(all(path.read_bytes() == b'old' for path in paths))
        self.assertEqual([], list(Path(self.folder.name).glob('*.part.*')))

    def test_publication_failure_rolls_back_new_bundle_without_touching_old_one(self):
        clip = self.clip()
        target = Path(self.folder.name)/self.name(clip)
        original = (target, target.with_suffix('.json'), target.with_suffix('.clean.zip'))
        for path in original:
            path.write_bytes(b'old-complete')
        replace = os.replace

        def fail_archive(source, destination):
            if str(source).endswith('.part.clean.zip'):
                raise OSError('atomic archive publication failed')
            return replace(source, destination)

        with patch('alarm_recorder.subprocess.Popen', side_effect=FakeEncoder), \
             patch('alarm_recorder.os.replace', side_effect=fail_archive):
            with self.assertRaisesRegex(OSError, 'atomic archive publication failed'):
                self.recorder._write(clip)
        self.assertEqual(set(original), set(Path(self.folder.name).iterdir()))
        self.assertTrue(all(path.read_bytes() == b'old-complete' for path in original))
        self.assertEqual([], self.saved)

    def test_repeated_name_gets_a_new_paired_basename(self):
        clip = self.clip()
        original, _ = self.write(clip)
        before = {path: path.read_bytes() for path in (original, original.with_suffix('.json'),
                                                      original.with_suffix('.clean.zip'))}
        with patch('alarm_recorder.subprocess.Popen', side_effect=FakeEncoder):
            self.recorder._write(clip)
        self.assertNotEqual(self.saved[0][0], self.saved[1][0])
        self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))
        newest = Path(self.folder.name)/self.saved[1][0]
        receipt = json.loads(newest.with_suffix('.json').read_text())
        self.assertEqual(newest.name, receipt['name'])
        self.assertEqual(newest.with_suffix('.clean.zip').name, receipt['clean_archive'])
        reloaded = AlarmRecorder(self.folder.name, 'fake-ffmpeg')
        self.addCleanup(reloaded.stop)
        self.assertEqual(newest.name, reloaded.recording_for('test:1'))

    def test_archive_is_pruned_with_its_video_and_index_reloads(self):
        self.recorder.configure(keep=1)
        old, _ = self.write(self.clip(number=1))
        newer, _ = self.write(self.clip(number=2))
        for path in (old, old.with_suffix('.json'), old.with_suffix('.clean.zip')):
            self.assertFalse(path.exists())
        self.assertTrue(newer.with_suffix('.clean.zip').is_file())
        reloaded = AlarmRecorder(self.folder.name, 'fake-ffmpeg')
        self.addCleanup(reloaded.stop)
        self.assertEqual(newer.name, reloaded.recording_for('test:2'))
        self.assertIsNone(reloaded.recording_for('test:1'))

    def test_loading_removes_only_recorder_partial_artifacts(self):
        folder = Path(self.folder.name)
        for suffix in ('.part.mp4', '.part.clean.zip', '.part.json'):
            (folder/('20231114-000000-test'+suffix)).write_bytes(b'partial')
        complete = folder/'20231114-000000-test.clean.zip'
        complete.write_bytes(b'complete')
        recorder = AlarmRecorder(folder, 'fake-ffmpeg')
        self.addCleanup(recorder.stop)
        self.assertEqual([], list(folder.glob('*.part.*')))
        self.assertEqual(b'complete', complete.read_bytes())

    def test_stop_interrupts_archive_writer_clears_all_buffers_and_leaves_no_extra_worker(self):
        clip = self.clip()
        entered, release = threading.Event(), threading.Event()
        write_entry = zipfile.ZipFile.writestr

        def gated_write(bundle, name, data, *args, **kwargs):
            if name == 'manifest.json':
                entered.set()
                if not release.wait(2.):
                    raise AssertionError('Fake writer was not released')
            return write_entry(bundle, name, data, *args, **kwargs)

        with patch('alarm_recorder.subprocess.Popen', side_effect=FakeEncoder), \
             patch('alarm_recorder.zipfile.ZipFile.writestr', gated_write):
            self.recorder._submit(clip)
            self.assertTrue(entered.wait(1.))
            stopping = threading.Thread(target=self.recorder.stop)
            stopping.start()
            deadline = time.monotonic()+1.
            while not self.recorder._stopped and time.monotonic() < deadline:
                time.sleep(.001)
            release.set()
            stopping.join(3.)
            self.assertFalse(stopping.is_alive())
            self.assertFalse(self.recorder._worker.is_alive())
            self.assertEqual('alarm-recorder', self.recorder._worker.name)
        self.assertIsNone(self.recorder._process)
        self.assertEqual(0, len(self.recorder._buffer))
        self.assertTrue(self.recorder._jobs.empty())
        self.assertEqual([], list(Path(self.folder.name).iterdir()))
        self.recorder.add_frame(b'after', 200., clean_jpeg=b'clean', source_at=199.)
        self.recorder._submit(clip)
        self.assertEqual(0, len(self.recorder._buffer))
        self.assertTrue(self.recorder._jobs.empty())


if __name__ == '__main__':
    unittest.main()
