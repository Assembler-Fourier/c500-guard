"""Low-light alignment flips never crowd out real alarm evidence (21 Nov 2026)."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_watch import ROOT, watcher
from alarm_recorder import AlarmRecorder


class WeakTrackingNoteTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher(zones_path=ROOT / 'config/views/ptz2/zones.json')
        self.obj.armed = True
        self.obj.recorder = mock.Mock(enabled=True)
        self.obj._alignment_blocked = False

    def note(self, now, state='tracking', weak=False, blocked=False):
        self.obj.alignment = {'state': state, 'weak': weak, 'message': f'{state} weak={weak}'}
        self.obj._alignment_blocked = blocked
        self.obj._note_alignment_transition(now)

    def transitions(self):
        return [c for c in self.obj.log_event.call_args_list if c.args[0] == 'alignment-transition']

    def test_low_light_flapping_records_no_video_and_logs_once_a_minute(self):
        now = 100.
        for step in range(60):                       # two minutes of flips every two seconds
            self.note(now, weak=bool(step % 2))
            now += 2.
        self.obj.recorder.alarm.assert_not_called()
        self.assertLessEqual(len(self.transitions()), 4)
        later = self.transitions()[-1]
        self.assertGreater(later.kwargs.get('unlogged_low_light_flips', 0), 20)

    def test_real_losses_still_record_video_and_are_logged(self):
        self.note(100.)
        self.note(102., weak=True)                   # weak but confirmed: no video
        self.obj.recorder.alarm.assert_not_called()
        self.note(104., state='holding', weak=True)  # landmarks lost: video as before
        self.assertEqual(1, self.obj.recorder.alarm.call_count)
        self.note(106., state='calibration_required', blocked=True)
        self.note(108.)
        states = [c.kwargs['current'][0] for c in self.transitions()]
        self.assertEqual(['tracking', 'tracking', 'holding', 'calibration_required', 'tracking'], states)

    def test_a_view_change_is_never_folded_into_the_flip_quota(self):
        self.note(100.)
        self.note(102., weak=True)
        self.obj.view_name = 'anpr'
        self.note(104., weak=False)
        self.assertEqual(3, len(self.transitions()))


class RecordingRetentionTests(unittest.TestCase):
    def make(self, folder, recorder, name, ids, age):
        target = Path(folder) / name
        for suffix in ('.mp4', '.json', '.clean.zip'):
            path = target.with_suffix(suffix)
            path.write_text(json.dumps({'name': name, 'alarms': [{'id': i} for i in ids]}) if suffix == '.json' else 'x')
            os.utime(path, (1_000_000 + age, 1_000_000 + age))
        for ident in ids:
            recorder._index[ident] = name

    def test_alignment_diagnostics_keep_to_their_share_and_go_first(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = AlarmRecorder(folder, None, keep=10)
            self.make(folder, recorder, '20261120-200000-real_1.mp4', ['real_1'], 0)
            self.make(folder, recorder, '20261120-200100-diag-barrier-motion-1.mp4', ['diag-barrier-motion-1'], 1)
            self.make(folder, recorder, '20261120-200200-diag-alignment-2.mp4',
                      ['diag-alignment-2', 'real_2'], 2)    # named after a diagnostic but holds a real alarm
            for n in range(12):
                self.make(folder, recorder, f'20261120-21{n:02d}00-diag-alignment-x{n}.mp4',
                          [f'diag-alignment-x{n}'], 10 + n)
            recorder._prune()
            left = sorted(p.name for p in Path(folder).glob('*.mp4'))
            self.assertIn('20261120-200000-real_1.mp4', left)
            self.assertIn('20261120-200100-diag-barrier-motion-1.mp4', left)
            self.assertIn('20261120-200200-diag-alignment-2.mp4', left)
            only_diag = [n for n in left if '-x' in n]
            self.assertEqual(['20261120-211000-diag-alignment-x10.mp4', '20261120-211100-diag-alignment-x11.mp4'],
                             only_diag)                # a fifth of keep=10, newest kept
            self.assertFalse((Path(folder) / '20261120-210000-diag-alignment-x0.clean.zip').exists())
            self.assertNotIn('diag-alignment-x0', recorder._index)
            self.assertIn('real_2', recorder._index)

    def test_total_cap_still_holds_with_only_alarm_clips(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = AlarmRecorder(folder, None, keep=3)
            for n in range(5):
                self.make(folder, recorder, f'20261120-20{n:02d}00-real_{n}.mp4', [f'real_{n}'], n)
            recorder._prune()
            self.assertEqual(['20261120-200200-real_2.mp4', '20261120-200300-real_3.mp4',
                              '20261120-200400-real_4.mp4'], sorted(p.name for p in Path(folder).glob('*.mp4')))


if __name__ == '__main__':
    unittest.main()
