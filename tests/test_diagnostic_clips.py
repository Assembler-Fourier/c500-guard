"""Diagnostic recording never owns an alarm, speaker or physical barrier."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_watch import ROOT, watch, watcher
from alarm_recorder import AlarmRecorder


class DiagnosticClipTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher(zones_path=ROOT/'config/views/ptz2/zones.json')
        self.obj.armed = True
        self.obj.recorder = mock.Mock(enabled=True)
        self.obj.loudness = mock.Mock(kind=None)

    def test_diagnostic_request_does_not_create_or_acknowledge_an_alarm(self):
        self.obj.alarm, self.obj.alarm_id = 'main', 'real-car'
        self.obj.buzzer.set('main')
        self.obj.visits.acknowledge = mock.Mock()
        self.obj._record_diagnostic_clip('alignment', 'Landmarks briefly unclear', 100.)
        self.obj.recorder.alarm.assert_called_once()
        call = self.obj.recorder.alarm.call_args
        self.assertEqual('diagnostic', call.args[1])
        self.assertEqual(100., call.kwargs['now'])
        self.assertEqual(('main', 'real-car', 'main'),
                         (self.obj.alarm, self.obj.alarm_id, self.obj.buzzer.want))
        self.obj.visits.acknowledge.assert_not_called()
        self.obj.loudness.begin.assert_not_called()
        self.obj.loudness.end.assert_not_called()

    def test_requests_are_bounded_per_topic_at_sixty_seconds(self):
        for stamp in (100., 100.1, 110., 159.99):
            self.obj._record_diagnostic_clip('alignment', 'Synthetic', stamp)
        self.assertEqual(1, self.obj.recorder.alarm.call_count)
        self.obj._record_diagnostic_clip('barrier-motion', 'Synthetic', 120.)
        self.obj._record_diagnostic_clip('alignment', 'Synthetic', 160.)
        self.assertEqual(3, self.obj.recorder.alarm.call_count)

    def test_disarmed_or_disabled_recording_creates_no_recording_claim(self):
        self.obj.armed = False
        self.obj._record_diagnostic_clip('alignment', 'Synthetic', 100.)
        self.obj.armed = True
        self.obj.recorder.enabled = False
        self.obj._record_diagnostic_clip('alignment', 'Synthetic', 101.)
        self.obj.recorder.alarm.assert_not_called()
        self.obj.log_event.assert_not_called()

    def test_recorder_failure_never_escapes_or_changes_alarm(self):
        self.obj.recorder.alarm.side_effect = OSError('Synthetic full disk')
        self.obj._record_diagnostic_clip('alignment', 'Synthetic', 100.)
        self.assertIsNone(self.obj.alarm)
        self.assertIsNone(self.obj.buzzer.want)
        self.obj.loudness.begin.assert_not_called()

    def test_real_recorder_keeps_alarm_identity_when_diagnostic_joins_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = AlarmRecorder(Path(folder), None)
            recorder.add_frame(b'synthetic-jpeg-not-encoded', now=90.)
            recorder.alarm('real-car', 'main', 'Real alarm', now=100.)
            self.obj.recorder = recorder
            self.obj._record_diagnostic_clip('alignment', 'Synthetic', 119.)
            self.assertTrue(recorder.pending_for('real-car'))
            self.assertEqual('real-car', recorder._clip['alarms'][0]['id'])
            self.assertEqual('main', recorder._clip['alarms'][0]['kind'])
            self.assertEqual('diagnostic', recorder._clip['alarms'][1]['kind'])
            self.assertLessEqual(recorder._clip['end_at'], 90.+recorder.MAX_S)
            self.assertIsNone(recorder._worker)
            self.assertEqual([], list(Path(folder).iterdir()))

    def test_subsecond_alignment_loss_keeps_clip_even_when_text_log_is_throttled(self):
        self.obj.alignment = {'state': 'tracking', 'weak': False, 'message': 'Tracking'}
        self.obj._alignment_blocked = False
        self.obj._note_alignment_transition(100.)
        self.obj.alignment = {'state': 'calibration_required', 'weak': False, 'message': 'Brief lost landmarks'}
        self.obj._alignment_blocked = True
        self.obj._note_alignment_transition(100.2)
        self.obj.alignment = {'state': 'tracking', 'weak': False, 'message': 'Recovered'}
        self.obj._alignment_blocked = False
        self.obj._note_alignment_transition(100.4)
        self.obj.recorder.alarm.assert_called_once()
        transitions = [c for c in self.obj.log_event.call_args_list if c.args[0] == 'alignment-transition']
        self.assertEqual(1, len(transitions))

    def test_stable_tracking_and_first_snapshot_do_not_generate_video(self):
        self.obj.alignment = {'state': 'tracking', 'weak': False, 'message': 'Tracking'}
        self.obj._alignment_blocked = False
        for now in (100., 101., 102., 120.):
            self.obj._note_alignment_transition(now)
        self.obj.recorder.alarm.assert_not_called()
        self.assertEqual(1, self.obj.log_event.call_count)

    def test_valid_arm_motion_without_vehicle_alarm_requests_diagnostic_only(self):
        self.obj._arm_tracker = mock.Mock()
        self.obj._arm_tracker.update.return_value = {'exit': {
            'ok': True, 'theta': 30., 'shown': 30., 'motion': 'rising',
            'motion_started': 99., 'at': 100., 'line': None}}
        self.obj._arm_motion_recorder = None
        with mock.patch.object(watch.time, 'monotonic', return_value=100.):
            self.obj._observe_arm_angles(watch.np.zeros((4, 4, 3)), 100., True, {'barriers': []})
        self.obj.recorder.alarm.assert_called_once()
        self.assertIn('without an active vehicle alarm', self.obj.recorder.alarm.call_args.args[2])
        self.assertIsNone(self.obj.alarm)
        self.obj.loudness.begin.assert_not_called()


if __name__ == '__main__':
    unittest.main()
