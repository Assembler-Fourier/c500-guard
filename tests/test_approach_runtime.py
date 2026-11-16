"""Approach AI/light alarm integration; no camera, inference worker or sound."""
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from semantic_monitor import SemanticMonitor
from test_watch import watch, watcher


class ApproachRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        notifications = mock.patch.object(watch, 'notify')
        notifications.start()
        self.addCleanup(notifications.stop)
        self.obj = watcher()
        self.obj.args.warmup = 0.
        self.obj.args.adapt = 0.
        self.obj.alignment_config = {}
        self.obj._alignment_blocked = False
        self.zone = watch.Zone('Lower-left approach', (.05,.5,.4,.45),
                              trigger='ai_light', classes=['car','bus','truck'],
                              motion=False, light=True, severity='main',
                              light_severity='main', light_delta=3., priority=True,
                              route_group='left-approach', on_frames=2, off_frames=3)
        self.obj.zones = [self.zone]
        self.frame = watch.np.full((100,100,3), 80, dtype=watch.np.uint8)
        self.changed = self.frame.copy()
        self.changed[60:80,15:35] = 130
        self.obj.process(self.frame)
        with mock.patch.object(threading.Thread, 'start'):
            self.semantic = SemanticMonitor(self.obj.zones, Path('unused-model.onnx'), interval=1000.)
        self.semantic._enabled = True
        self.semantic._last_submit = self.now
        self.obj.semantic = self.semantic
        self.addCleanup(self.semantic.stop)
        self.first = {'label':'car', 'confidence':.94, 'bbox':[10,50,20,70]}
        self.second = {'label':'car', 'confidence':.93, 'bbox':[30,50,40,70]}

    def deliver(self, objects=(), *, after=.2, frame=None, ai_ready=True):
        self.now += after
        self.obj.last_frame_at = self.now
        if ai_ready:
            self.semantic._status.update(available=True, state='ready', updated_at=self.now,
                                         _epoch=self.semantic._epoch,
                                         _generation=self.semantic._worker_generation,
                                         detections=list(objects))
        self.semantic._clear_queue(self.semantic._output)
        self.semantic._output.put((self.semantic._epoch, self.semantic._worker_generation,
                                   self.now, list(objects)))
        self.obj.process(self.frame if frame is None else frame)

    def light_alarm(self):
        self.deliver(frame=self.changed)
        self.assertIsNone(self.obj.alarm)
        self.deliver(frame=self.changed)
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual('main', self.obj.buzzer.want)

    def test_localized_light_buzzes_after_two_frames_without_ai(self):
        self.light_alarm()
        self.assertEqual('light', self.obj.alarm_details['source'])
        self.assertIn('localized light change', self.obj.alarm_reason)
        self.assertEqual(1, self.obj._alarm_sequence)
        self.assertFalse(self.zone.motion_active)

    def test_global_brightness_and_small_noise_do_not_buzz(self):
        rng = watch.np.random.default_rng(9)
        for _ in range(12):
            noisy = watch.np.clip(self.frame.astype(float) + 15
                                  + rng.integers(-1,2,self.frame.shape), 0,255).astype('uint8')
            self.deliver(frame=noisy)
        self.assertIsNone(self.obj.alarm)
        self.assertIsNone(self.obj.buzzer.want)
        self.assertFalse(self.zone.light_metrics['hit'])
        self.assertGreater(self.zone.light_metrics['global_offset'], 13)

    def test_ack_quiet_is_ten_seconds_and_same_light_episode_does_not_repeat(self):
        self.light_alarm()
        self.assertTrue(self.obj.acknowledge())
        self.assertAlmostEqual(self.obj._ack_until-self.now,10)
        for _ in range(60):
            self.deliver(frame=self.changed)
            self.assertIsNone(self.obj.alarm)
            self.assertIsNone(self.obj.buzzer.want)
        self.assertEqual(1,self.obj._alarm_sequence)
        self.assertEqual(0,self.obj._ack_until)
        self.assertTrue(self.zone._notified['light'])

    def test_a_new_light_episode_can_alert_after_old_episode_clears(self):
        self.light_alarm()
        self.obj.acknowledge()
        for _ in range(55):
            self.deliver()
        self.assertFalse(self.zone.light_active)
        self.assertFalse(self.zone._notified['light'])
        self.deliver(frame=self.changed)
        self.deliver(frame=self.changed)
        self.assertEqual('main',self.obj.alarm)
        self.assertEqual(2,self.obj._alarm_sequence)

    def test_light_visit_consumes_only_first_car_and_second_waits_for_quiet(self):
        self.light_alarm()
        self.assertTrue(self.obj.acknowledge())
        self.deliver([self.first,self.second],frame=self.changed)
        self.deliver([self.first,self.second],frame=self.changed)
        self.assertEqual(1, self.obj.visits.status(self.now)['acknowledged_visits'])
        self.assertIsNone(self.obj.alarm)
        for _ in range(47):
            self.deliver([self.first,self.second],frame=self.changed)
            self.assertIsNone(self.obj.alarm)
        self.deliver([self.first,self.second],after=.21,frame=self.changed)
        self.assertEqual('main',self.obj.alarm)
        self.assertEqual(2,self.obj._alarm_sequence)
        self.assertEqual('ai',self.obj.alarm_details['source'])
        self.assertAlmostEqual(.35,self.obj.alarm_details['bbox_normalized'][0]
                               + (self.obj.alarm_details['bbox_normalized'][2]
                                  - self.obj.alarm_details['bbox_normalized'][0])/2)

    def test_simultaneous_ai_and_light_produce_one_alarm(self):
        self.deliver([self.first],frame=self.changed)
        self.deliver([self.first],frame=self.changed)
        self.assertEqual('main',self.obj.alarm)
        self.assertEqual(1,self.obj._alarm_sequence)
        self.assertEqual('ai',self.obj.alarm_details['source'])
        self.assertTrue(self.zone._notified['light'])
        self.obj.acknowledge()
        for _ in range(60):
            self.deliver([self.first],frame=self.changed)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(1,self.obj._alarm_sequence)

    def test_main_light_evidence_is_saved_to_event_log_and_recent_history(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory)/'events.jsonl'
            self.obj.log_event = types.MethodType(watch.Watcher.log_event,self.obj)
            with mock.patch.object(watch,'EVENT_LOG',log_path):
                self.light_alarm()
                record = watch.json.loads(log_path.read_text(encoding='utf-8').splitlines()[0])
            self.assertEqual('main',record['kind'])
            self.assertEqual('light',record['evidence']['source'])
            self.assertEqual('left-approach',record['evidence']['route_group'])
            self.assertTrue(record['evidence']['measurement']['hit'])
            self.assertGreater(record['evidence']['measurement']['largest_patch_pct'],0)
            self.assertEqual(record['evidence'],self.obj.recent_alerts[0]['evidence'])

    def test_ai_light_reports_ai_failure_but_light_capability_remains(self):
        self.now = 130.
        self.obj.first_frame_seen = True
        self.obj.last_frame_at = self.now
        self.semantic._status.update(available=False,state='retrying',updated_at=None)
        self.obj.check_health(self.now)
        self.assertIsNone(self.obj.alarm)
        self.now += 6
        self.obj.last_frame_at = self.now
        self.obj.check_health(self.now)
        self.assertEqual('fault',self.obj.alarm)
        self.assertEqual('ai',self.obj._fault_kind)
        self.obj.acknowledge()
        self.now += 10.1
        self.deliver(frame=self.changed, ai_ready=False)
        self.deliver(frame=self.changed, ai_ready=False)
        self.assertEqual('main',self.obj.alarm)
        self.assertEqual('light',self.obj.alarm_details['source'])


if __name__ == '__main__':
    unittest.main()
