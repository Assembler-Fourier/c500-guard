"""Quiet deadlines, input races and local HTTP acknowledgement contract."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from test_watch import watch, watcher


class AcknowledgementTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.semantic = None
        self.clock = mock.patch.object(watch.time, 'monotonic', return_value=100.).start()
        self.notify = mock.patch.object(watch, 'notify').start()

    def tearDown(self):
        mock.patch.stopall()

    def test_all_channels_quiet_until_exact_deadline(self):
        self.obj.raise_alarm('main', 'arrival')
        self.assertTrue(self.obj.acknowledge())
        self.obj.log_event.reset_mock()
        self.notify.reset_mock()
        for instant in (100., 102., 104.999):
            self.clock.return_value = instant
            for kind in ('early', 'main', 'fault'):
                self.assertFalse(self.obj.raise_alarm(kind, 'during quiet'))
                self.assertIsNone(self.obj.buzzer.want)
        self.assertFalse(self.obj.log_event.called)
        self.assertFalse(self.notify.called)
        self.clock.return_value = 105.
        self.assertTrue(self.obj.raise_alarm('main', 'current arrival'))
        self.assertEqual(self.obj.buzzer.want, 'main')

    def test_ordinary_input_and_repeated_ack_cannot_extend_quiet(self):
        self.assertFalse(self.obj.acknowledge())
        self.assertEqual(self.obj.input_status()['ack_remaining_s'], 0)
        self.obj.raise_alarm('main', 'arrival')
        alarm_id = self.obj.alarm_id
        self.assertTrue(self.obj.acknowledge(alarm_id))
        self.clock.return_value = 103.
        self.assertFalse(self.obj.acknowledge(alarm_id))
        self.assertFalse(self.obj.acknowledge())
        self.assertEqual(self.obj.input_status()['ack_remaining_s'], 2.)
        self.assertIsNone(self.obj.input_status()['alarm_id'])

    def test_global_input_counts_only_after_one_audible_second(self):
        self.obj.raise_alarm('main', 'arrival')
        alarm_id = self.obj.alarm_id
        self.clock.return_value = 100.99
        for via in ('input-any-key', 'input-mouse'):
            self.assertFalse(self.obj.acknowledge(alarm_id, via=via))
        self.assertEqual(self.obj.buzzer.want, 'main')
        self.assertEqual(self.obj.alarm_id, alarm_id)
        self.clock.return_value = 101.
        self.assertTrue(self.obj.acknowledge(alarm_id, via='input-any-key'))
        self.assertIsNone(self.obj.buzzer.want)

    def test_dashboard_and_ctl_acknowledge_immediately(self):
        for via in ('web', 'ctl'):
            with self.subTest(via=via):
                obj = watcher()
                obj.semantic = None
                obj.raise_alarm('main', 'arrival')
                self.assertTrue(obj.acknowledge(obj.alarm_id, via=via))
                self.assertIsNone(obj.buzzer.want)

    def test_old_input_cannot_ack_later_alarm_or_new_watcher(self):
        self.obj.raise_alarm('early', 'first')
        old_id = self.obj.alarm_id
        self.obj.raise_alarm('fault', 'new fault')
        self.assertFalse(self.obj.acknowledge(old_id))
        self.assertEqual(self.obj.buzzer.want, 'fault')
        other = watcher()
        other.raise_alarm('early', 'first after restart')
        self.assertNotEqual(other.alarm_id, old_id)
        self.assertFalse(other.acknowledge(old_id))

    def test_disarm_cancels_quiet_and_expiry_cannot_arm(self):
        self.obj.raise_alarm('main', 'arrival')
        self.obj.acknowledge()
        self.obj.set_armed(False)
        self.clock.return_value = 110.
        self.obj.check_health(110.)
        self.assertFalse(self.obj.armed)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(self.obj.input_status()['ack_remaining_s'], 0)

    def test_silence_precedes_slow_log_write(self):
        self.obj.raise_alarm('main', 'arrival')
        entered, release = threading.Event(), threading.Event()
        def slow_log(*args, **fields):
            entered.set()
            release.wait(2)
        self.obj.log_event = slow_log
        worker = threading.Thread(target=self.obj.acknowledge)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertIsNone(self.obj.buzzer.want)
            self.assertIsNone(self.obj.alarm)
        finally:
            release.set()
            worker.join(2)

    def test_failed_state_save_is_visible_and_same_setting_can_retry(self):
        self.obj.raise_alarm('main', 'arrival')
        with mock.patch.object(watch, 'ARMED_FILE') as path:
            path.with_suffix.return_value.write_text.side_effect = OSError('disk full')
            self.obj.set_armed(False)
            self.assertFalse(self.obj.armed)
            self.assertIsNone(self.obj.buzzer.want)
            self.assertIn('could not be saved', self.obj.state_dict()['control_warning'])
            path.with_suffix.return_value.write_text.side_effect = None
            self.obj.set_armed(False)
            self.assertEqual(self.obj.control_warning, '')

    def test_current_motion_rechecked_but_departed_motion_not_replayed(self):
        for still_present in (True, False):
            with self.subTest(still_present=still_present):
                obj = watcher()
                obj.semantic = None
                obj.args.warmup = 0
                obj.args.adapt = 0
                obj.zones = [watch.Zone(name='gate', rect=(0.,0.,1.,1.), on_frames=1)]
                base = watch.np.full((20,20,3), 80., watch.np.float32)
                changed = base.copy()
                changed[:5,:5] = 180.
                self.clock.return_value = 100.
                obj.process(base)
                obj.process(changed)
                self.assertEqual(obj.alarm, 'main')
                obj.acknowledge()
                self.clock.return_value = 103.
                obj.process(changed if still_present else base)
                self.assertIsNone(obj.alarm)
                self.clock.return_value = 105.
                obj.process(changed if still_present else base)
                self.assertEqual(obj.alarm, 'main' if still_present else None)

    def test_fault_cleared_during_quiet_not_replayed(self):
        self.obj.first_frame_seen = True
        self.obj.last_frame_at = 99.
        self.obj.check_health(100.)
        self.obj.acknowledge()
        self.clock.return_value = 105.
        self.obj.last_frame_at = 105.
        self.obj.check_health(105.)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(self.obj.fault_reason, '')


class InputAPIContractTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.notify = mock.patch.object(watch, 'notify').start()
        self.server = watch.start_web(self.obj, 0)
        self.url = 'http://127.0.0.1:'+str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        mock.patch.stopall()

    def request(self, path, data=None, via='input-any-key', origin=None):
        headers = {'X-Via':via}
        if origin: headers['Origin'] = origin
        req = urllib.request.Request(self.url+path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.load(response)

    def test_input_status_is_minimal_and_conditional_ack_is_atomic(self):
        self.obj.raise_alarm('main', 'arrival')
        status = self.request('/input-status')
        self.assertEqual(set(status), {'armed','alarm_id','alarm_started_at','ack_remaining_s'})
        self.assertTrue(status['alarm_id'])
        self.obj.raise_alarm('fault', 'fault superseded arrival')
        self.obj.alarm_since -= self.obj.input_ack_min_s  # The siren has now been heard.
        stale = self.request('/ack', json.dumps({'alarm_id':status['alarm_id']}).encode())
        self.assertFalse(stale['acknowledged'])
        self.assertEqual(self.obj.alarm, 'fault')
        result = self.request('/ack', json.dumps({'alarm_id':self.obj.alarm_id}).encode(), via='input-mouse')
        self.assertTrue(result['acknowledged'])
        self.assertGreater(result['ack_remaining_s'], 4.8)
        self.assertIsNone(self.obj.alarm)

    def test_helper_polls_are_visible_and_alert_images_are_name_checked(self):
        self.assertFalse(self.obj.state_dict()['input_helper']['connected'])
        self.request('/input-status')
        self.assertTrue(self.obj.state_dict()['input_helper']['connected'])
        self.assertEqual(set(self.request('/alerts')), {'version', 'alerts'})
        for name in ('..%2F..%2Fevents.jsonl', '..\\armed.json', 'x.jpg', '20260911-190000-abcdef1-main.jpg'):
            with self.subTest(name=name), self.assertRaises(urllib.error.HTTPError) as caught:
                self.request('/alerts/' + name)
            self.assertEqual(caught.exception.code, 404)

    def test_bad_input_payloads_and_cross_origin_cannot_ack(self):
        self.obj.raise_alarm('main', 'arrival')
        for payload in (b'', b'null', b'[]', b'{"alarm_id":null}', b'{"alarm_id":""}', b'x'*257):
            with self.subTest(payload=payload), self.assertRaises(urllib.error.HTTPError) as caught:
                self.request('/ack',payload)
            self.assertEqual(caught.exception.code,400)
            self.assertEqual(self.obj.alarm,'main')
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/ack',b'{}', origin='http://unrelated.example')
        self.assertEqual(caught.exception.code,403)
        self.assertEqual(self.obj.alarm,'main')


if __name__ == '__main__':
    unittest.main()
