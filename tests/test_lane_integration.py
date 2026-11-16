"""Watcher lane proof and audio integration with fake speakers, threads and HTTP."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_watch import ROOT, watch, watcher
from service_cycle import ServiceCycleMonitor
from guard_presence import LoudnessPlan


class LaneIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.now = 103.25
        self.clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.obj = watcher(zones_path=ROOT/'config/views/ptz2/zones.json')
        self.obj.armed = True
        self.obj._pending_rearm = self.obj._alignment_blocked = False
        self.obj.warmup_started = 0.
        self.obj.alignment = {'state': 'tracking'}
        self.obj.alarm, self.obj.alarm_id = 'main', 'current-car'
        self.obj.alarm_since = 100.
        self.obj.alarm_reason = 'Exit approach: car'
        self.obj.alarm_details = dict(route_group='campus-exit', route_role='exit', source='ai')
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'trusted': True, 'barriers': [], 'service': {'routes': {}}}
        self.obj.loudness = mock.Mock(kind='main')
        self.obj.buzzer.set('main')
        self.obj.visits.acknowledge = mock.Mock()

    def clear_intent(self):
        monitor = ServiceCycleMonitor()
        alarm = dict(id='current-car', route='campus-exit', started_at=100.)
        for stamp in (100., 101., 102., 103.):
            # The camera/arm is 200 ms newer than the latest completed model.
            actions = monitor.update({'campus-exit': dict(
                barrier='down', barrier_observed_at=stamp+.2,
                occupancy='occupied' if stamp == 100. else 'clear',
                occupancy_observed_at=stamp, model_ready=True)}, alarm, now=stamp+.2)
        return actions[0]

    def movement_intent(self, route='campus-exit'):
        return dict(type='raised', alarm_id='current-car', route=route,
                    alarm_started_at=100., origin_observed_at=101.,
                    position_since=102., observed_at=103.2)

    def test_async_model_and_current_camera_proof_resolves_current_visit(self):
        intent = self.clear_intent()
        self.assertGreater(intent['barrier_observed_at'], intent['observed_at'])
        self.assertTrue(self.obj._resolve_service_visit(intent))
        self.obj.visits.acknowledge.assert_called_once_with('current-car', self.now)
        self.assertIsNone(self.obj.alarm)
        self.assertIsNone(self.obj.buzzer.want)
        self.assertLess(self.obj._ack_until, self.now)

    def test_clear_proof_rejects_invalid_old_and_future_sources(self):
        original = self.clear_intent()
        cases = [dict(alarm_started_at=99.), dict(occupied_at=99.),
                 dict(clear_since=102.5), dict(barrier_observed_at=99.),
                 dict(barrier_observed_at=104.), dict(observed_at=104.),
                 dict(barrier_state='unknown'), dict(route='campus-entrance')]
        for key in ('alarm_started_at', 'occupied_at', 'clear_since', 'barrier_observed_at', 'observed_at'):
            cases.extend({key: value} for value in (None, True, float('nan'), float('inf'), '103'))
        for change in cases:
            with self.subTest(change=change):
                self.assertFalse(self.obj._resolve_service_visit({**original, **change}))
                self.assertEqual('main', self.obj.alarm)
        self.obj.visits.acknowledge.assert_not_called()

    def test_sticky_competing_visit_blocks_both_service_paths(self):
        self.obj.barrier_workflow.status.return_value['service']['routes']['campus-exit'] = {
            'alarm_id': 'current-car', 'blocked_by_new_arrival': True}
        self.assertFalse(self.obj._resolve_service_visit(self.clear_intent()))
        self.assertFalse(self.obj._silence_for_barrier_change(self.movement_intent()))
        self.assertEqual('main', self.obj.alarm)

    def test_clear_road_requires_no_current_angle_contradiction(self):
        intent = self.clear_intent()
        self.obj._arm_live = {'exit': {'shown': 90., 'at_wall': watch.time.time()}}
        self.assertFalse(self.obj._resolve_service_visit(intent))
        self.assertEqual('main', self.obj.alarm)
        self.obj._arm_live['exit']['shown'] = 0.
        self.assertTrue(self.obj._resolve_service_visit(intent))

    def test_explicit_pedestrian_role_cannot_fall_back_to_exit_route_name(self):
        self.obj.alarm_details['route_role'] = 'pedestrian'
        self.assertIsNone(self.obj._alarm_lane())
        self.assertFalse(self.obj._resolve_service_visit(self.clear_intent()))
        self.assertFalse(self.obj._silence_for_barrier_change(self.movement_intent()))

    def test_explicit_entrance_lane_wins_over_exit_substring_in_route(self):
        route = 'Exit-side-Entrance'
        self.obj.alarm_details.update(route_group=route, route_role='entrance')
        self.obj._arm_live = {'entrance': {'shown': 90., 'at_wall': watch.time.time()}}
        self.assertTrue(self.obj._silence_for_barrier_change(self.movement_intent(route)))

    def test_current_matching_lane_lowering_still_acknowledges(self):
        self.obj._arm_live = {'exit': {'shown': 0., 'at_wall': watch.time.time()}}
        self.assertTrue(self.obj._silence_for_barrier_change({**self.movement_intent(), 'type': 'lowered'}))
        self.assertIsNone(self.obj.alarm)
        self.assertTrue(self.obj.acked)

    def test_missing_or_invalid_movement_observation_is_refused_without_exception(self):
        for stamp in (None, True, float('nan'), float('inf'), '103'):
            with self.subTest(stamp=stamp):
                self.assertFalse(self.obj._silence_for_barrier_change({**self.movement_intent(), 'observed_at': stamp}))
        missing = self.movement_intent()
        missing.pop('observed_at')
        self.assertFalse(self.obj._silence_for_barrier_change(missing))
        self.obj.visits.acknowledge.assert_not_called()

    def test_current_angle_is_read_before_service_movement_is_applied(self):
        self.obj.warmup_started = 0.
        self.obj.barrier_workflow.observe.return_value = [self.movement_intent()]
        self.obj._arm_live = {'exit': {'shown': 0., 'at_wall': watch.time.time()}}
        def current_angle(*args):
            self.obj._arm_live = {'exit': {'shown': 90., 'at_wall': watch.time.time()}}
        with mock.patch.object(self.obj, '_observe_arm_angles', side_effect=current_angle), \
                mock.patch.object(self.obj, '_ack_on_any_arm'), \
                mock.patch.object(self.obj, '_note_arm_memory'):
            self.obj._observe_barrier_service(watch.np.zeros((4, 4, 3)), 103.2)
        self.assertIsNone(self.obj.alarm)
        self.obj.visits.acknowledge.assert_called_once()

    def test_failed_workflow_discards_old_motion_and_does_not_ack(self):
        self.obj.barrier_workflow.observe.side_effect = ValueError('synthetic bad frame')
        self.obj._barrier_error_logged_at = self.now
        self.obj._arm_tracker = mock.Mock()
        self.obj._arm_live = {'exit': {'motion': 'rising', 'motion_started': 102.}}
        self.obj._observe_barrier_service(watch.np.zeros((4, 4, 3)), 103.2)
        self.assertEqual({}, self.obj._arm_live)
        self.obj._arm_tracker.reset.assert_called_once()
        self.obj.barrier_workflow.reset.assert_called_once()
        self.obj.visits.acknowledge.assert_not_called()


class LaneSoundIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher(zones_path=ROOT/'config/views/ptz2/zones.json')
        self.obj.loudness = mock.Mock(kind=None)
        self.targets = []
        def thread(*args, **kwargs):
            return SimpleNamespace(start=lambda: self.targets.append(kwargs['target']))
        self.thread = mock.patch.object(watch.threading, 'Thread', side_effect=thread)
        self.thread.start()
        self.addCleanup(self.thread.stop)
        self.notify = mock.patch.object(watch, 'notify')
        self.notify.start()
        self.addCleanup(self.notify.stop)

    def test_lane_test_selects_lane_and_rejects_overlap(self):
        self.assertTrue(self.obj.play_test_tone(lane='entrance'))
        self.assertEqual('entrance', self.obj.loudness.begin.call_args.kwargs['lane'])
        with self.assertRaises(PermissionError):
            self.obj.play_test_tone(lane='exit')
        with mock.patch.object(watch.time, 'sleep'):
            self.targets[0]()
        self.assertIsNone(self.obj._test_tone)
        self.assertIsNone(self.obj.buzzer.want)
        self.assertTrue(self.obj.play_test_tone(lane='exit'))

    def test_test_completion_cannot_stop_a_new_real_lane_alarm(self):
        self.obj.play_test_tone(lane='entrance')
        self.assertTrue(self.obj.raise_alarm('main', 'Exit car', details={
            'route_role': 'exit', 'route_group': 'campus-exit', 'source': 'ai'}))
        self.assertEqual('exit', self.obj.loudness.begin.call_args.kwargs['lane'])
        alarm_id = self.obj.alarm_id
        with mock.patch.object(watch.time, 'sleep'):
            self.targets[0]()
        self.assertEqual(alarm_id, self.obj.alarm_id)
        self.assertEqual('main', self.obj.buzzer.want)
        self.obj.loudness.end.assert_not_called()

    def request(self, body, *, origin=None):
        class FakeServer:
            def __init__(self, address, handler):
                self.RequestHandlerClass = handler
            def serve_forever(self):
                pass
        with mock.patch.object(watch, 'ThreadingHTTPServer', FakeServer):
            server = watch.start_web(self.obj, 0)
        handler = object.__new__(server.RequestHandlerClass)
        handler.path = '/test-tone'
        handler.headers = {'Host': '127.0.0.1:8787', 'X-Via': 'web', 'Content-Length': str(len(body))}
        if origin:
            handler.headers['Origin'] = origin
        handler.rfile = io.BytesIO(body)
        result = []
        handler._send = lambda code, content_type, data: result.append((code, data))
        handler.do_POST()
        return result[0]

    def test_endpoint_accepts_lane_without_network_or_speaker_calls(self):
        code, data = self.request(b'{"lane":"exit"}')
        self.assertEqual(200, code)
        self.assertTrue(json.loads(data)['played'])
        self.assertEqual('exit', self.obj.loudness.begin.call_args.kwargs['lane'])
        self.assertEqual(409, self.request(b'{"lane":"entrance"}')[0])

    def test_endpoint_rejects_invalid_lane_body_and_origin(self):
        for body in (b'{"lane":true}', b'{"lane":"pedestrian"}', b'{"seconds":100}',
                     b'null', b'[]', b'invalid', b'x'*129):
            with self.subTest(body=body):
                self.assertEqual(400, self.request(body)[0])
        self.assertEqual(403, self.request(b'{"lane":"exit"}', origin='http://other.invalid')[0])
        self.obj.loudness.begin.assert_not_called()

    def test_controller_selects_lane_assets_without_changing_faults_or_volume(self):
        profiles = []
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        controller = watch.AlarmLoudness(
            SimpleNamespace(plan=lambda **k: LoudnessPlan('unknown', 1., 1., 0., 'test')),
            read_fn=lambda: {'available': True, 'volume': .7, 'muted': False},
            apply_fn=lambda **k: {'applied': True}, clock=lambda: 100.,
            restore_path=Path(folder.name)/'restore.json', profile_fn=profiles.append)
        for lane in ('entrance', 'exit'):
            controller.begin('main', source='ai', confirmed=True, lane=lane)
            self.assertEqual(f'{lane}_soft', profiles[-1])
            self.assertEqual('soft', controller.status()['base_profile'])
            self.assertAlmostEqual(.30, controller.applied)
            self.assertEqual(lane, controller.status()['lane'])
            controller.end()
        controller.begin('fault', lane='exit')
        self.assertEqual('', controller.status()['lane'])
        self.assertEqual('fault', profiles[-1])
        controller.end()
        controller.begin('barrier_open', lane='entrance')
        self.assertEqual('', controller.status()['lane'])
        self.assertFalse(profiles[-1].startswith(('exit_', 'entrance_')))
        controller.end()


if __name__ == '__main__':
    unittest.main()
