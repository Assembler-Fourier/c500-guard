"""Current validated warnings precede optional model/evidence work.

These are bounded blocking-injection tests with fake camera/audio. Advancing
the source clock proves stale work is dropped without swallowing the alarm
which was already valid before the injected block.
"""
from pathlib import Path
import tempfile
import threading
import unittest

import test_entrance_headlights as fixtures
from vehicle_sightings import VehicleSightings


class LightAlarmPriorityTests(unittest.TestCase):
    setUp = fixtures.HeadlightAlarmTests.setUp
    deliver = fixtures.HeadlightAlarmTests.deliver

    def prepare_light(self, trigger='headlights'):
        self.lane.trigger = trigger
        if trigger == 'taillight':
            self.lane.route_group = 'campus-exit'
            self.lane.route_role = 'exit'
        self.deliver(self.road, 2)
        lit = fixtures.lamp_pair(self.road) if trigger == 'headlights' else fixtures.lamp_pair(
            self.road, value=(255., 170., 70.), r=.035)
        self.deliver(lit, 4)
        self.assertIsNone(self.obj.alarm)
        return lit

    def enter_blocked_frame(self, frame, install):
        entered, release = threading.Event(), threading.Event()
        result, failures = [], []
        def block(*args, **kwargs):
            entered.set()
            if not release.wait(3.):
                raise RuntimeError('test did not release its injected block')
            return []
        install(block)
        self.now += .1
        stamp = self.now
        def run():
            try:
                result.append(self.obj.process(frame, received_at=stamp))
            except BaseException as error:
                failures.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(2.), 'the injected operation was not reached')
            self.assertEqual('main', self.obj.alarm)
            self.assertEqual('main', self.obj.buzzer.want)
            ident = self.obj.alarm_id
            self.now += 3.2
        finally:
            release.set()
            thread.join(3.)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], failures)
        self.assertEqual([False], result, 'late frame should still be rejected')
        self.assertEqual(ident, self.obj.alarm_id)
        self.assertEqual('main', self.obj.alarm)

    def test_valid_headlights_sound_before_a_slow_semantic_consumer(self):
        lit = self.prepare_light()
        self.enter_blocked_frame(lit, lambda block: setattr(self.obj.semantic.update, 'side_effect', block))
        self.assertTrue(self.lane._notified['light'])

    def test_valid_tail_lights_sound_before_a_slow_semantic_consumer(self):
        lit = self.prepare_light('taillight')
        self.enter_blocked_frame(lit, lambda block: setattr(self.obj.semantic.update, 'side_effect', block))
        self.assertTrue(self.lane._notified['light'])

    def install_archive_block(self, block, *, async_writes=False):
        scratch = tempfile.TemporaryDirectory(prefix='sighting-order-')
        self.addCleanup(scratch.cleanup)
        sightings = VehicleSightings(Path(scratch.name), clock=lambda: self.now,
                                     async_writes=async_writes)
        # A previous track has just expired and is ready for archival. Its
        # actual _finish->_write call must follow the current alarm dispatch.
        sightings._tracks[999] = dict(last=self.now-5, roles={'exit'}, observations=[{}], crops=[])
        sightings._write = block
        self.obj.sightings = sightings
        self.obj.semantic.status.return_value['observed_at'] = self.now
        self.obj.semantic.frame_at.return_value = self.road

    def check_async_archive(self, frame):
        entered, release = threading.Event(), threading.Event()
        def blocked_write(*_args, **_kwargs):
            entered.set()
            if not release.wait(3.):
                raise RuntimeError('test did not release its injected archive block')
        self.install_archive_block(blocked_write, async_writes=True)
        sightings = self.obj.sightings
        self.now += .1
        try:
            self.assertTrue(self.obj.process(frame, received_at=self.now))
            self.assertTrue(entered.wait(2.))
            self.assertFalse(release.is_set())
            self.assertEqual('main', self.obj.buzzer.want)
            self.assertEqual('ai', self.obj.alarm_details['source'])
            self.assertTrue(self.obj.loudness.evidence.call_args.kwargs['confirmed'])
            # Another fresh frame still completes while the archive is blocked.
            self.now += .1
            self.assertTrue(self.obj.process(frame, received_at=self.now))
        finally:
            release.set()
            sightings.flush(timeout=0.)
            if sightings._writer is not None:
                sightings._writer.join(2.)
                self.assertFalse(sightings._writer.is_alive())

    def test_light_alarm_precedes_a_slow_previous_vehicle_archive(self):
        lit = self.prepare_light()
        self.enter_blocked_frame(lit, self.install_archive_block)
        self.assertEqual('light', self.obj.alarm_details['source'])

    def install_ai_event(self):
        self.obj.semantic.update.return_value = [('main', 'Entrance waiting: car', 'ai-test')]
        self.obj.semantic.event_context.side_effect = lambda _event: {
            'zone_name': 'Entrance waiting', 'route_group': 'campus-entrance',
            'route_role': 'entrance', 'alert_stage': 'barrier', 'label': 'car',
            'track_id': 42, 'observed_at': self.now,
            'bbox_normalized': [.7, .35, .83, .5]}

    def test_current_ai_alarm_precedes_a_slow_previous_vehicle_archive(self):
        self.deliver(self.road, 2)
        self.install_ai_event()
        self.enter_blocked_frame(self.road, self.install_archive_block)
        self.assertEqual('ai', self.obj.alarm_details['source'])
        self.obj.semantic.mark_emitted.assert_called_once_with('ai-test')

    def test_light_and_ai_in_one_frame_create_only_one_alarm(self):
        lit = self.prepare_light()
        self.install_ai_event()
        before = self.obj._alarm_sequence
        self.deliver(lit)
        self.assertEqual(before+1, self.obj._alarm_sequence)
        self.assertTrue(self.lane._notified['light'])
        self.obj.semantic.mark_emitted.assert_called_once_with('ai-test')

    def test_async_archive_keeps_current_ai_alarm_and_later_frame_running(self):
        self.deliver(self.road, 2)
        self.install_ai_event()
        before = self.obj._alarm_sequence
        self.check_async_archive(self.road)
        self.assertEqual(before+1, self.obj._alarm_sequence)

    def test_async_archive_preserves_same_frame_light_ai_visit_and_confirmation(self):
        lit = self.prepare_light()
        self.install_ai_event()
        before = self.obj._alarm_sequence
        self.check_async_archive(lit)
        self.assertEqual(before+1, self.obj._alarm_sequence)
        self.assertTrue(self.lane._notified['light'])

    def test_expired_input_never_becomes_a_new_priority_light_alarm(self):
        lit = self.prepare_light()
        self.assertFalse(self.obj.process(lit, received_at=self.now-3.2))
        self.assertIsNone(self.obj.alarm)
        self.assertIsNone(self.obj.buzzer.want)


if __name__ == '__main__':
    unittest.main()
