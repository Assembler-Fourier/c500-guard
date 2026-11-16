"""Pure timing and concurrency tests: no live watcher, network, device or I/O."""
import json
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline_health import PipelineHealth


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class PipelineHealthTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.health = PipelineHealth(clock=self.clock)

    def test_empty_status_is_explicitly_unknown_not_fresh(self):
        s = self.health.status()
        self.assertEqual(s['current_stage'], 'idle')
        self.assertIsNone(s['last_received_age_s'])
        self.assertIsNone(s['last_processed_age_s'])
        self.assertIsNone(s['since_last_processed_s'])
        self.assertIsNone(s['stage_id'])
        self.assertEqual(s['frames_processed'], 0)
        self.assertEqual(s['stages'], {})
        json.dumps(s, allow_nan=False)

    def test_stage_latest_and_max_duration_do_not_get_confused(self):
        token = self.health.enter('alignment')
        self.clock.now += .4
        self.assertEqual(self.health.status()['stage_age_s'], .4)
        self.assertTrue(self.health.leave(token))
        self.clock.now += 1
        token = self.health.enter('alignment')
        self.clock.now += .1
        self.health.leave(token)
        s = self.health.status()
        self.assertEqual(s['stages']['alignment'], {'last_ms': 100., 'max_ms': 400., 'count': 2})
        self.assertEqual(s['current_stage'], 'idle')
        self.assertEqual(s['recent_slow'], [])

    def test_active_slow_stage_visible_before_work_returns(self):
        token = self.health.enter('alignment')
        self.clock.now += 2
        s = self.health.status()
        self.assertTrue(s['active_slow'])
        self.assertTrue(s['stage_slow'])
        self.assertEqual(s['stage_id'], token)
        self.assertEqual(s['current_stage_started_at'], 100.)
        self.assertEqual(s['stage_age_s'], 2.)
        self.assertEqual(s['recent_slow'], [])
        self.health.leave(token)
        s = self.health.status()
        self.assertFalse(s['active_slow'])
        self.assertEqual(s['recent_slow'][0]['duration_ms'], 2000.)

    def test_new_stage_completes_old_span_and_late_leave_cannot_clear_it(self):
        first = self.health.enter('alignment')
        self.clock.now += .2
        second = self.health.enter('detection')
        self.assertFalse(self.health.leave(first))
        s = self.health.status()
        self.assertEqual(s['current_stage'], 'detection')
        self.assertEqual(s['stage_id'], second)
        self.assertEqual(s['stages']['alignment']['last_ms'], 200.)
        self.assertFalse(self.health.leave(True))
        self.assertFalse(self.health.leave('detection'))
        self.assertTrue(self.health.leave(second))

    def test_context_records_span_and_releases_stage_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, 'frame failed'):
            with self.health.stage('detection'):
                self.clock.now += .75
                raise RuntimeError('frame failed')
        s = self.health.status()
        self.assertEqual(s['current_stage'], 'idle')
        self.assertEqual(s['stages']['detection']['last_ms'], 750.)

    def test_receiver_continues_while_processed_frame_is_old(self):
        self.health.frame_received(98.)
        self.health.frame_processed(98.)
        self.clock.now = 103.
        self.health.frame_received(102.9)
        s = self.health.status()
        self.assertEqual(s['last_received_age_s'], .1)
        self.assertEqual(s['last_processed_age_s'], 5.)
        self.assertEqual(s['since_last_processed_s'], 3.)
        self.assertEqual(s['last_processing_delay_ms'], 2000.)
        self.assertEqual(s['frames_received'], 2)
        self.assertEqual(s['frames_processed'], 1)

    def test_slow_completion_does_not_make_an_old_source_frame_fresh(self):
        self.health.frame_received(100.)
        self.clock.now = 116.
        self.health.frame_processed(100.)
        s = self.health.status()
        self.assertEqual(s['since_last_processed_s'], 0.)
        self.assertEqual(s['last_processed_age_s'], 16.)
        self.assertEqual(s['last_processing_delay_ms'], 16000.)

    def test_invalid_frame_stamps_are_rejected_without_erasing_good_state(self):
        self.health.frame_received(99.)
        self.health.frame_processed(99.)
        invalid = [None, True, False, '99', -1, float('nan'), float('inf'), 101., 10**1000]
        for stamp in invalid:
            with self.subTest(stamp=repr(stamp)[:50]):
                self.assertFalse(self.health.frame_received(stamp))
                self.assertFalse(self.health.frame_processed(stamp))
        s = self.health.status()
        self.assertEqual(s['invalid_timestamps'], len(invalid) * 2)
        self.assertEqual(s['last_received_at'], 99.)
        self.assertEqual(s['last_processed_at'], 99.)
        self.assertEqual(s['frames_processed'], 1)
        json.dumps(s, allow_nan=False)

    def test_out_of_order_thread_results_do_not_move_freshness_backwards(self):
        self.health.frame_received(99.)
        self.health.frame_processed(99.)
        self.assertFalse(self.health.frame_received(98.))
        self.assertFalse(self.health.frame_processed(98.))
        s = self.health.status()
        self.assertEqual(s['last_received_age_s'], 1.)
        self.assertEqual(s['last_processed_age_s'], 1.)
        self.assertEqual(s['out_of_order_frames'], 2)

    def test_zero_monotonic_stamp_is_valid(self):
        self.clock.now = 0.
        self.assertTrue(self.health.frame_received(0.))
        self.assertTrue(self.health.frame_processed(0.))
        self.assertEqual(self.health.status()['last_received_age_s'], 0.)

    def test_stage_and_slow_record_storage_is_bounded(self):
        health = PipelineHealth(clock=self.clock, max_stages=3, max_slow=2)
        for i in range(12):
            with health.stage(f'stage-{i}'):
                self.clock.now += 2.5
        s = health.status()
        self.assertEqual(len(s['stages']), 3)
        self.assertEqual(len(s['recent_slow']), 2)
        self.assertEqual([r['stage'] for r in s['recent_slow']], ['stage-10', 'stage-11'])
        self.assertEqual(s['recent_slow'][0]['age_s'], 2.5)
        token = health.enter('x' * 1000)
        self.assertEqual(len(health.status()['current_stage']), 64)
        health.leave(token)

    def test_returned_snapshots_cannot_mutate_internal_diagnostics(self):
        with self.health.stage('alignment'):
            self.clock.now += 3
        s = self.health.status()
        s['stages']['alignment']['count'] = -50
        s['recent_slow'][0]['duration_ms'] = -40
        s['recent_slow'].clear()
        current = self.health.status()
        self.assertEqual(current['stages']['alignment']['count'], 1)
        self.assertEqual(current['recent_slow'][0]['duration_ms'], 3000.)

    def test_drop_counter_rejects_malformed_values(self):
        self.assertTrue(self.health.record_stale_drop())
        self.assertTrue(self.health.record_stale_drop(3))
        for value in [None, True, False, -1, 0, .5, float('nan'), float('inf'), '1', 2**32]:
            self.assertFalse(self.health.record_stale_drop(value))
        self.assertEqual(self.health.status()['stale_dropped'], 4)

    def test_backward_or_invalid_clock_keeps_output_finite_and_nonnegative(self):
        token = self.health.enter('alignment')
        self.clock.now = 101.
        self.assertEqual(self.health.status()['stage_age_s'], 1.)
        for value in [99., float('nan'), float('inf'), None, True, -1.]:
            self.clock.now = value
            s = self.health.status()
            self.assertEqual(s['stage_age_s'], 1.)
            json.dumps(s, allow_nan=False)
        self.assertTrue(self.health.leave(token))
        self.assertGreater(self.health.status()['clock_errors'], 0)

    def test_throwing_injected_clock_cannot_raise_from_diagnostics(self):
        def broken():
            raise RuntimeError('clock failed')
        health = PipelineHealth(clock=broken)
        with health.stage('alignment'):
            self.assertTrue(health.frame_received(0.))
        self.assertGreater(health.status()['clock_errors'], 0)
        json.dumps(health.status(), allow_nan=False)

    def test_context_does_not_hold_telemetry_lock_during_blocked_work(self):
        entered, release, read_done = threading.Event(), threading.Event(), threading.Event()
        snapshots = []

        def blocked_main_loop():
            with self.health.stage('alignment'):
                entered.set()
                release.wait(3)

        def diagnostic_reader():
            snapshots.append(self.health.status())
            self.health.frame_received(100.)
            self.health.record_stale_drop()
            read_done.set()

        worker = threading.Thread(target=blocked_main_loop)
        reader = threading.Thread(target=diagnostic_reader)
        try:
            worker.start()
            self.assertTrue(entered.wait(2))
            reader.start()
            self.assertTrue(read_done.wait(2), 'diagnostics blocked behind stage work')
            self.assertEqual(snapshots[0]['current_stage'], 'alignment')
        finally:
            release.set()
            worker.join(3)
            if reader.ident is not None:
                reader.join(3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(reader.is_alive())

    def test_invalid_configuration_fails_at_construction(self):
        for value in [True, None, 0, -1, float('inf'), float('nan'), '2']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PipelineHealth(slow_after_s=value)
        for value in [True, 0, -1, 129, 1.2, '5']:
            for name in ['max_slow', 'max_stages']:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    PipelineHealth(**{name: value})
        for value in [None, '', '   ', 1]:
            with self.assertRaises(ValueError):
                self.health.enter(value)

    def test_independent_barrier_span_preserves_complete_detection_and_source_frame(self):
        with self.health.frame_span(99.95):
            with self.health.stage('detection') as outer:
                stage_id=self.health.status()['stage_id']
                self.clock.now=100.2
                with self.health.measure('barriers',source_at=99.95):
                    self.clock.now=100.4
                    self.assertEqual(stage_id,self.health.status()['stage_id'])
                self.clock.now=102.14
                self.assertTrue(self.health.status()['measurement_slow'])
                self.assertTrue(self.health.status()['active_slow'])
        result=self.health.status()
        self.assertEqual(2140.,result['stages']['detection']['last_ms'])
        self.assertEqual(200.,result['measurements']['barriers']['last_ms'])
        self.assertEqual(2140.,result['measurements']['whole_frame']['last_ms'])
        frame=result['recent_frames'][0]
        self.assertEqual(99.95,frame['source_at'])
        self.assertEqual(2190.,frame['source_age_ms'])
        self.assertEqual(['detection','whole_frame'],[r['stage'] for r in result['recent_slow']])
        self.assertEqual([],result['active_measurements'])

    def test_measurement_tokens_cannot_close_stages_or_each_other_and_allow_reentrance(self):
        stage=self.health.enter('detection')
        first=self.health.begin_measurement('part')
        self.clock.now=100.1
        second=self.health.begin_measurement('part')
        self.assertFalse(self.health.leave(first))
        self.assertFalse(self.health.end_measurement(stage))
        self.assertFalse(self.health.end_measurement(True))
        self.assertFalse(self.health.end_measurement('part'))
        self.clock.now=100.3
        self.assertTrue(self.health.end_measurement(first))
        self.assertFalse(self.health.end_measurement(first))
        self.assertEqual(stage,self.health.status()['stage_id'])
        self.clock.now=100.5
        self.assertTrue(self.health.end_measurement(second))
        self.assertEqual({'last_ms':400.,'max_ms':400.,'count':2},
                         self.health.status()['measurements']['part'])
        self.assertTrue(self.health.leave(stage))

    def test_exception_closes_whole_frame_and_measurement_without_hiding_error(self):
        with self.assertRaisesRegex(RuntimeError,'failed'):
            with self.health.frame_span(100.):
                with self.health.measure('diagnostic-write'):
                    self.clock.now=102.14
                    raise RuntimeError('failed')
        report=self.health.status()
        self.assertEqual([],report['active_measurements'])
        self.assertEqual(2140.,report['recent_frames'][0]['duration_ms'])
        self.assertEqual(2,len(report['recent_slow']))

    def test_measurement_storage_active_and_completed_is_bounded(self):
        health=PipelineHealth(clock=self.clock,max_stages=2,max_slow=2)
        tokens=[health.begin_measurement(str(i)) for i in range(4)]
        self.assertEqual([None,None],tokens[2:])
        self.assertEqual(2,len(health.status()['active_measurements']))
        self.assertEqual(2,health.status()['measurements_dropped'])
        for token in tokens:health.end_measurement(token)
        for i in range(8):
            with health.measure(str(i)):
                self.clock.now+=.01
            with health.frame_span(self.clock.now):
                self.clock.now+=2.1
        report=health.status()
        self.assertLessEqual(len(report['measurements']),2)
        self.assertEqual(2,len(report['recent_frames']))
        self.assertEqual(2,len(report['recent_slow']))

    def test_invalid_source_and_clock_never_produce_nonfinite_measurements(self):
        for value in (True,'100',-1,float('nan'),float('inf'),101.,10**1000):
            with self.health.frame_span(value):
                pass
            self.assertIsNone(self.health.status()['recent_frames'][-1]['source_at'])
        token=self.health.begin_measurement('part',source_at=99.)
        self.clock.now=101.
        self.health.status()
        for value in (99.,float('nan'),None,True):
            self.clock.now=value
            json.dumps(self.health.status(),allow_nan=False)
        self.assertTrue(self.health.end_measurement(token))
        self.assertEqual(1000.,self.health.status()['measurements']['part']['last_ms'])
        json.dumps(self.health.status(),allow_nan=False)

    def test_measurement_snapshots_are_copies_and_invalid_names_rejected(self):
        for name in (None,True,'','  '):
            with self.assertRaises(ValueError):self.health.begin_measurement(name)
        with self.health.frame_span(100.):
            active=self.health.status()['active_measurements']
            active[0]['started_at']=-50
            self.assertEqual(100.,self.health.status()['active_measurements'][0]['started_at'])
            self.clock.now=102.2
        report=self.health.status()
        report['measurements']['whole_frame']['count']=-1
        report['recent_frames'][0]['source_at']=-1
        report['recent_slow'][0]['duration_ms']=-1
        saved=self.health.status()
        self.assertEqual(1,saved['measurements']['whole_frame']['count'])
        self.assertEqual(100.,saved['recent_frames'][0]['source_at'])
        self.assertEqual(2200.,saved['recent_slow'][0]['duration_ms'])

    def test_independent_context_does_not_hold_lock_during_work(self):
        ready,release=threading.Event(),threading.Event()
        def work():
            with self.health.frame_span(100.):
                ready.set();release.wait(2)
        thread=threading.Thread(target=work)
        thread.start()
        try:
            self.assertTrue(ready.wait(1))
            self.assertEqual(1,len(self.health.status()['active_measurements']))
            self.assertTrue(self.health.frame_received(100.))
        finally:
            release.set();thread.join(2)
        self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
