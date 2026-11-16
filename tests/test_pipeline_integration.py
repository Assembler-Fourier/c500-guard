"""Diagnostics remain available when slow main-loop work owns the state lock."""
import json
import io
import queue
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from test_watch import watch, watcher
from pipeline_health import PipelineHealth


class PipelineIntegrationTests(unittest.TestCase):
    def _run_one_frame(self, *, health_error=False):
        """Run the actual frame loop and barrier wrapper, with every service fake."""
        obj=watcher();clock=[100.0]
        obj.pipeline=PipelineHealth(clock=lambda: clock[0])
        obj.args.web=0;obj.args.black_level=1
        obj.reader=mock.Mock()
        obj.reader.frames=queue.Queue()
        obj.reader.frames.put((100.0,bytes(watch.PROC_H*watch.PROC_W*3)))
        obj.presence=mock.Mock();obj.secondary=mock.Mock()
        obj._update_awake=mock.Mock();obj._note_error=mock.Mock()
        def advance(seconds, value=None):
            clock[0]+=seconds
            return value
        obj.align_frame=lambda frame: advance(.2,frame)
        obj.barrier_workflow=mock.Mock()
        obj.barrier_workflow.observe.side_effect=lambda *a,**kw: advance(.2,[])
        obj.barrier_workflow.status.return_value={}
        obj._observe_arm_angles=lambda *a: advance(.74)
        obj._ack_on_any_arm=mock.Mock();obj._note_arm_memory=mock.Mock()
        def process(frame, received_at):
            advance(.2)
            obj._observe_barrier_service(frame,received_at)
            return True
        obj.process=process
        def health(now):
            advance(.3)
            if health_error:
                raise RuntimeError('fake health failure')
        obj.check_health=health
        obj._submit_analysis=lambda *a,**kw: advance(.3)
        obj.render_preview=lambda *a,**kw: advance(.2)
        obj.paint_status=lambda: setattr(obj,'_running',False)
        with mock.patch.object(watch.time,'monotonic',side_effect=lambda: clock[0]), \
             mock.patch.object(watch.threading.Thread,'start'), \
             mock.patch.object(watch.sys,'stdout',io.StringIO()):
            if health_error:
                with self.assertRaisesRegex(RuntimeError,'fake health failure'):
                    obj.run()
            else:
                obj.run()
        obj._note_error.assert_not_called()
        return obj,obj.pipeline.status()

    def test_actual_frame_loop_records_full_span_and_post_barrier_detection(self):
        obj,report=self._run_one_frame()
        self.assertEqual(report['stages']['detection']['last_ms'],1140.)
        self.assertEqual(report['measurements']['barriers']['last_ms'],200.)
        self.assertEqual(report['measurements']['whole_frame']['last_ms'],2140.)
        self.assertEqual(report['recent_frames'][0]['source_at'],100.)
        self.assertEqual(report['recent_frames'][0]['source_age_ms'],2140.)
        self.assertEqual([row['stage'] for row in report['recent_slow']],['whole_frame'])
        self.assertEqual(report['frames_processed'],1)
        self.assertEqual(report['active_measurements'],[])
        self.assertIsNone(obj.alarm)

    def test_actual_frame_loop_closes_frame_span_on_uncaught_exception(self):
        _,report=self._run_one_frame(health_error=True)
        self.assertEqual(report['recent_frames'][0]['duration_ms'],1640.)
        self.assertEqual(report['active_measurements'],[])
        self.assertEqual(report['current_stage'],'idle')

    def test_slow_whole_frame_without_active_stage_includes_location_only_stack(self):
        obj=watcher();clock=[100.]
        obj.pipeline=PipelineHealth(clock=lambda: clock[0])
        with obj._pipeline_frame(100.):
            clock[0]+=2.14
            report=obj.diagnostics()
            self.assertEqual(report['current_stage'],'idle')
            self.assertTrue(report['measurement_slow'])
            self.assertTrue(report['main_stack'])
            self.assertLessEqual(len(report['main_stack']),12)
            self.assertTrue(all(set(row)=={'file','function','line'} for row in report['main_stack']))

    def test_diagnostics_does_not_wait_for_the_main_state_lock(self):
        obj=watcher();server=watch.start_web(obj,0)
        url='http://127.0.0.1:'+str(server.server_address[1])
        try:
            with obj._state_lock:
                with urllib.request.urlopen(url+'/diagnostics',timeout=1.0) as response:
                    data=json.load(response)
                    self.assertEqual(response.headers['Cache-Control'],'no-store')
                self.assertIn('current_stage',data)
                self.assertIn('capture',data)
        finally:
            server.shutdown();server.server_close()

    def test_diagnostics_retains_the_local_host_boundary(self):
        obj=watcher();server=watch.start_web(obj,0)
        url='http://127.0.0.1:'+str(server.server_address[1])
        try:
            request=urllib.request.Request(url+'/diagnostics',headers={'Host':'example.org'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request,timeout=1.0)
            self.assertEqual(error.exception.code,403)
        finally:
            server.shutdown();server.server_close()

    def test_decoder_progress_is_distinct_from_last_processed_frame(self):
        obj=watcher();now=time.monotonic()
        obj.pipeline.frame_received(now-10)
        obj.pipeline.frame_processed(now-10)
        obj.reader._progress_at=now;obj.reader._got_frame=True
        report=obj.diagnostics()
        self.assertLess(report['capture']['last_frame_age_s'],1)
        self.assertGreaterEqual(report['last_processed_age_s'],10)

    def test_watchdog_runs_while_periodic_speaker_query_is_blocked(self):
        obj=watcher();entered=threading.Event();release=threading.Event()
        def blocked_read():
            entered.set();release.wait(3)
            return {'available':False}
        obj._read_audio=blocked_read
        obj._watchdog_expired=mock.Mock(return_value=True)
        reader=threading.Thread(target=obj._audio_health_loop,daemon=True)
        reader.start()
        try:
            self.assertTrue(entered.wait(1))
            watchdog=threading.Thread(target=obj._health_loop,daemon=True)
            watchdog.start();watchdog.join(1)
            self.assertFalse(watchdog.is_alive())
            obj._watchdog_expired.assert_called_once()
            self.assertTrue(reader.is_alive())
        finally:
            obj._running=False;release.set();reader.join(2)
        self.assertFalse(reader.is_alive())

    def test_expired_candidate_is_rejected_before_visit_or_audio_mutation(self):
        obj=watcher();obj.visits=mock.Mock();obj.loudness=mock.Mock()
        stamp=time.monotonic()-4
        accepted=obj.raise_alarm('main','old car',details={'source':'motion','frame_received_at':stamp})
        self.assertFalse(accepted)
        obj.visits.consider.assert_not_called()
        obj.loudness.begin.assert_not_called()
        self.assertIsNone(obj.alarm)


if __name__=='__main__':unittest.main()
