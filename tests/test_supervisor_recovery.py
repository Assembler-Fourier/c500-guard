"""Only sustained unresponsiveness may restart a verified watcher."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import service


class SupervisorRecoveryTests(unittest.TestCase):
    def run_checks(self, states, stop_result=0, paused=False):
        states=list(states)
        with mock.patch.object(service.ctl,'PAUSED') as pause, \
             mock.patch.object(service.ctl,'WATCHER_PID') as pid, \
             mock.patch.object(service.ctl,'watcher_state',side_effect=states), \
             mock.patch.object(service.ctl,'cmd_start',return_value=1) as start, \
             mock.patch.object(service.ctl,'stop_watcher',return_value=stop_result) as stop, \
             mock.patch.object(service.ctl,'ensure_bridge',return_value=True), \
             mock.patch.object(service.time,'sleep',side_effect=[None]*(len(states)-1)+[KeyboardInterrupt]), \
             mock.patch('builtins.print'):
            pause.exists.return_value=paused
            pid.exists.return_value=True
            with self.assertRaises(KeyboardInterrupt):
                service.supervise()
            return start.call_count,stop.call_count

    def test_three_failed_checks_recover_owned_process(self):
        self.assertEqual(self.run_checks([None,None,None]),(3,1))

    def test_good_response_resets_failure_count(self):
        self.assertEqual(self.run_checks([None,None,{'armed':False},None]),(3,0))

    def test_unknown_process_is_never_replaced(self):
        self.assertEqual(self.run_checks([None,None,None],stop_result=1),(2,1))

    def test_stalled_detector_is_stopped_after_two_checks(self):
        stalled={'armed':True,'loop_age_s':60}
        self.assertEqual(self.run_checks([stalled,stalled]),(0,1))

    def test_brief_loop_delay_or_old_watcher_is_not_a_stall(self):
        self.assertEqual(self.run_checks([{'armed':True,'loop_age_s':3},{'armed':True},{'armed':True,'loop_age_s':True}]),(0,0))

    def test_paused_supervisor_does_not_restart(self):
        self.assertEqual(self.run_checks([None,None,None],paused=True),(0,0))


if __name__=='__main__': unittest.main()
