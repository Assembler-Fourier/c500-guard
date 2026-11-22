"""The launcher waits for final startup verification; no Windows/task/network/Tk operations run."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_app_shutdown import load_launcher


def completed(request_id, *, ok=True, action="on", errors=None):
    return {"request_id": request_id, "action": action, "ok": ok, "complete": ok,
            "power_state": "on" if ok and action == "on" else "off",
            "completed_at": "2026-11-21T18:00:00Z", "errors": errors or []}


class LauncherStartupCompletionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.app = load_launcher(Path(self.folder.name))
        self.now = 0.0
        self.patch(self.app.time, "time", side_effect=lambda: self.now)
        self.patch(self.app.time, "sleep", side_effect=self.advance)
        self.patch(self.app.subprocess, "run", side_effect=AssertionError("real process operation"))
        self.patch(self.app.subprocess, "Popen", side_effect=AssertionError("real window/process"))
        self.patch(self.app.urllib.request, "urlopen", side_effect=AssertionError("real network access"))
        self.patch(self.app.tk, "Tk", side_effect=AssertionError("real splash window"))
        self.off = self.patch(self.app, "switched_off", return_value=False)
        self.state = self.patch(self.app, "system_state", side_effect=[None] + [{"armed": True}] * 200)
        self.task = self.patch(self.app, "run_task", return_value=True)
        self.status = self.patch(self.app, "task_status", return_value="Ready")
        self.old = completed("previous-start")
        self.result = self.patch(self.app, "receipt", side_effect=[self.old, completed("this-start")])
        self.splash = mock.Mock()
        self.splash.ask.return_value = "close"

    def patch(self, target, name, **kwargs):
        patcher = mock.patch.object(target, name, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def advance(self, seconds):
        self.now += seconds

    def test_existing_paused_service_opens_without_starting_or_arming(self):
        self.state.side_effect = None
        self.state.return_value = {"armed": False}
        self.assertTrue(self.app.start_system(self.splash))
        self.task.assert_not_called()
        self.result.assert_not_called()
        self.splash.ask.assert_not_called()

    def test_armed_during_on_still_waits_for_final_success(self):
        self.status.side_effect = ["Running", "Queued", "Ready"]
        self.assertTrue(self.app.start_system(self.splash))
        self.assertEqual(self.status.call_count, 3)
        self.assertGreaterEqual(self.now, 10)
        self.splash.ask.assert_not_called()

    def test_post_arm_failure_shows_reason_instead_of_opening_the_app(self):
        self.status.side_effect = ["Running", "Ready"]
        self.result.side_effect = [self.old, completed("this-start", ok=False,
                                                     errors=["AI stopped after arming"])]
        self.assertFalse(self.app.start_system(self.splash))
        self.assertEqual(self.splash.ask.call_args.args[0], "Did not start")
        self.assertIn("AI stopped after arming", self.splash.ask.call_args.args[1])

    def test_stale_successful_receipt_cannot_complete_another_start(self):
        self.result.side_effect = None
        self.result.return_value = self.old
        self.assertFalse(self.app.start_system(self.splash))
        self.splash.ask.assert_called_once()

    def test_successful_receipt_needs_a_responding_service(self):
        self.state.side_effect = None
        self.state.return_value = None
        self.assertFalse(self.app.start_system(self.splash))
        self.splash.ask.assert_called_once()

    def test_new_off_receipt_cannot_be_mistaken_for_start_success(self):
        self.result.side_effect = [self.old, completed("user-off", action="off")]
        self.assertFalse(self.app.start_system(self.splash))
        self.splash.ask.assert_called_once()

    def test_full_off_marker_wins_over_success_receipt_and_lingering_state(self):
        self.off.return_value = True
        self.assertFalse(self.app.start_system(self.splash))
        self.assertIn("OFF", self.splash.ask.call_args.args[1])

    def test_failure_can_retry_and_complete_a_new_attempt(self):
        attempt = [0]

        def start(task):
            attempt[0] += 1
            return True

        self.task.side_effect = start
        self.state.side_effect = lambda: {"armed": True} if attempt[0] == 2 else None
        receipts = [self.old, completed("failed", ok=False, errors=["camera not ready"]),
                    completed("retry")]
        self.result.side_effect = lambda: receipts[attempt[0]]
        self.splash.ask.return_value = "retry"
        self.assertTrue(self.app.start_system(self.splash))
        self.assertEqual(self.task.call_count, 2)
        self.splash.ask.assert_called_once()
        self.assertIn("camera not ready", self.splash.ask.call_args.args[1])

    def test_retry_checks_a_new_start_even_if_failed_service_still_answers(self):
        attempt = [0]

        def start(task):
            attempt[0] += 1
            return True

        self.task.side_effect = start
        self.state.side_effect = lambda: {"armed": True} if attempt[0] else None
        receipts = [self.old, completed("failed", ok=False, errors=["input helper failed"]),
                    completed("retry")]
        self.result.side_effect = lambda: receipts[attempt[0]]
        self.splash.ask.return_value = "retry"
        self.assertTrue(self.app.start_system(self.splash))
        self.assertEqual(self.task.call_count, 2)
        self.splash.ask.assert_called_once()

    def test_rejected_windows_task_never_claims_to_start(self):
        self.task.return_value = False
        self.assertFalse(self.app.start_system(self.splash))
        self.assertEqual(self.splash.ask.call_args.args[0], "Could not start")
        self.status.assert_not_called()

    def test_running_task_times_out_with_bounded_fake_clock(self):
        self.status.return_value = "Running"
        self.result.side_effect = None
        self.result.return_value = self.old
        self.assertFalse(self.app.start_system(self.splash))
        self.assertGreaterEqual(self.now, 150)
        self.assertLess(self.now, 152)
        self.splash.ask.assert_called_once()

    def test_keyboard_menu_hint_recovers_when_start_finishes(self):
        self.state.side_effect = [None] + [
            {"frames": 10, "alignment": {"state": "keyboard_menu"}}] * 3 + [{"armed": True}] * 100
        self.assertTrue(self.app.start_system(self.splash))
        titles = [call.args[0] for call in self.splash.say.call_args_list]
        self.assertIn("Make the camera full-screen", titles)
        self.splash.ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
