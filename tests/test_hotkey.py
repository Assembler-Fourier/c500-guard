"""Mocked Windows API checks. These tests never install a hook or inject keys."""
import ctypes
import io
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hotkey_win as hk


@unittest.skipUnless(sys.platform.startswith("win"), "Windows ctypes ABI tests")
class HotkeyTests(unittest.TestCase):
    def setUp(self):
        self.user32, self.kernel32 = MagicMock(), MagicMock()
        self.user32.SetWindowsHookExW.return_value = 0x123456789ABC
        self.user32.CallNextHookEx.return_value = 0x123456789ABC
        self.user32.GetAsyncKeyState.return_value = 0x8000
        self.kernel32.GetModuleHandleW.return_value = 0xABCDEF123456
        self.dll_patch = patch.object(ctypes, "WinDLL", side_effect=[self.user32, self.kernel32])
        self.dll_patch.start()
        self.addCleanup(self.dll_patch.stop)
        self.hotkey = hk.Hotkey(hk.DEFAULT_API)
        # Even lifecycle tests cannot read or modify the real watcher.
        self.poll_patch = patch.object(self.hotkey, "_refresh_alarm_status", return_value=(None, None))
        self.poll_patch.start()
        self.addCleanup(self.poll_patch.stop)
        self.key = self.hotkey._KBD(vkCode=hk.VK_F4)
        self.pointer = ctypes.addressof(self.key)
        self.mouse = self.hotkey._MOUSE()
        self.mouse_pointer = ctypes.addressof(self.mouse)

    def tearDown(self):
        self.hotkey._stop.set()
        self.hotkey._input_wake.set()
        for worker in (self.hotkey._worker, self.hotkey._input_worker):
            if worker.is_alive():
                worker.join(timeout=1)

    def press(self, message=hk.WM_SYSKEYDOWN):
        return self.hotkey._on_key(hk.HC_ACTION, message, self.pointer)

    def test_x64_handles_and_results_have_pointer_sized_signatures(self):
        self.assertEqual(ctypes.sizeof(self.user32.SetWindowsHookExW.restype), ctypes.sizeof(ctypes.c_void_p))
        self.assertEqual(ctypes.sizeof(self.user32.CallNextHookEx.restype), ctypes.sizeof(ctypes.c_void_p))
        self.assertEqual(ctypes.sizeof(self.user32.DispatchMessageW.restype), ctypes.sizeof(ctypes.c_void_p))
        self.assertEqual(ctypes.sizeof(self.hotkey._proc._restype_), ctypes.sizeof(ctypes.c_void_p))
        self.assertEqual(len(self.user32.SetWindowsHookExW.argtypes), 4)
        self.assertEqual(len(self.user32.GetMessageW.argtypes), 4)
        self.assertEqual(len(self.user32.UnhookWindowsHookEx.argtypes), 1)
        self.assertEqual(self.hotkey._KBD._fields_[-1][1], ctypes.c_size_t)
        self.assertEqual(self.hotkey._MOUSE._fields_[-1][1], ctypes.c_size_t)
        self.assertEqual(ctypes.sizeof(self.hotkey._mouse_proc._restype_), ctypes.sizeof(ctypes.c_void_p))

    def test_callback_only_enqueues_and_swallows_chord(self):
        with patch.object(self.hotkey, "toggle") as toggle, patch.object(self.hotkey, "say") as say, \
                patch.object(hk.urllib.request, "urlopen") as http:
            self.assertEqual(self.press(), hk.SWALLOW)
            self.assertTrue(self.hotkey._requests.get_nowait())
            toggle.assert_not_called()
            say.assert_not_called()
            http.assert_not_called()
        self.user32.CallNextHookEx.assert_not_called()

    def test_long_held_repeat_and_keyup_are_swallowed_once(self):
        with patch.object(hk.time, "monotonic", side_effect=[1, 8]):
            self.assertEqual(self.press(), hk.SWALLOW)
            self.assertEqual(self.press(), hk.SWALLOW)
            self.user32.GetAsyncKeyState.return_value = 0
            self.assertEqual(self.press(), hk.SWALLOW)
            self.assertEqual(self.press(hk.WM_SYSKEYUP), hk.SWALLOW)
        self.assertEqual(self.hotkey._requests.qsize(), 1)
        self.assertFalse(self.hotkey._f4_down)

    def test_pending_http_and_debounce_do_not_enqueue_delayed_second_toggle(self):
        with patch.object(hk.time, "monotonic", side_effect=[1, 1.1, 8]):
            self.press()
            self.press(hk.WM_KEYUP)
            self.press()
            self.press(hk.WM_KEYUP)
            self.press()
        self.assertEqual(self.hotkey._requests.qsize(), 1)

    def test_unrelated_key_and_negative_hook_code_pass_through(self):
        self.key.vkCode = 0x41
        self.assertEqual(self.press(), 0x123456789ABC)
        self.hotkey._on_key(-1, hk.WM_KEYDOWN, 0)
        self.assertEqual(self.user32.CallNextHookEx.call_count, 2)
        self.assertTrue(self.hotkey._requests.empty())

    def test_http_and_beep_run_on_daemon_worker(self):
        callback_thread = threading.get_ident()
        request_threads, beep_threads = [], []
        completed = threading.Event()

        def request(*args, **kwargs):
            request_threads.append(threading.get_ident())
            return io.BytesIO(b'{"armed":true}')

        def beep(armed):
            beep_threads.append(threading.get_ident())
            completed.set()

        with patch.object(hk.urllib.request, "urlopen", side_effect=request), \
                patch.object(self.hotkey, "beep", side_effect=beep):
            self.hotkey._worker.start()
            self.assertEqual(self.press(), hk.SWALLOW)
            self.assertTrue(completed.wait(1), "toggle worker did not process request")
        self.assertEqual(len(request_threads), 1)
        self.assertNotEqual(request_threads[0], callback_thread)
        self.assertEqual(request_threads, beep_threads)
        self.assertTrue(self.hotkey._worker.daemon)

    def test_message_error_unhooks_and_returns_failure(self):
        self.user32.GetMessageW.return_value = -1
        with patch.object(self.hotkey, "say"):
            self.assertEqual(self.hotkey.run(), 1)
        self.assertEqual(self.user32.UnhookWindowsHookEx.call_count, 2)
        self.user32.TranslateMessage.assert_not_called()
        self.assertTrue(self.hotkey._stop.is_set())

    def test_normal_quit_unhooks_and_preserves_module_handle(self):
        self.user32.GetMessageW.return_value = 0
        self.assertEqual(self.hotkey.run(), 0)
        self.assertEqual(self.user32.SetWindowsHookExW.call_args_list, [
            call(hk.WH_KEYBOARD_LL, self.hotkey._proc, 0xABCDEF123456, 0),
            call(hk.WH_MOUSE_LL, self.hotkey._mouse_proc, 0xABCDEF123456, 0)])
        self.assertEqual(self.user32.UnhookWindowsHookEx.call_count, 2)
        self.assertIsNone(self.hotkey._hook)
        self.assertIsNone(self.hotkey._mouse_hook)

    def test_mouse_hook_failure_cleans_up_keyboard_without_starting_workers(self):
        self.user32.SetWindowsHookExW.side_effect = [0x123456789ABC, 0]
        with patch.object(self.hotkey, "say"):
            self.assertEqual(self.hotkey.run(), 1)
        self.user32.UnhookWindowsHookEx.assert_called_once_with(0x123456789ABC)
        self.assertIsNone(self.hotkey._worker.ident)
        self.assertIsNone(self.hotkey._input_worker.ident)
        self.assertTrue(self.hotkey._stop.is_set())

    def test_keyboard_hook_failure_does_not_install_mouse_or_start_workers(self):
        self.user32.SetWindowsHookExW.return_value = 0
        with patch.object(self.hotkey, "say"):
            self.assertEqual(self.hotkey.run(), 1)
        self.assertEqual(self.user32.SetWindowsHookExW.call_count, 1)
        self.user32.UnhookWindowsHookEx.assert_not_called()
        self.assertIsNone(self.hotkey._worker.ident)
        self.assertIsNone(self.hotkey._input_worker.ident)

    def test_timer_reregisters_hooks_new_before_old_and_cleans_up(self):
        handles = iter([0x1001, 0x1002, 0x2001, 0x2002])
        self.user32.SetWindowsHookExW.side_effect = lambda *args: next(handles)
        self.user32.SetTimer.return_value = 77
        calls = []

        def get_message(pointer, *args):
            calls.append(True)
            if len(calls) == 1:
                pointer._obj.message = hk.WM_TIMER
                return 1
            return 0

        self.user32.GetMessageW.side_effect = get_message
        with patch.object(self.hotkey, "say"):
            self.assertEqual(self.hotkey.run(), 0)
        self.assertEqual(self.user32.SetTimer.call_args.args[2], hk.HOOK_REFRESH_MS)
        # Each fresh hook is installed before its predecessor is removed.
        self.assertEqual(self.user32.UnhookWindowsHookEx.call_args_list,
                         [call(0x1001), call(0x1002), call(0x2002), call(0x2001)])
        self.user32.DispatchMessageW.assert_not_called()
        self.user32.KillTimer.assert_called_once_with(None, 77)

    def test_failed_hook_refresh_keeps_working_hook(self):
        self.hotkey._hook, self.hotkey._mouse_hook = 0x1001, 0x1002
        self.user32.SetWindowsHookExW.return_value = 0
        with patch.object(self.hotkey, "say") as say:
            self.hotkey._refresh_hooks(0xABCDEF123456)
        self.assertEqual((self.hotkey._hook, self.hotkey._mouse_hook), (0x1001, 0x1002))
        self.user32.UnhookWindowsHookEx.assert_not_called()
        self.assertEqual(say.call_count, 2)

    def cache_alarm(self, alarm_id="alarm-a"):
        self.hotkey._alarm_snapshot = (alarm_id, time.monotonic())

    def test_physical_key_acknowledges_observed_alarm_without_swallow_or_io(self):
        self.cache_alarm()
        self.key.vkCode = 0x41
        with patch.object(hk.urllib.request, "urlopen") as http, \
                patch.object(self.hotkey, "say") as say, patch.object(self.hotkey, "beep") as beep:
            self.assertEqual(self.press(hk.WM_KEYDOWN), 0x123456789ABC)
            self.assertEqual(self.hotkey._ack_requests.get_nowait()[:2], ("alarm-a", "input-any-key"))
            http.assert_not_called()
            say.assert_not_called()
            beep.assert_not_called()
        self.assertTrue(self.hotkey._requests.empty())

    def test_keyup_injected_key_and_no_alarm_do_not_acknowledge(self):
        self.key.vkCode = 0x41
        self.press(hk.WM_KEYDOWN)
        self.assertEqual(len(self.hotkey._fallback_inputs), 1)
        self.hotkey._fallback_inputs.clear()
        self.cache_alarm()
        self.press(hk.WM_KEYUP)
        for flags in (hk.LLKHF_INJECTED, hk.LLKHF_LOWER_IL_INJECTED):
            self.key.flags = flags
            self.press(hk.WM_KEYDOWN)
        self.assertTrue(self.hotkey._ack_requests.empty())
        self.assertFalse(self.hotkey._fallback_inputs)
        self.assertEqual(self.user32.CallNextHookEx.call_count, 4)

    def test_all_mouse_button_downs_acknowledge_and_pass_through(self):
        for number, message in enumerate(hk.MOUSE_BUTTON_DOWN):
            with self.subTest(message=message):
                self.cache_alarm(f"alarm-{number}")
                result = self.hotkey._on_mouse(hk.HC_ACTION, message, self.mouse_pointer)
                self.assertEqual(result, 0x123456789ABC)
                self.assertEqual(self.hotkey._ack_requests.get_nowait()[:2],
                                 (f"alarm-{number}", "input-mouse"))

    def test_mouse_move_wheel_release_injected_and_negative_code_never_ack(self):
        self.cache_alarm()
        for message in (0x200, 0x20A, 0x20E, 0x202, 0x205, 0x208, 0x20C):
            # Invalid pointer proves irrelevant events do not inspect input.
            self.hotkey._on_mouse(hk.HC_ACTION, message, 0)
        self.hotkey._on_mouse(-1, hk.WM_LBUTTONDOWN, 0)
        for flags in (hk.LLMHF_INJECTED, hk.LLMHF_LOWER_IL_INJECTED):
            self.mouse.flags = flags
            self.hotkey._on_mouse(hk.HC_ACTION, hk.WM_LBUTTONDOWN, self.mouse_pointer)
        self.assertTrue(self.hotkey._ack_requests.empty())
        self.assertEqual(self.user32.CallNextHookEx.call_count, 10)

    def test_repeated_input_is_collapsed_and_expired_cache_uses_timestamp_fallback(self):
        self.cache_alarm()
        for _ in range(20):
            self.hotkey._queue_ack("input-any-key")
        self.assertEqual(self.hotkey._ack_requests.qsize(), 1)
        self.hotkey._ack_requests.get_nowait()
        self.hotkey._queue_ack("input-mouse")
        self.assertTrue(self.hotkey._ack_requests.empty())
        self.hotkey._alarm_snapshot = ("alarm-b", time.monotonic() - hk.INPUT_CACHE_MAX_AGE - .01)
        self.hotkey._queue_ack("input-any-key")
        self.assertTrue(self.hotkey._ack_requests.empty())
        self.assertEqual(len(self.hotkey._fallback_inputs), 1)

    def test_full_ack_queue_does_not_mark_new_alarm_as_delivered(self):
        self.cache_alarm("alarm-a")
        self.hotkey._queue_ack("input-any-key")
        self.cache_alarm("alarm-b")
        self.hotkey._queue_ack("input-mouse")
        self.assertEqual(self.hotkey._last_queued_alarm_id, "alarm-a")
        self.assertEqual(self.hotkey._ack_requests.get_nowait()[0], "alarm-a")
        self.hotkey._queue_ack("input-mouse")
        self.assertEqual(self.hotkey._ack_requests.get_nowait()[0], "alarm-b")

    def test_ack_request_keeps_original_id_even_if_cache_changes(self):
        self.cache_alarm("alarm-a")
        self.hotkey._queue_ack("input-any-key")
        self.cache_alarm("alarm-b")
        alarm_id, via, pressed_at = self.hotkey._ack_requests.get_nowait()
        with patch.object(hk.urllib.request, "urlopen", return_value=io.BytesIO(
                b'{"ok":true,"acknowledged":false,"ack_remaining_s":0}')) as http:
            self.assertFalse(self.hotkey.acknowledge(alarm_id, via))
        request = http.call_args.args[0]
        self.assertEqual(request.full_url, hk.DEFAULT_API + "/ack")
        self.assertEqual(json.loads(request.data), {"alarm_id": "alarm-a"})
        self.assertEqual(request.get_header("X-via"), "input-any-key")
        self.assertEqual(http.call_args.kwargs["timeout"], hk.INPUT_HTTP_TIMEOUT)

    def test_poll_caches_only_active_id_and_clears_it_on_unavailable_server(self):
        for body, expected in [
                (b'{"armed":true,"alarm_id":"alarm-a","ack_remaining_s":0}', "alarm-a"),
                (b'{"armed":true,"alarm_id":null,"ack_remaining_s":4}', None),
                (b'{"alarm_id":true}', None), (b'[]', None), (b'not-json', None)]:
            with self.subTest(body=body), patch.object(hk.urllib.request, "urlopen",
                                                      return_value=io.BytesIO(body)) as http:
                hk.Hotkey._refresh_alarm_status(self.hotkey)
                self.assertEqual(self.hotkey._alarm_snapshot[0], expected)
                self.assertEqual(http.call_args.args[0], hk.DEFAULT_API + "/input-status")
        self.cache_alarm()
        with patch.object(hk.urllib.request, "urlopen", side_effect=OSError("offline")):
            hk.Hotkey._refresh_alarm_status(self.hotkey)
        self.assertIsNone(self.hotkey._alarm_snapshot[0])

    def test_slow_ack_cannot_delay_or_displace_toggle(self):
        ack_started, release_ack, toggled = threading.Event(), threading.Event(), threading.Event()

        def slow_ack(*args):
            ack_started.set()
            release_ack.wait(1)
            return True

        self.cache_alarm()
        with patch.object(self.hotkey, "acknowledge", side_effect=slow_ack), \
                patch.object(self.hotkey, "toggle", side_effect=toggled.set):
            self.hotkey._worker.start()
            self.hotkey._input_worker.start()
            self.hotkey._queue_ack("input-mouse")
            try:
                self.assertTrue(ack_started.wait(1))
                self.assertEqual(self.press(), hk.SWALLOW)
                self.assertTrue(toggled.wait(.5), "toggle waited for acknowledgement HTTP")
            finally:
                release_ack.set()
        self.assertTrue(self.hotkey._input_worker.daemon)

    def test_status_poll_and_ack_http_stay_on_input_worker(self):
        callback_thread = threading.get_ident()
        http_threads = []
        acknowledged = threading.Event()

        def request(target, **kwargs):
            http_threads.append(threading.get_ident())
            if isinstance(target, str):
                return io.BytesIO(b'{"alarm_id":"alarm-a","ack_remaining_s":0}')
            self.assertEqual(json.loads(target.data), {"alarm_id": "alarm-a"})
            acknowledged.set()
            return io.BytesIO(b'{"ok":true,"acknowledged":true,"ack_remaining_s":5}')

        self.hotkey._refresh_alarm_status.side_effect = lambda: hk.Hotkey._refresh_alarm_status(self.hotkey)
        with patch.object(hk.urllib.request, "urlopen", side_effect=request):
            self.hotkey._input_worker.start()
            deadline = time.monotonic() + 1
            while self.hotkey._alarm_snapshot[0] is None and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertEqual(self.hotkey._alarm_snapshot[0], "alarm-a")
            self.key.vkCode = 0x41
            self.assertEqual(self.press(hk.WM_KEYDOWN), 0x123456789ABC)
            self.assertTrue(acknowledged.wait(1))
            self.hotkey._stop.set()
            self.hotkey._input_wake.set()
            self.hotkey._input_worker.join(1)
        self.assertGreaterEqual(len(http_threads), 2)
        self.assertEqual(set(http_threads), {self.hotkey._input_worker.ident})
        self.assertNotIn(callback_thread, http_threads)

    def test_failed_ack_allows_new_press_but_never_replays_automatically(self):
        delivered = threading.Event()

        def fail_ack(*args):
            delivered.set()
            return False

        self.cache_alarm()
        with patch.object(self.hotkey, "acknowledge", side_effect=fail_ack) as ack:
            self.hotkey._input_worker.start()
            self.hotkey._queue_ack("input-any-key")
            self.assertTrue(delivered.wait(1))
            # task_done happens after the retry marker is reset.
            self.hotkey._ack_requests.join()
            self.assertIsNone(self.hotkey._last_queued_alarm_id)
            self.assertEqual(ack.call_count, 1)
            self.hotkey._queue_ack("input-mouse")
            self.hotkey._ack_requests.join()
            self.assertEqual(ack.call_count, 2)

    def test_alarm_onset_before_first_status_poll_is_acknowledged(self):
        # Last poll said no alarm. An alarm starts at 100.05 and the user's
        # press at 100.10 must not be lost while waiting for the next poll.
        completed = threading.Event()
        self.hotkey._alarm_snapshot = (None, 100.)
        with patch.object(hk.time, "monotonic", return_value=100.10):
            self.hotkey._queue_ack("input-any-key")
        self.hotkey._refresh_alarm_status.return_value = ("new-alarm", 100.05)
        with patch.object(self.hotkey, "acknowledge",
                          side_effect=lambda *args: completed.set() or True) as ack:
            self.hotkey._input_worker.start()
            self.assertTrue(completed.wait(1))
            self.hotkey._stop.set()
            self.hotkey._input_wake.set()
            self.hotkey._input_worker.join(1)
        ack.assert_called_once_with("new-alarm", "input-any-key")

    def test_input_queued_before_future_alarm_cannot_acknowledge_it(self):
        with patch.object(hk.time, "monotonic", return_value=100.10):
            self.hotkey._queue_ack("input-mouse")

        self.hotkey._refresh_alarm_status.return_value = ("future-alarm", 100.11)
        with patch.object(self.hotkey._input_wake, "wait", side_effect=lambda *_: self.hotkey._stop.set()), \
                patch.object(self.hotkey, "acknowledge") as ack:
            self.hotkey._input_loop()
        ack.assert_not_called()
        self.assertFalse(self.hotkey._fallback_inputs)

    def test_typing_coalesces_latest_timestamp_without_extra_status_polls(self):
        for timestamp in (100.01, 100.02, 100.03):
            with patch.object(hk.time, "monotonic", return_value=timestamp):
                self.hotkey._queue_ack("input-any-key")
        self.assertEqual(list(self.hotkey._fallback_inputs), [(100.03, "input-any-key")])
        self.assertTrue(self.hotkey._ack_requests.empty())
        self.hotkey._refresh_alarm_status.assert_not_called()

        def stop_after_poll():
            self.hotkey._stop.set()
            return None, None

        self.hotkey._refresh_alarm_status.side_effect = stop_after_poll
        self.hotkey._input_loop()
        self.hotkey._refresh_alarm_status.assert_called_once()

    def test_input_arriving_during_poll_is_preserved_for_next_poll(self):
        completed = threading.Event()
        with patch.object(hk.time, "monotonic", return_value=100.01):
            self.hotkey._queue_ack("input-any-key")
        polls = []

        def poll():
            polls.append(True)
            if len(polls) == 1:
                # Alarm began after the old press, then new physical input
                # arrives while the old input's status request is in flight.
                with patch.object(hk.time, "monotonic", return_value=100.03):
                    self.hotkey._queue_ack("input-mouse")
            return "new-alarm", 100.02

        self.hotkey._refresh_alarm_status.side_effect = poll
        with patch.object(self.hotkey, "acknowledge",
                          side_effect=lambda *args: completed.set() or True) as ack:
            self.hotkey._input_worker.start()
            self.assertTrue(completed.wait(1))
            self.hotkey._stop.set()
            self.hotkey._input_wake.set()
            self.hotkey._input_worker.join(1)
        ack.assert_called_once_with("new-alarm", "input-mouse")
        self.assertGreaterEqual(len(polls), 2)

    def test_missing_or_invalid_alarm_start_time_disables_fallback(self):
        for value in (None, True, -1, "100", float("nan")):
            with self.subTest(value=value):
                body = json.dumps({"alarm_id":"alarm-a", "alarm_started_at":value}).encode()
                with patch.object(hk.urllib.request, "urlopen", return_value=io.BytesIO(body)):
                    self.assertEqual(hk.Hotkey._refresh_alarm_status(self.hotkey), ("alarm-a", None))

    def test_stale_cached_id_acknowledges_replacement_only_if_it_predates_input(self):
        for replacement_start, expected_calls in ((100.09, 2), (100.11, 1)):
            with self.subTest(replacement_start=replacement_start):
                self.hotkey._last_queued_alarm_id = "old-alarm"
                self.hotkey._refresh_alarm_status.return_value = ("new-alarm", replacement_start)
                with patch.object(self.hotkey, "acknowledge", side_effect=[False, True]) as ack:
                    self.hotkey._deliver_ack("old-alarm", "input-any-key", 100.10)
                self.assertEqual(ack.call_count, expected_calls)
                self.assertEqual(ack.call_args_list[0], call("old-alarm", "input-any-key"))
                if expected_calls == 2:
                    self.assertEqual(ack.call_args_list[1], call("new-alarm", "input-any-key"))

    def test_timeout_resolution_is_bounded_and_does_not_repeat_applied_ack(self):
        for result, expected_calls in (((None, None), 1), (("same-alarm", 100.), 2)):
            with self.subTest(status=result):
                self.hotkey._last_queued_alarm_id = "same-alarm"
                self.hotkey._refresh_alarm_status.reset_mock()
                self.hotkey._refresh_alarm_status.return_value = result
                with patch.object(self.hotkey, "acknowledge", return_value=None) as ack:
                    self.hotkey._deliver_ack("same-alarm", "input-mouse", 100.10)
                self.assertEqual(ack.call_count, expected_calls)
                self.hotkey._refresh_alarm_status.assert_called_once()

    def test_new_input_while_old_ack_is_pending_retains_latest_timestamp(self):
        self.hotkey._alarm_snapshot = ("old-alarm", 100.)
        with patch.object(hk.time, "monotonic", return_value=100.01):
            self.hotkey._queue_ack("input-any-key")
        # A later click may relate to a replacement alarm while the old ACK
        # is pending. Coalesce its time instead of discarding it as a repeat.
        with patch.object(hk.time, "monotonic", return_value=100.10):
            self.hotkey._queue_ack("input-mouse")
        self.assertEqual(self.hotkey._ack_requests.get_nowait(),
                         ("old-alarm", "input-any-key", 100.01))
        self.assertEqual(list(self.hotkey._fallback_inputs), [(100.10, "input-mouse")])


if __name__ == "__main__":
    unittest.main()
