"""Synthetic detector/stream checks. No camera connection or real audio/state."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import platform_support

# Importing watch normally creates its state directory; isolate that side effect.
_state = tempfile.TemporaryDirectory(prefix="watch-tests-")
with mock.patch.object(platform_support, "state_dir", return_value=Path(_state.name)):
    spec = importlib.util.spec_from_file_location("watch_under_test", ROOT / "src" / "watch.py")
    watch = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = watch
    spec.loader.exec_module(watch)


class FakeBuzzer:
    def __init__(self, *_args, **_kwargs):
        self.want = None

    def set(self, kind):
        self.want = kind

    def shutdown(self):
        self.set(None)


def watcher(zones_path=None):
    # Generic/legacy image fixtures use the ANPR coordinate system. Live PTZ
    # scenarios pass their own profile instead of changing every old fixture.
    args = types.SimpleNamespace(
        source="rtsp://example.invalid/video", fps=8., zones=str(zones_path or ROOT / "config" / "views" / "anpr" / "zones.json"),
        silent=True, connect_grace=.3, lost_after=.2, freeze_after=20., black_after=5.,
        warmup=4., pixel_threshold=22., adapt=.02,
    )
    with mock.patch.object(watch, "ensure_sounds", return_value={}), \
         mock.patch.object(watch, "Buzzer", FakeBuzzer), \
         mock.patch('semantic_monitor.SemanticMonitor', return_value=None), \
         mock.patch.object(watch.Watcher, "_load_armed", return_value=True):
        obj = watch.Watcher(args)
    obj.log_event = mock.Mock()
    # Sound tuning is module state: the site config (10% start, gentle climb)
    # must not leak into tests written for the built-in defaults.
    import alarm_audio
    alarm_audio.reset_configuration()
    return obj


class WatcherTests(unittest.TestCase):
    def test_tiny_valid_zone_does_not_crash_preview(self):
        obj = watcher()
        obj.zones = [watch.Zone(name='Small', rect=(.2,.2,.0001,.0001))]
        obj.render_preview(watch.np.zeros((watch.PROC_H,watch.PROC_W,3), dtype=watch.np.uint8))
        self.assertTrue(obj.snapshot_preview().startswith(b'\xff\xd8'))
    def test_sleep_prevention_tracks_armed_state_on_detector_thread(self):
        obj = watcher()
        inhibitor = mock.MagicMock()
        inhibitor.active = False
        inhibitor.__enter__.side_effect = lambda: setattr(inhibitor, 'active', True)
        inhibitor.__exit__.side_effect = lambda: setattr(inhibitor, 'active', False)
        obj.sleep_inhibitor = inhibitor
        obj._update_awake()
        obj._update_awake()
        inhibitor.__enter__.assert_called_once()
        obj.armed = False
        obj._update_awake()
        inhibitor.__exit__.assert_called_once()

    def setUp(self):
        self.notify = mock.patch.object(watch, "notify").start()

    def tearDown(self):
        mock.patch.stopall()

    def test_missing_or_invalid_armed_file_starts_disarmed(self):
        obj = watcher()
        with mock.patch.object(watch, "ARMED_FILE", Path(_state.name) / "missing.json"):
            self.assertFalse(obj._load_armed())
        with mock.patch.object(watch, "ARMED_FILE") as state:
            for data in ('{"armed":"false"}', '{"armed":1}', 'bad json'):
                state.read_text.return_value = data
                self.assertFalse(obj._load_armed())
            state.read_text.return_value = '{"armed":true}'
            self.assertTrue(obj._load_armed())

    def test_disarmed_rejects_all_alarm_channels(self):
        obj = watcher()
        obj.set_armed(False)
        obj.log_event.reset_mock()
        for kind in ("early", "main", "fault"):
            obj.raise_alarm(kind, "synthetic")
        self.assertIsNone(obj.alarm)
        self.assertIsNone(obj.buzzer.want)
        self.assertFalse(obj.log_event.called)
        self.assertFalse(self.notify.called)

    def test_disarm_waits_for_inflight_raise_then_silences(self):
        obj = watcher()
        entered = threading.Event()
        release = threading.Event()
        original = obj._raise_alarm_locked

        def paused(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(2))
            original(*args, **kwargs)

        with mock.patch.object(obj, "_raise_alarm_locked", side_effect=paused):
            raise_thread = threading.Thread(target=obj.raise_alarm, args=("main", "test"))
            raise_thread.start()
            self.assertTrue(entered.wait(2))
            disarm_thread = threading.Thread(target=obj.set_armed, args=(False,))
            disarm_thread.start()
            release.set()
            raise_thread.join(2)
            disarm_thread.join(2)
            self.assertFalse(raise_thread.is_alive())
            self.assertFalse(disarm_thread.is_alive())
        self.assertFalse(obj.armed)
        self.assertIsNone(obj.alarm)
        self.assertIsNone(obj.buzzer.want)

    def test_same_fault_stays_acknowledged_beyond_quiet_deadline(self):
        obj = watcher()
        obj.first_frame_seen = True
        obj.last_frame_at = 1.
        with mock.patch.object(watch.time, 'monotonic', return_value=3.) as clock:
            obj.check_health(3.)
            self.assertEqual(obj.alarm, "fault")
            obj.acknowledge()
            for instant in (7.99, 13., 60.):
                clock.return_value = instant
                obj.check_health(instant)
                self.assertIsNone(obj.alarm)
                self.assertIsNone(obj.buzzer.want)
                self.assertTrue(obj.fault_reason)

    def test_recovered_fault_can_alarm_again_in_a_new_episode(self):
        obj = watcher()
        obj.zones = []
        obj.first_frame_seen = True
        obj.last_frame_at = 1.
        with mock.patch.object(watch.time, 'monotonic', return_value=3.) as clock:
            obj.check_health(3.)
            old_id = obj.alarm_id
            obj.acknowledge()
            clock.return_value = 8.
            obj.last_frame_at = 8.
            obj.check_health(8.)
            self.assertEqual(obj.fault_reason, '')
            self.assertIsNone(obj._fault_kind)
            clock.return_value = 14.
            obj.check_health(14.)
            self.assertEqual(obj.alarm, 'fault')
            self.assertNotEqual(obj.alarm_id, old_id)

    def test_new_fault_during_vehicle_quiet_waits_but_is_not_lost(self):
        obj = watcher()
        obj.zones = []
        obj.first_frame_seen = True
        with mock.patch.object(watch.time, 'monotonic', return_value=100.) as clock:
            obj.last_frame_at = 100.
            obj.raise_alarm('main', 'vehicle arrival')
            obj.acknowledge()
            for instant in (101., 109.999):
                clock.return_value = instant
                obj.check_health(instant)
                self.assertIsNone(obj.alarm)
                self.assertTrue(obj.fault_reason)
            clock.return_value = 110.
            obj.check_health(110.)
            self.assertEqual(obj.alarm, 'fault')
            self.assertEqual(obj._fault_kind, 'lost')

    def test_different_fault_during_fault_quiet_is_deferred_not_lost(self):
        obj = watcher()
        obj.zones = []
        obj.first_frame_seen = True
        obj.last_frame_at = 1.
        with mock.patch.object(watch.time, 'monotonic', return_value=3.) as clock:
            obj.check_health(3.)
            old_id = obj.alarm_id
            obj.acknowledge()
            obj.dark_since = .1
            clock.return_value = 6.
            obj.last_frame_at = 6.
            obj.check_health(6.)
            self.assertIsNone(obj.alarm)
            self.assertIn('picture black', obj.fault_reason)
            clock.return_value = 13.
            obj.last_frame_at = 13.
            obj.check_health(13.)
            self.assertEqual(obj.alarm, 'fault')
            self.assertEqual(obj._fault_kind, 'black')
            self.assertNotEqual(obj.alarm_id, old_id)

    def test_fault_rearms_when_deliberately_switched_back_on(self):
        obj = watcher()
        obj.first_frame_seen = True
        obj.last_frame_at = 1.
        obj.check_health(3.)
        obj.acknowledge()
        obj.set_armed(False)
        obj.set_armed(True)
        obj.check_health(4.)
        self.assertEqual(obj.alarm, "fault")

    def test_warmup_starts_on_first_frame_and_does_not_consume_arrival(self):
        obj = watcher()
        obj.zones = [watch.Zone(name="synthetic", rect=(0., 0., 1., 1.))]
        base = watch.np.full((10, 10, 3), 80., dtype=watch.np.float32)
        changed = base.copy()
        changed[:3, :3, :] = 180.
        with mock.patch.object(watch.time, "monotonic", return_value=100.) as clock:
            obj.process(base)
            clock.return_value = 101.
            obj.process(changed)
            clock.return_value = 102.
            obj.process(changed)
            self.assertIsNone(obj.alarm)
            self.assertFalse(obj.zones[0].motion_active)
            clock.return_value = 105.
            obj.process(changed)
            obj.process(changed)
        self.assertEqual(obj.alarm, "main")

    def test_log_disk_error_does_not_prevent_disarm(self):
        obj = watcher()
        obj.log_event = types.MethodType(watch.Watcher.log_event, obj)
        with mock.patch.object(watch, "EVENT_LOG") as log:
            log.open.side_effect = OSError("disk unavailable")
            obj.raise_alarm("main", "test")
            obj.set_armed(False)
        self.assertIsNone(obj.buzzer.want)

    def test_detector_error_keeps_running_then_faults_if_persistent(self):
        obj = watcher()
        with mock.patch("builtins.print"):
            try:
                raise ValueError("bad frame")
            except ValueError as exc:
                obj._note_error("Detection", exc)
        start = obj._frame_error_since
        obj.first_frame_seen = True
        obj.last_frame_at = start + 3.5
        obj.check_health(start + 2.9)
        self.assertIsNone(obj.alarm)
        obj.check_health(start + 3.5)
        self.assertEqual(obj.alarm, "fault")
        self.assertIn("bad frame", obj.fault_reason)

    def test_state_exposes_the_alarm_level_behind_each_reading(self):
        zones = watcher().state_dict()['zones']
        self.assertTrue(zones)
        for zone in zones:
            self.assertIn('light_threshold', zone)
            self.assertIn('motion_threshold', zone)
            for key in ('light_threshold', 'motion_threshold'):
                self.assertTrue(zone[key] is None or isinstance(zone[key], (int, float)))

    def test_a_finished_siren_test_never_silences_a_real_alarm(self):
        obj = watcher()
        obj.play_test_tone(seconds=.2)
        self.assertEqual(obj.buzzer.want, 'main')
        obj.raise_alarm('fault', 'feed lost')      # a real alarm takes the buzzer
        time.sleep(.5)
        self.assertEqual(obj.buzzer.want, 'fault')

    def test_losing_the_camera_closes_the_program_but_a_hiccup_does_not(self):
        obj = watcher()
        obj.first_frame_seen = True
        obj.camera_gone_s = 60.
        with mock.patch.object(obj, '_request_full_stop') as stop:
            obj.last_frame_at = 100.
            obj.check_health(140.)                      # 40 s gap: a reconnect
            stop.assert_not_called()
            obj.last_frame_at = 145.                    # frames came back
            obj.check_health(145.)
            self.assertEqual(obj.camera_gone_status(145.)['missing_s'], 0.)
            obj.check_health(210.)                      # 65 s without the camera
            stop.assert_called_once()
            self.assertIn('camera offline', stop.call_args.args[0])
            self.assertEqual(obj.camera_gone_status(210.)['stopping_in_s'], 0.)

    def test_a_camera_that_never_arrived_does_not_close_the_program(self):
        obj = watcher()
        obj.first_frame_seen = False
        obj.camera_gone_s = 30.
        with mock.patch.object(obj, '_request_full_stop') as stop:
            obj.last_frame_at = 0.
            obj.check_health(500.)
            stop.assert_not_called()

    def test_automatic_stop_is_silent_logged_and_not_repeated(self):
        obj = watcher()
        obj.buzzer.set('fault')
        with mock.patch.dict('sys.modules', {'power_control': mock.MagicMock()}) as modules:
            obj._request_full_stop('camera offline for 130s')
            obj._request_full_stop('camera offline for 190s')
            modules['power_control'].request_full_off.assert_called_once()
        self.assertIsNone(obj.buzzer.want)
        self.assertEqual(obj.log_event.call_args.args[0], 'power')

    def test_camera_stop_limit_reads_the_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zones.json'
            for value, expected in ((90, 90.), (0, 0.), (5, 30.), ('x', watch.CAMERA_GONE_S), (9e9, 3600.)):
                path.write_text(json.dumps({"zones": [], "power": {"stop_when_camera_gone_s": value}}))
                self.assertEqual(watch.Watcher._load_camera_gone_limit(path), expected)
            path.write_text(json.dumps({"zones": []}))
            self.assertEqual(watch.Watcher._load_camera_gone_limit(path), watch.CAMERA_GONE_S)

    def test_preview_error_is_not_a_detection_fault(self):
        obj = watcher()
        with mock.patch("builtins.print"):
            obj._note_error("Preview", RuntimeError("encoder"), detection=False)
        self.assertEqual(obj._frame_error_since, 0.)

    def test_stalled_detector_loop_sounds_fault_and_exits_for_supervisor(self):
        obj = watcher()
        with mock.patch.object(watch.os, "_exit") as exit_, mock.patch.object(watch.time, "sleep"), \
                mock.patch("builtins.print"):
            self.assertFalse(obj._watchdog_expired(obj._loop_heartbeat + watch.LOOP_STALL_S - 1))
            exit_.assert_not_called()
            self.assertTrue(obj._watchdog_expired(obj._loop_heartbeat + watch.LOOP_STALL_S + 1))
        exit_.assert_called_once_with(3)
        self.assertEqual(obj.buzzer.want, "fault")
        self.assertEqual(obj.log_event.call_args.args[0], "fault")

    def test_speaker_check_failure_is_reported_not_raised(self):
        with mock.patch.object(watch.ps, "audio_output_status", side_effect=OSError("no device")):
            status = watch.Watcher._read_audio()
        self.assertFalse(status["available"])
        self.assertIn("no device", status["detail"])

    def test_alarm_history_snapshot_and_acknowledgement_survive_restart(self):
        obj = watcher()
        obj.log_event = types.MethodType(watch.Watcher.log_event, obj)
        obj.preview_jpeg = b"\xff\xd8fake-jpeg"
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(watch, "EVENT_LOG", Path(folder) / "events.jsonl"), \
                mock.patch.object(watch, "ALERT_DIR", Path(folder) / "alerts"):
            self.assertTrue(obj.raise_alarm("main", "Gate: 3% changed"))
            alarm_id = obj.alarm_id
            snapshot = obj.alerts()["alerts"][0]["snapshot"]
            self.assertRegex(snapshot, watch.ALERT_NAME)
            obj._snapshots.join()
            self.assertEqual((Path(folder) / "alerts" / snapshot).read_bytes(), b"\xff\xd8fake-jpeg")
            obj.alarm_since -= 2
            self.assertTrue(obj.acknowledge(alarm_id, via="input-mouse"))
            self.assertEqual(obj.alerts()["alerts"][0]["acked"]["via"], "input-mouse")
            restarted = watcher()
            alert = restarted.alerts()["alerts"][0]
        self.assertEqual(alert["id"], alarm_id)
        self.assertEqual(alert["snapshot"], snapshot)
        self.assertEqual(alert["reason"], "Gate: 3% changed")
        self.assertEqual(alert["kind"], "main")
        self.assertEqual(alert["acked"]["via"], "input-mouse")
        self.assertGreaterEqual(alert["acked"]["after_s"], 2)

    def test_history_ignores_malformed_lines_and_missing_snapshots(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(watch, "EVENT_LOG", Path(folder) / "events.jsonl"), \
                mock.patch.object(watch, "ALERT_DIR", Path(folder) / "alerts"):
            (Path(folder) / "events.jsonl").write_text(
                'not json\n[]\n{"ts":"2026-09-11T19:04:19","kind":"main","detail":"old format"}\n'
                '{"ts":"2026-09-11T19:05:00","kind":"fault","detail":"x","snapshot":"../../secret.jpg"}\n',
                encoding="utf-8")
            alerts = watcher().alerts()["alerts"]
        self.assertEqual([a["kind"] for a in alerts], ["fault", "main"])
        self.assertIsNone(alerts[0]["snapshot"])
        self.assertEqual((alerts[1]["date"], alerts[1]["time"]), ("2026-09-11", "19:04:19"))

    def test_source_credentials_and_query_are_redacted(self):
        source = "rtsp://username:password@192.0.2.1/video?token=private#fragment"
        result = watch.safe_source(source)
        self.assertEqual(result, "rtsp://192.0.2.1/video?redacted")
        reader = watch.FrameReader(source, 8.)
        proc = types.SimpleNamespace(stderr=io.BytesIO(("error " + source + "\n").encode()))
        reader._drain_stderr(proc)
        for secret in ("username", "password", "private", "fragment"):
            self.assertNotIn(secret, str(reader.stderr_tail))


class StreamTests(unittest.TestCase):
    def test_stalled_child_is_terminated_and_reader_reconnects(self):
        # A real child keeps stdout open and emits nothing, reproducing the
        # blocked read seen on lost RTSP feeds without using any network.
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        reader = watch.FrameReader("synthetic", 8., connect_grace=.2, stall_after=.2)
        with mock.patch.object(watch, "build_command", return_value=command):
            reader.start()
            try:
                deadline = time.monotonic() + 4.
                while reader.restarts == 0 and time.monotonic() < deadline:
                    time.sleep(.03)
                self.assertGreaterEqual(reader.restarts, 1)
            finally:
                reader.stop()
        self.assertFalse(reader._thread.is_alive())
        self.assertFalse(reader._watchdog.is_alive())

    def test_stop_interrupts_restart_backoff(self):
        reader = watch.FrameReader("synthetic", 8.)
        with mock.patch.object(watch, "build_command", side_effect=ValueError("invalid source")):
            reader.start()
            deadline = time.monotonic() + 2.
            while not reader.stderr_tail and time.monotonic() < deadline:
                time.sleep(.01)
            started = time.monotonic()
            reader.stop()
        self.assertLess(time.monotonic() - started, 1.)
        self.assertFalse(reader._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
