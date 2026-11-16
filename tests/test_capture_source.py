"""The USB capture stick source: found by USB id, never confused with the desk webcam."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_watch import watch
import platform_support as ps
import guard_presence
import barrier_monitor
from barrier_workflow import BarrierWorkflow


LISTING = '''[in#0 @ 000001] "USB2.0 HD UVC WebCam" (video)
[in#0 @ 000001]   Alternative name "@device_pnp_\\\\?\\usb#vid_56cd&pid_7890&mi_00#6&11111111&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global"
[in#0 @ 000001] "USB Video" (video)
[in#0 @ 000001]   Alternative name "@device_pnp_\\\\?\\usb#vid_12ab&pid_3456&mi_00#6&22222222&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global"
[in#0 @ 000001] "Digital Audio Interface (USB Digital Audio)" (audio)
[in#0 @ 000001]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{A5256781}"
'''


class DeviceListingTests(unittest.TestCase):
    def test_listing_keeps_order_and_ids_and_skips_audio(self):
        devices = ps.parse_dshow_devices(LISTING)
        self.assertEqual(['USB2.0 HD UVC WebCam', 'USB Video'], [d['name'] for d in devices])
        self.assertEqual(['56cd:7890', '12ab:3456'], [d['usb_id'] for d in devices])
        self.assertTrue(devices[1]['path'].startswith('@device_pnp_'))

    def test_find_by_usb_id(self):
        devices = ps.parse_dshow_devices(LISTING)
        self.assertEqual(1, ps.find_usb_camera('12AB:3456', devices)[0])
        self.assertEqual(0, ps.find_usb_camera('56cd:7890', devices)[0])
        self.assertIsNone(ps.find_usb_camera('1234:5678', devices))

    def test_two_identical_sticks_are_ambiguous_not_guessed(self):
        devices = ps.parse_dshow_devices(LISTING + LISTING.split('\n', 2)[2])
        self.assertIsNone(ps.find_usb_camera('12ab:3456', devices))

    def test_bad_usb_id_is_refused(self):
        for value in ('', '345f', '12ab:34560', 'zzzz:2109', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ps.parse_usb_id(value)


class CaptureCommandTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(watch, 'ffmpeg_bin', return_value='ffmpeg')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_command_opens_the_stick_by_its_device_path(self):
        devices = ps.parse_dshow_devices(LISTING)
        with mock.patch.object(watch.ps, 'dshow_video_devices', return_value=devices):
            command = watch.build_command('usbcam:12ab:3456', 10.0)
        joined = ' '.join(command)
        self.assertIn('-f dshow', joined)
        self.assertIn('-vcodec mjpeg', joined)
        self.assertIn('-video_size 1280x720', joined)
        self.assertIn('video=' + devices[1]['path'], command)
        self.assertNotIn('vid_56cd', joined)          # never the desk webcam
        self.assertEqual('rawvideo', command[command.index('-f', command.index('-pix_fmt')) + 1])

    def test_missing_stick_is_a_clear_message_not_a_crash(self):
        with mock.patch.object(watch.ps, 'dshow_video_devices', return_value=[]):
            with self.assertRaises(watch.CaptureMissing) as caught:
                watch.build_command('usbcam:12ab:3456', 10.0)
        self.assertIn('12AB:3456', str(caught.exception))
        self.assertIn('not plugged in', str(caught.exception))

    def test_reader_reports_missing_stick_and_keeps_trying(self):
        reader = watch.FrameReader('usbcam:12ab:3456', 10.0)
        calls = []

        def missing(*_args):
            calls.append(1)
            if len(calls) >= 2:
                reader._stop.set()
            raise watch.CaptureMissing('Capture stick 12AB:3456 is not plugged in or is not answering.')
        with mock.patch.object(watch, 'build_command', side_effect=missing), \
             mock.patch.object(reader._stop, 'wait', return_value=False):
            reader._run()
        self.assertEqual(2, len(calls))
        self.assertIn('not plugged in', reader.stderr_tail[0])
        self.assertEqual(0, reader.restarts)


class DeskWebcamTests(unittest.TestCase):
    def test_usb_id_config_is_normalised(self):
        config = guard_presence.PresenceConfig.from_dict({'device_usb_id': '56CD:7890'})
        self.assertEqual('56cd:7890', config.device_usb_id)

    def test_unreadable_usb_id_never_falls_back_to_camera_zero(self):
        config = guard_presence.PresenceConfig.from_dict({'device_usb_id': 'webcam'})
        presence = mock.Mock(config=config)
        self.assertIsNone(guard_presence.GuardPresence._device_index(presence))

    def test_index_follows_the_webcam_whatever_the_order(self):
        config = guard_presence.PresenceConfig.from_dict({'device_usb_id': '56cd:7890'})
        presence = mock.Mock(config=config)
        devices = ps.parse_dshow_devices(LISTING)
        with mock.patch.object(ps, 'dshow_video_devices', return_value=devices):
            self.assertEqual(0, guard_presence.GuardPresence._device_index(presence))
        with mock.patch.object(ps, 'dshow_video_devices', return_value=list(reversed(devices))):
            self.assertEqual(1, guard_presence.GuardPresence._device_index(presence))
        with mock.patch.object(ps, 'dshow_video_devices', return_value=[devices[1]]):
            self.assertIsNone(guard_presence.GuardPresence._device_index(presence))

    def test_without_usb_id_the_old_index_is_used(self):
        presence = mock.Mock(config=guard_presence.PresenceConfig.from_dict({'device_index': 2}))
        self.assertEqual(2, guard_presence.GuardPresence._device_index(presence))


class BarrierLineTests(unittest.TestCase):
    LINES = {'exit': {'down': [[.36, .20], [.54, .23]], 'up': [[.541, .01], [.541, .205]]},
             'entrance': {'down': [[.49, .28], [.80, .30]], 'up': [[.478, .01], [.478, .265]]}}

    def test_configured_lines_are_used(self):
        monitor = barrier_monitor.BarrierMonitor(lines=self.LINES)
        self.assertEqual(self.LINES['exit']['up'], monitor.lines['exit']['up'])
        self.assertIs(barrier_monitor.LINES, barrier_monitor.BarrierMonitor().lines)

    def test_bad_lines_are_refused(self):
        bad = json.loads(json.dumps(self.LINES))
        bad['exit']['up'] = [[.5, .1], [.5, .1]]
        for lines in (bad, {'exit': self.LINES['exit']}, {'exit': 1, 'entrance': 2}):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                barrier_monitor.validate_lines(lines)

    def test_calibration_store_is_bound_to_source_and_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            plain = BarrierWorkflow('usbcam:12ab:3456', folder, None)
            lined = BarrierWorkflow('usbcam:12ab:3456', folder, None, lines=self.LINES)
            other = json.loads(json.dumps(self.LINES)); other['exit']['up'] = [[.55, .01], [.55, .2]]
            moved = BarrierWorkflow('usbcam:12ab:3456', folder, None, lines=other)
            paths = {str(w.monitor.storage_path) for w in (plain, lined, moved)}
        self.assertEqual(3, len(paths))


class FixtureConfigTests(unittest.TestCase):
    def test_fixture_source_and_view_settings(self):
        root = Path(__file__).resolve().parents[1] / 'config'
        source = json.loads((root / 'source.json').read_text(encoding='utf-8'))
        self.assertEqual('usbcam:12ab:3456', source['source'])
        zones = json.loads((root / 'zones.json').read_text(encoding='utf-8'))
        self.assertEqual('56cd:7890', zones['presence']['device_usb_id'])
        barrier_monitor.validate_lines(zones['barriers']['lines'])
        for zone in zones['zones']:
            for x, y in zone['polygon']:
                self.assertTrue(0.01 <= x <= 0.99 and 0.01 <= y <= 0.99, zone['name'])


if __name__ == '__main__':
    unittest.main()


class PriorityTests(unittest.TestCase):
    @unittest.skipUnless(ps.WINDOWS, 'Windows process priority')
    def test_process_runs_at_normal_cpu_and_memory_priority(self):
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel32.GetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        me = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(me, 0x00004000)            # below normal, as Task Scheduler starts us
        result = ps.normal_priority()
        self.assertEqual({'priority': True, 'memory': True, 'throttling': True}, result)
        self.assertEqual(0x20, kernel32.GetPriorityClass(me))
        memory = wintypes.ULONG(0)
        kernel32.GetProcessInformation(me, 0, ctypes.byref(memory), ctypes.sizeof(memory))
        self.assertEqual(5, memory.value)

    def test_other_platforms_do_nothing(self):
        with mock.patch.object(ps, 'WINDOWS', False):
            self.assertEqual({'priority': False, 'memory': False, 'throttling': False}, ps.normal_priority())
