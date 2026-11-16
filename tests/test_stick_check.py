"""No picture for 2 minutes: switch off only when the capture stick has really been pulled."""
import time
import unittest
from unittest import mock

from test_watch import watch, watcher


class StickRuleTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.args.source = 'usbcam:12ab:3456'
        self.obj.camera_gone_s = 120.
        self.obj.first_frame_seen = True
        self.now = time.monotonic()
        self.obj.last_frame_at = self.now - 130          # 130 s without a picture
        self.obj.reader = mock.Mock()
        self.stop = mock.patch.object(self.obj, '_request_full_stop').start()

    def tearDown(self):
        mock.patch.stopall()

    def test_pulled_stick_switches_everything_off(self):
        self.obj.reader.device_present = False
        self.obj.check_health(self.now)
        self.stop.assert_called_once()

    def test_frozen_picture_with_the_stick_connected_stays_on_and_says_so(self):
        self.obj.reader.device_present = True
        self.obj.check_health(self.now)
        self.obj.check_health(self.now + 1)
        self.stop.assert_not_called()
        notes = [call.args[1] for call in self.obj.log_event.call_args_list if call.args[0] == 'power']
        self.assertEqual(1, len(notes))
        self.assertIn('still connected', notes[0])
        status = self.obj.camera_gone_status(self.now)
        self.assertTrue(status['staying_on'])
        self.assertIsNone(status['stopping_in_s'])
        self.obj.reader.device_present = False                 # then the guard pulls it
        self.obj.check_health(self.now + 2)
        self.stop.assert_called_once()

    def test_other_sources_keep_the_old_rule(self):
        self.obj.args.source = 'rtsp://example.invalid/video'
        self.obj.reader.device_present = True
        self.obj.check_health(self.now)
        self.stop.assert_called_once()

    def test_countdown_is_shown_before_the_limit(self):
        self.obj.reader.device_present = True
        self.obj.last_frame_at = self.now - 30
        status = self.obj.camera_gone_status(self.now)
        self.assertFalse(status['staying_on'])
        self.assertAlmostEqual(90., status['stopping_in_s'], delta=.5)


class ReaderPresenceTests(unittest.TestCase):
    def test_reader_records_whether_the_stick_was_found(self):
        missing = watch.CaptureMissing('Capture stick 12AB:3456 is not plugged in or is not answering.')
        with mock.patch.object(watch, 'build_command', side_effect=missing):
            reader = watch.FrameReader('usbcam:12ab:3456', 10)
            self.assertIsNone(reader.device_present)
            reader.start()
            deadline = time.monotonic() + 3
            while reader.device_present is None and time.monotonic() < deadline:
                time.sleep(.02)
            reader.stop()
        self.assertIs(False, reader.device_present)
        self.assertIn('not plugged in', reader.stderr_tail[0])


if __name__ == '__main__':
    unittest.main()
