"""Native playback failure must not masquerade as a playing or delivered alarm."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from test_watch import watch, watcher


class BuzzerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now).start()
        self.player = mock.Mock()
        self.paths = {k: Path(k + '.wav') for k in
                      ('main', 'early', 'fault', 'soft', 'notice', 'wake', 'limited')}
        with mock.patch.object(watch.ps, 'AudioPlayer', return_value=self.player):
            self.buzzer = watch.Buzzer(self.paths)

    def fail_once(self):
        self.player.start_loop.side_effect = [OSError('isolated device failure'), None]
        self.buzzer.set('main')

    def test_first_playback_failure_is_visible_and_same_profile_retries(self):
        self.fail_once()
        status = self.buzzer.playback_status()
        self.assertEqual('main', status['requested'])
        self.assertFalse(status['playing'])
        self.assertIn('OSError', status['error'])
        self.now += .2
        self.buzzer.set('main')
        self.buzzer.set_profile('soft')
        self.assertEqual(1, self.player.start_loop.call_count)
        self.now += .31
        self.buzzer.set_profile('soft')
        self.assertEqual(2, self.player.start_loop.call_count)
        self.assertTrue(self.buzzer.playback_status()['playing'])
        self.assertEqual('', self.buzzer.playback_status()['error'])
        self.assertEqual(self.paths['soft'], self.buzzer._playing)

    def test_failed_wake_profile_switch_is_retried_without_new_alarm(self):
        self.buzzer.set('main')
        self.player.start_loop.side_effect = [RuntimeError('device detached'), None]
        self.buzzer.set_profile('wake')
        self.assertIsNone(self.buzzer._playing)
        self.now += .51
        self.buzzer.set_profile('wake')
        self.assertEqual('main', self.buzzer._want)
        self.assertEqual(self.paths['wake'], self.buzzer._playing)

    def test_persistent_failure_does_not_flood_native_player(self):
        self.player.start_loop.side_effect = OSError('still detached')
        for _ in range(100):
            self.buzzer.set('main')
            self.buzzer.set_profile('wake')
            self.now += .001
        self.assertEqual(1, self.player.start_loop.call_count)
        self.now += .5
        self.buzzer.set_profile('wake')
        self.assertEqual(2, self.player.start_loop.call_count)
        self.assertFalse(self.buzzer.playback_status()['playing'])

    def test_ack_cancels_pending_playback_recovery(self):
        self.fail_once()
        self.buzzer.set(None)
        self.now += 10
        self.buzzer.set_profile('wake')
        self.buzzer.set(None)
        self.assertEqual(1, self.player.start_loop.call_count)
        self.assertIsNone(self.buzzer.playback_status()['requested'])
        self.assertEqual('', self.buzzer.playback_status()['error'])

    def test_evidence_silent_profile_cancels_pending_retry(self):
        self.fail_once()
        self.buzzer.set_profile('silent')
        self.now += 10
        self.buzzer.set('main')
        self.buzzer.set_profile('silent')
        self.assertEqual(1, self.player.start_loop.call_count)
        self.assertFalse(self.buzzer.playback_status()['playing'])
        self.assertEqual('', self.buzzer.playback_status()['error'])

    def test_successful_repeated_requests_do_not_restart_waveform(self):
        self.buzzer.set('main')
        for _ in range(10):
            self.now += 1
            self.buzzer.set('main')
            self.buzzer.set_profile('soft')
        self.player.start_loop.assert_called_once_with(self.paths['soft'])

    def test_fault_retries_its_fault_asset(self):
        self.player.start_loop.side_effect = [OSError('device unavailable'), None]
        self.buzzer.set('fault')
        self.now += .5
        self.buzzer.set('fault')
        self.assertEqual(self.paths['fault'], self.buzzer._playing)
        self.assertEqual(2, self.player.start_loop.call_count)

    def test_replaced_sound_failure_recovers_using_new_asset(self):
        self.buzzer.set('main')
        updated = {**self.paths, 'soft': Path('new-soft.wav')}
        self.player.start_loop.side_effect = [OSError('device busy'), None]
        self.buzzer.replace_sounds(updated)
        self.assertIsNone(self.buzzer._playing)
        self.now += .5
        self.buzzer.set_profile('soft')
        self.assertEqual(updated['soft'], self.buzzer._playing)

    def test_shutdown_and_muted_mode_never_resume_retry(self):
        self.fail_once()
        self.buzzer.shutdown()
        self.now += 10
        self.buzzer.set_profile('wake')
        self.assertEqual(1, self.player.start_loop.call_count)
        self.buzzer.muted = True
        self.buzzer.set('main')
        self.buzzer.set_profile('notice')
        self.assertEqual(1, self.player.start_loop.call_count)

    def test_watcher_keeps_alert_history_when_native_playback_fails(self):
        obj = watcher()
        obj.buzzer = self.buzzer
        obj.loudness = SimpleNamespace(kind=None, begin=mock.Mock(), end=mock.Mock())
        obj._queue_snapshot = mock.Mock(return_value=None)
        obj._consider_already_opening = mock.Mock()
        obj.recorder = mock.Mock()
        self.player.start_loop.side_effect = OSError('device unavailable')
        with mock.patch.object(watch, 'notify'):
            self.assertTrue(obj.raise_alarm('main', 'Isolated arrival'))
        self.assertEqual('main', obj.alarm)
        self.assertEqual(obj.alarm_id, obj.recent_alerts[0]['id'])
        obj.recorder.alarm.assert_called_once()
        self.assertFalse(obj.buzzer.playback_status()['playing'])


if __name__ == '__main__':
    unittest.main()
