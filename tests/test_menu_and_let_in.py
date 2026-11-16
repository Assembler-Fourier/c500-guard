"""Keyboard menu over the picture pauses zones with a clear fault; a car let in on the other lane starts gentler."""
import time
import unittest
from unittest import mock

import numpy as np

from test_watch import watch, watcher


def frame(menu=False, level=60):
    image = np.full((watch.PROC_H, watch.PROC_W, 3), level, np.uint8)
    if menu:
        image[0:18, :, :] = 250                       # the keyboard's white "Live View" bar
        image[6:12, 20:60, :] = 40                    # its icons
    return image


class KeyboardMenuTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.semantic = mock.Mock()
        self.obj.semantic.status.return_value = {'available': True, 'state': 'ready', 'age_s': .1}
        mock.patch.object(watch, 'notify').start()

    def tearDown(self):
        mock.patch.stopall()

    def test_menu_bar_is_recognised_and_ordinary_views_are_not(self):
        self.assertTrue(watch.Watcher._keyboard_menu_visible(frame(menu=True)))
        for level in (20, 60, 140, 200):
            self.assertFalse(watch.Watcher._keyboard_menu_visible(frame(level=level)), level)
        bright = frame(level=60)
        bright[0:18, 300:640, :] = 250                # bright sky or glare across part of the top
        self.assertFalse(watch.Watcher._keyboard_menu_visible(bright))

    def test_menu_pauses_zones_and_raises_a_clear_fault(self):
        self.assertIsNone(self.obj.align_frame(frame(menu=True)))
        self.assertEqual('keyboard_menu', self.obj.alignment['state'])
        now = time.monotonic()
        self.obj.first_frame_seen, self.obj.last_frame_at = True, now + 1e6
        self.obj.check_health(now + 3)
        self.assertEqual('fault', self.obj.alarm)
        self.assertIn("keyboard's menu is open", self.obj.fault_reason)
        self.obj.aligner = mock.Mock()
        self.obj.aligner.update.return_value = (frame(), {'reference_ready': True, 'translation_px': [0, 0],
                                                          'current_to_reference': [[1, 0, 0], [0, 1, 0]]})
        self.assertIsNotNone(self.obj.align_frame(frame()))
        self.assertIsNone(self.obj._menu_since)


class LetInTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.obj.semantic = None
        self.obj.armed = True
        self.obj.loudness = mock.Mock()
        mock.patch.object(watch, 'notify').start()
        self.exit = next(z for z in self.obj.zones if z.route_role == 'exit')
        self.now = time.monotonic()

    def tearDown(self):
        mock.patch.stopall()

    def test_raising_an_arm_is_remembered(self):
        status = lambda state, at: {'trusted': True, 'barriers': [
            {'id': 'entrance', 'state': state, 'fresh': True, 'calibrated': True,
             'observed_at': at, 'position_since': at}]}
        self.obj._ack_on_any_arm(status('down', self.now - .2), self.now - .2)
        self.obj._ack_on_any_arm(status('up', self.now - .1), self.now - .1)
        self.assertIn('entrance', self.obj._arm_raised_at)

    def test_exit_alarm_just_after_the_entrance_arm_went_up_starts_gentler(self):
        self.obj._arm_raised_at = {'entrance': self.now - 6}
        context = self.obj._alarm_context(self.exit, {}, self.now)
        self.assertTrue(context['cross_lane']['arm_raised'])
        self.assertIn('let in at the entrance barrier', context['cross_lane']['text'])
        self.obj.raise_alarm('main', 'Exit corner - immediate: immediate change alarm (30% changed)',
                             details={'source': 'motion', 'strict_change': True, 'zone_name': self.exit.name,
                                      'route_group': self.exit.route_group, 'route_role': 'exit', **context})
        self.assertFalse(self.obj.loudness.begin.call_args.kwargs['urgent'])
        self.assertTrue(self.obj.alarm_details['started_gentle'])

    def test_real_exit_without_an_arm_change_keeps_the_urgent_tier(self):
        self.obj._arm_raised_at = {'entrance': self.now - 60}
        context = self.obj._alarm_context(self.exit, {}, self.now)
        self.assertNotIn('cross_lane', context)
        self.obj.raise_alarm('main', 'Exit corner - immediate: immediate change alarm (30% changed)',
                             details={'source': 'motion', 'strict_change': True, 'zone_name': self.exit.name,
                                      'route_group': self.exit.route_group, 'route_role': 'exit'})
        self.assertTrue(self.obj.loudness.begin.call_args.kwargs['urgent'])

    def test_setting_can_keep_every_alarm_urgent(self):
        self.obj.soften_let_in = False
        self.obj._arm_raised_at = {'entrance': self.now - 3}
        context = self.obj._alarm_context(self.exit, {}, self.now)
        self.obj.raise_alarm('main', 'Exit corner', details={'source': 'motion', 'strict_change': True,
                                                             'zone_name': self.exit.name, **context})
        self.assertTrue(self.obj.loudness.begin.call_args.kwargs['urgent'])


if __name__ == '__main__':
    unittest.main()
