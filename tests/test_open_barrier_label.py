"""A lane alarm while that lane's own arm was already up is labelled "barrier already open" (label only).

14 Nov 22:08:47: the exit arm had been left up for 82 s; an ambulance drove straight through.
"""
import re
import time
import unittest

from test_watch import watch, watcher
import dashboard_ui


class OpenBarrierLabelTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.exit = next(z for z in self.obj.zones if z.route_role == 'exit')
        self.now = time.monotonic()

    def test_alarm_with_its_own_arm_left_up_is_labelled(self):
        self.obj._arm_last_known = {'exit': ('up', self.now - 1)}
        self.obj._arm_raised_at = {'exit': self.now - 82}
        context = self.obj._alarm_context(self.exit, {}, self.now)
        self.assertEqual('exit', context['own_arm_up']['arm'])
        self.assertEqual(82.0, context['own_arm_up']['up_for_s'])
        self.assertIn('already open (up 82 s)', context['own_arm_up']['text'])

    def test_arm_up_since_before_start_is_labelled_without_a_time(self):
        self.obj._arm_last_known = {'exit': ('up', self.now - 2)}
        context = self.obj._alarm_context(self.exit, {}, self.now)
        self.assertIsNone(context['own_arm_up']['up_for_s'])
        self.assertIn('already open, so', context['own_arm_up']['text'])

    def test_no_label_when_down_stale_just_raised_or_other_arm(self):
        cases = [({'exit': ('down', self.now - 1)}, {}),
                 ({'exit': ('up', self.now - 45)}, {}),                       # reading too old to trust
                 ({'exit': ('up', self.now - 1)}, {'exit': self.now - 4}),     # raised just before: normal
                 ({'entrance': ('up', self.now - 1)}, {'entrance': self.now - 90})]
        for known, raised in cases:
            with self.subTest(known=known, raised=raised):
                self.obj._arm_last_known, self.obj._arm_raised_at = dict(known), dict(raised)
                self.assertNotIn('own_arm_up', self.obj._alarm_context(self.exit, {}, self.now))

    def test_the_label_never_changes_how_the_alarm_starts(self):
        self.obj._arm_last_known = {'exit': ('up', self.now - 1)}
        details = {'source': 'motion', 'strict_change': True, 'zone_name': self.exit.name,
                   **self.obj._alarm_context(self.exit, {}, self.now)}
        self.assertTrue(self.obj.raise_alarm('main', 'exit corner', details=details))
        self.assertEqual('main', self.obj.alarm)
        self.assertNotIn('started_gentle', self.obj.alarm_details)

    def test_activity_shows_the_tag_and_explanation(self):
        page = dashboard_ui.dashboard_page('BarrierWatcher')
        self.assertIn('openArm:evidence.own_arm_up', page)
        self.assertIn('Barrier open</em>', page)
        self.assertRegex(page, re.escape("[d.crossLane,d.openArm,d.cameFrom].filter(Boolean).join(' ')"))


if __name__ == '__main__':
    unittest.main()
