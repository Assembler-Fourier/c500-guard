"""Immediate corner changes bypass recognition while retaining alarm/ACK rules.

Real configured geometry and the Watcher.process path; no camera, model worker,
speaker changes, notifications, or persistent production state.
"""
import unittest
from unittest import mock

from test_watch import watch, watcher


class StrictCornerTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        notifications = mock.patch.object(watch, 'notify')
        notifications.start()
        self.addCleanup(notifications.stop)
        self.obj = watcher()
        self.obj.semantic = None
        self.obj.loudness = mock.Mock()
        self.obj.args.warmup = 0.0
        self.obj.args.adapt = 0.0
        self.obj.alignment_config = {}
        self.obj._alignment_blocked = False
        self.corner = next(zone for zone in self.obj.zones
                           if zone.name == 'Exit corner - immediate')
        self.exit_approach = next(zone for zone in self.obj.zones if zone.name == 'Exit approach')
        # Isolate this channel: other routes have deliberately different rules.
        self.obj.zones = [self.corner]
        self.base = watch.np.full((360, 640, 3), 80.0, dtype=watch.np.float32)
        # Test patterns were laid out around row 180, column 106. Shift them to
        # a place fully inside whatever corner zone is configured, so the tests
        # follow the real geometry instead of one picture's pixel numbers.
        mask = self.corner.mask_for(640, 360)
        area = int(mask.sum())
        # A patch must be at least one connected car-sized piece: the zone's
        # connected_pct floor when configured, else the fixed 24-pixel floor.
        floor = max(24, watch.math.ceil(area * (self.corner.connected_pct or 0) / 100))
        self.side = max(8, watch.math.ceil(watch.math.sqrt(floor)) + 1)
        span = max(26, self.side + 4)
        anchor = next(((row, col) for row in range(8, 340) for col in range(8, 600)
                       if mask[row - 2:row + self.side + 2, col - 8:col + span].all()), None)
        self.assertIsNotNone(anchor, 'corner zone too small for the test patterns')
        self.dy, self.dx = anchor[0] - 180, anchor[1] - 106
        self.patch_slice = (slice(180 + self.dy, 180 + self.side + self.dy),
                            slice(106 + self.dx, 106 + self.side + self.dx))
        self.assertTrue(mask[self.patch_slice].all())
        # A weak (dim) patch: above the 8-level change threshold, below the
        # 10-level strong threshold, the size of a real car-sized patch.
        self.weak_value = 89.0
        self.obj.semantic = mock.Mock()
        self.obj.semantic.update.return_value = []
        self.obj.semantic.status.return_value = {"available": True, "state": "ready", "age_s": 0., "detections": []}
        self.obj.process(self.base)

    def deliver(self, frame, after=0.1):
        self.now += after
        self.obj.process(frame)

    def changed_patch(self, delta=12.0):
        frame = self.base.copy()
        frame[self.patch_slice] += delta
        return frame

    def assert_urgent_main(self, frames=1):
        self.assertEqual(self.obj.alarm, 'main')
        self.assertEqual(self.obj.buzzer.want, 'main')
        self.assertIs(self.obj.alarm_details['strict_change'], True)
        self.assertEqual(self.obj.alarm_details['source'], 'motion')
        self.assertEqual(self.obj.alarm_details['zone_name'], self.corner.name)
        self.assertEqual(self.obj.alarm_details['confirmation_frames'], frames)
        # A verified corner change is urgent: the faster, stronger tier applies
        # before recognition, because a driver is already stopped at the barrier.
        self.obj.loudness.begin.assert_called_with('main', source='motion', confirmed=False, lane='exit',
                                                   urgent=True)

    def test_config_is_one_frame_raw_change_at_the_lower_exit_edge(self):
        self.assertTrue(self.corner.strict_change)
        self.assertEqual(self.corner.trigger, 'motion')
        self.assertTrue(self.corner.motion)
        self.assertFalse(self.corner.light)
        self.assertFalse(self.corner.ai_enabled)
        self.assertEqual(self.corner.classes, [])
        self.assertEqual(self.corner.severity, 'main')
        self.assertEqual(self.corner.on_frames, 1)
        self.assertEqual(self.corner.off_frames, 15)
        self.assertEqual(self.corner.change_threshold, 8.0)
        self.assertEqual(self.corner.motion_pct, 0.6)
        # The road just above the keyboard's blue icon at the lower-left edge:
        # below the kerb (row 172 at the frame edge) and above the icon (row 234).
        mask = self.corner.mask_for(640, 360)
        self.assertFalse(mask[:180].any())
        self.assertFalse(mask[234:].any())
        self.assertFalse(mask[:, 130:].any())
        self.assertFalse((mask & ~self.exit_approach.mask_for(640, 360)).any())

    def test_first_bright_patch_frame_raises_main_without_ai(self):
        self.obj.semantic = None
        self.assertIsNone(self.obj.semantic)
        self.deliver(self.changed_patch(12.0))
        self.assert_urgent_main()
        self.assertEqual(self.obj._alarm_sequence, 1)
        self.assertEqual(self.obj.loudness.begin.call_count, 1)

    def test_first_dark_patch_frame_raises_main_without_ai(self):
        self.deliver(self.changed_patch(-12.0))
        self.assert_urgent_main()
        self.assertLess(self.corner.last_delta, 0)

    def test_clear_change_survives_even_a_saturated_noise_floor(self):
        self.obj.bg_noise.fill(watch.MAX_NOISE)
        self.deliver(self.changed_patch(20.0))
        self.assert_urgent_main()

    def test_outside_corner_change_does_not_trigger_this_channel(self):
        frame = self.base.copy()
        frame[200:240, 240:280] = 160.0
        self.assertFalse(self.corner.mask_for(640, 360)[200:240, 240:280].any())
        for _ in range(20):
            self.deliver(frame)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(self.corner.last_pct, 0.0)
        self.obj.loudness.begin.assert_not_called()

    def test_uniform_exposure_change_stays_quiet(self):
        self.deliver(watch.np.full_like(self.base, 100.0))
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(self.corner.last_pct, 0.0)
        self.assertEqual(self.corner.change_metrics['global_offset'], 20.0)

    def test_large_uniform_exposure_change_stays_quiet(self):
        self.deliver(watch.np.full_like(self.base, 140.0))
        self.assertIsNone(self.obj.alarm)

    def test_global_exposure_plus_an_unrelated_outside_object_stays_quiet(self):
        frame = self.base + 20.0
        frame[190:230,340:380] += 50.0
        self.deliver(frame)
        self.assertIsNone(self.obj.alarm)

    def test_local_headlight_survives_global_exposure_correction(self):
        frame = self.base + 20.0
        frame[self.patch_slice] += 30.0
        self.deliver(frame)
        self.assert_urgent_main()

    def test_urgent_alarm_is_delivered_before_synchronous_ai_work(self):
        def update(*args):
            self.assert_urgent_main()
            return []
        self.obj.semantic = mock.Mock()
        self.obj.semantic.update.side_effect = update
        self.deliver(self.changed_patch())
        self.obj.semantic.update.assert_called_once()

    def test_forty_six_scattered_changed_pixels_stay_quiet(self):
        # The largest observed empty-road false alarm counted only 46 pixels.
        frame = self.base.copy()
        mask = self.corner.mask_for(640, 360)
        candidates = [(y,x) for y in range(160+self.dy,199+self.dy,3)
                      for x in range(80+self.dx,158+self.dx,3)
                      if 0 <= y < 360 and 0 <= x < 640 and mask[y,x]]
        for y,x in candidates[:46]:
            frame[y,x] = 180.0
        self.deliver(frame)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(self.corner.last_pct, 0.0)

    def test_compression_shimmer_patches_stay_quiet(self):
        # Measured on the capture feed, 13 Nov 2026: empty-road corner alarms had
        # 270-700 changed pixels but no connected patch above 157 pixels.
        if not self.corner.connected_pct:
            self.skipTest('the configured corner has no connected-patch floor')
        mask = self.corner.mask_for(640, 360)
        frame = self.base.copy()
        placed = 0
        for row in range(0, 360, 14):
            for col in range(0, 640, 14):
                block = (slice(row, row + 10), slice(col, col + 10))
                if mask[block].all() and placed < 6:
                    frame[block] = 180.0
                    placed += 1
        self.assertGreaterEqual(placed, 3)
        for _ in range(3):
            self.deliver(frame)
        self.assertIsNone(self.obj.alarm)
        self.assertGreater(self.corner.change_metrics['changed_pixels'], 250)

    def test_small_connected_speck_stays_quiet(self):
        frame = self.base.copy()
        frame[181+self.dy:184+self.dy,107+self.dx:110+self.dx] = 150.0
        self.deliver(frame)
        self.assertIsNone(self.obj.alarm)

    def test_thin_jitter_line_stays_quiet(self):
        frame = self.base.copy()
        frame[184+self.dy,100+self.dx:130+self.dx] = 150.0
        self.deliver(frame)
        self.assertIsNone(self.obj.alarm)

    def test_small_headlight_patch_requires_second_frame_in_same_place(self):
        # Smallest patch that still counts: 0.6% of this zone, never under 12
        # pixels, and well below the 48-pixel one-frame threshold.
        frame = self.base.copy()
        frame[self.patch_slice] = self.weak_value
        self.deliver(frame)
        self.assertIsNone(self.obj.alarm)
        self.deliver(frame)
        self.assert_urgent_main(frames=2)

    def test_one_frame_small_flash_does_not_alert(self):
        frame = self.base.copy()
        frame[self.patch_slice] = self.weak_value
        self.deliver(frame)
        for _ in range(10):
            self.deliver(self.base)
        self.assertIsNone(self.obj.alarm)

    def test_small_patches_at_different_positions_do_not_confirm(self):
        frame = self.base.copy()
        frame[self.patch_slice] = self.weak_value
        self.deliver(frame)
        frame = self.base.copy()
        rows, cols = self.patch_slice
        frame[rows, cols.start + self.side + 2:cols.stop + self.side + 2] = self.weak_value
        self.deliver(frame)
        self.assertIsNone(self.obj.alarm)

    def test_stale_weak_observation_does_not_confirm(self):
        frame = self.base.copy()
        frame[self.patch_slice] = self.weak_value
        self.deliver(frame)
        self.deliver(frame, after=2.0)
        self.assertIsNone(self.obj.alarm)

    def test_unchanged_frames_stay_quiet(self):
        for _ in range(30):
            self.deliver(self.base)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(self.obj._alarm_sequence, 0)
        self.obj.loudness.begin.assert_not_called()

    def test_change_below_pixel_threshold_stays_quiet(self):
        self.deliver(self.changed_patch(3.0))
        self.assertIsNone(self.obj.alarm)
        self.assertEqual(self.corner.last_pct, 0.0)

    def test_ack_keeps_ten_seconds_and_never_repeats_the_same_episode(self):
        changed = self.changed_patch()
        self.deliver(changed)
        self.assert_urgent_main()
        first_id = self.obj.alarm_id
        self.assertTrue(self.obj.acknowledge(first_id, via='web'))
        self.assertAlmostEqual(self.obj._ack_until - self.now, 10.0)
        self.obj.loudness.end.assert_called_once()
        for _ in range(130):
            self.deliver(changed)
            self.assertIsNone(self.obj.alarm)
            self.assertIsNone(self.obj.buzzer.want)
        self.assertEqual(self.obj._alarm_sequence, 1)
        self.assertEqual(self.obj.loudness.begin.call_count, 1)
        self.assertTrue(self.corner.motion_active)
        self.assertTrue(self.corner._notified['motion'])

    def test_clear_episode_then_new_change_after_quiet_raises_again(self):
        self.deliver(self.changed_patch())
        first_id = self.obj.alarm_id
        self.assertTrue(self.obj.acknowledge(first_id, via='web'))
        quiet_until = self.obj._ack_until
        for _ in range(self.corner.off_frames + 65):
            self.deliver(self.base)
        self.assertFalse(self.corner.motion_active)
        self.assertFalse(self.corner._notified['motion'])
        self.now = quiet_until + 0.01
        self.deliver(self.changed_patch(-12.0))
        self.assert_urgent_main()
        self.assertNotEqual(self.obj.alarm_id, first_id)
        self.assertEqual(self.obj._alarm_sequence, 2)

    def test_new_episode_during_quiet_waits_until_quiet_expires(self):
        self.deliver(self.changed_patch())
        self.assertTrue(self.obj.acknowledge(via='web'))
        quiet_until = self.obj._ack_until
        for _ in range(self.corner.off_frames + 65):
            self.deliver(self.base)
        changed = self.changed_patch(-12.0)
        self.deliver(changed)
        self.assertLess(self.now, quiet_until)
        self.assertIsNone(self.obj.alarm)
        self.assertFalse(self.corner._notified['motion'])
        self.now = quiet_until + 0.01
        self.deliver(changed)
        self.assert_urgent_main()
        self.assertEqual(self.obj._alarm_sequence, 2)

    def test_disarmed_strict_change_stays_silent(self):
        self.obj.armed = False
        self.deliver(self.changed_patch())
        self.assertIsNone(self.obj.alarm)
        self.obj.loudness.begin.assert_not_called()

    def test_an_existing_main_alarm_is_not_restarted_by_another_channel(self):
        self.assertTrue(self.obj.raise_alarm('main', 'Ordinary vehicle alarm', details={'source': 'ai'}))
        original_id = self.obj.alarm_id
        self.obj.loudness.begin.assert_called_once_with('main', source='ai', confirmed=True, lane='',
                                                        urgent=False)
        self.deliver(self.changed_patch())
        self.assertEqual(self.obj.alarm_id, original_id)
        self.assertEqual(self.obj._alarm_sequence, 1)
        self.assertFalse(self.corner._notified['motion'])


if __name__ == '__main__':
    unittest.main()
