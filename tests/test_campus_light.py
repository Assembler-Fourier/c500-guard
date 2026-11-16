"""Synthetic top-road light regression checks against the deployed configuration.

No camera, model worker, sound or real traffic is used. Pixel changes verify
masking, timing and AI/light integration; they do not establish field accuracy.
"""
from pathlib import Path
# Coordinates below are the September site positions mapped into the keyboard
# capture picture (13 Nov 2026) with the same lens-corrected fit as the zones.
import threading
import unittest
from unittest import mock

from test_watch import watch, watcher
from semantic_monitor import SemanticMonitor
from region_geometry import polygon_mask


def car_at(x,y):
    return {'label':'car','confidence':.9,'bbox':[x-14,y-20,x+14,y]}


class CampusTopLightTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        clock = mock.patch.object(watch.time,'monotonic',side_effect=lambda:self.now)
        clock.start(); self.addCleanup(clock.stop)
        notifications = mock.patch.object(watch,'notify')
        notifications.start(); self.addCleanup(notifications.stop)
        self.obj = watcher()
        # These tests concern compensated top-road signals. The explicitly
        # strict corner intentionally has the opposite global-exposure policy.
        self.obj.zones = [z for z in self.obj.zones if not z.strict_change]
        self.obj.args.warmup = 0.
        self.obj.args.adapt = 0.
        self.obj.alignment_config = {}
        self.top = next(zone for zone in self.obj.zones if zone.name=='Top road arrival')
        self.frame = watch.np.full((360,640,3),80,dtype=watch.np.uint8)
        self.top_patch = self.frame.copy()
        self.top_patch[48:78,390:462] = 180
        self.obj.process(self.frame)
        with mock.patch.object(threading.Thread,'start'):
            self.semantic = SemanticMonitor(self.obj.zones,Path('unused-model.onnx'),interval=1000.)
        self.semantic._enabled = True
        self.semantic._last_submit = self.now
        self.obj.semantic = self.semantic
        self.addCleanup(self.semantic.stop)

    def deliver(self, *, frame=None, objects=(), after=.2):
        self.now += after
        self.semantic._clear_queue(self.semantic._output)
        self.semantic._output.put((self.semantic._epoch,self.semantic._worker_generation,
                                   self.now,list(objects)))
        self.obj.process(self.frame if frame is None else frame)

    def top_light_alarm(self):
        for _ in range(self.top.on_frames - 1):
            self.deliver(frame=self.top_patch)
            self.assertIsNone(self.obj.alarm)
        self.deliver(frame=self.top_patch)
        self.assertEqual('main',self.obj.alarm)
        self.assertEqual('main',self.obj.buzzer.want)
        self.assertEqual('Top road arrival',self.obj.alarm_details['zone_name'])
        self.assertEqual('light',self.obj.alarm_details['source'])

    def test_actual_config_has_independent_top_band_light_and_ai_confirmation(self):
        self.assertEqual('ai_light',self.top.trigger)
        self.assertTrue(self.top.light)
        self.assertFalse(self.top.motion)
        self.assertEqual(20,self.top.light_delta)   # retuned for the clean capture feed, 13 Nov 2026
        self.assertEqual('main',self.top.light_severity)
        self.assertEqual(self.top.confirmation_polygon,self.top.light_polygon)
        expected = (self.top.mask_for(640,360)
                    & polygon_mask(640,360,self.top.light_polygon))
        watch.np.testing.assert_array_equal(expected,self.top.light_mask_for(640,360))
        self.assertTrue(expected[63,425])
        self.assertFalse(expected[133,492])
        self.assertFalse(expected[214,393])

    def test_top_local_bright_patch_buzzes_without_ai_detection(self):
        self.top_light_alarm()
        self.assertEqual('advance',self.obj.alarm_details['alert_stage'])
        self.assertTrue(self.obj.alarm_details['measurement']['hit'])
        self.assertEqual(1,self.obj._alarm_sequence)
        self.assertFalse(self.top.motion_active)

    def test_light_change_inside_shared_curve_below_top_band_is_silent(self):
        patch = self.frame.copy()
        patch[114:158,457:520] = 210
        self.assertTrue(self.top.contains_point(492/640,133/360))
        for _ in range(8):
            self.deliver(frame=patch)
        self.assertIsNone(self.obj.alarm)
        self.assertIsNone(self.obj.buzzer.want)
        self.assertFalse(self.top.light_metrics['hit'])

    def test_foreground_parking_light_is_silent(self):
        patch = self.frame.copy()
        patch[199:237,348:504] = 210
        for _ in range(8):
            self.deliver(frame=patch)
        self.assertIsNone(self.obj.alarm)
        self.assertFalse(self.top.light_metrics['hit'])

    def test_global_exposure_change_is_not_a_top_headlight(self):
        patch = watch.np.full_like(self.frame,110)
        for _ in range(8):
            self.deliver(frame=patch)
        self.assertIsNone(self.obj.alarm)
        self.assertFalse(self.top.light_metrics['hit'])
        self.assertGreater(self.top.light_metrics['global_offset'],29)

    def test_ack_ten_second_quiet_does_not_repeat_same_light_episode(self):
        self.top_light_alarm()
        self.assertTrue(self.obj.acknowledge())
        self.assertAlmostEqual(10,self.obj._ack_until-self.now)
        for _ in range(65):
            self.deliver(frame=self.top_patch)
            self.assertIsNone(self.obj.alarm)
            self.assertIsNone(self.obj.buzzer.want)
        self.assertEqual(1,self.obj._alarm_sequence)
        self.assertTrue(self.top._notified['light'])

    def test_parking_only_ai_detections_still_require_top_confirmation_gate(self):
        for x,y in [(514,108),(509,122),(496,138),(482,153),(468,164)]+[(468,164)]*20:
            self.deliver(objects=[car_at(x,y)])
            self.assertIsNone(self.obj.alarm)
        self.assertEqual(0,self.obj._alarm_sequence)

    def test_top_ai_arrival_still_alerts_without_any_brightness_change(self):
        self.deliver(objects=[car_at(483,56)])
        self.assertIsNone(self.obj.alarm)
        self.deliver(objects=[car_at(473,68)])
        self.assertEqual('main',self.obj.alarm)
        self.assertEqual('ai',self.obj.alarm_details['source'])
        self.assertEqual('advance',self.obj.alarm_details['alert_stage'])
        self.assertTrue(self.obj.alarm_details['confirmation_required'])
        self.assertFalse(self.top.light_metrics['hit'])


if __name__ == '__main__':
    unittest.main()
