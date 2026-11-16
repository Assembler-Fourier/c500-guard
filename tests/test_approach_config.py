import copy
import json
from pathlib import Path
import unittest
from unittest import mock
from test_watch import watch, watcher
from zone_editor import validate_config
from approach_light import measure_local_brightness


class ApproachConfigTests(unittest.TestCase):
    def config(self):
        return {'zones':[dict(name='Arrival',rect=[.1,.1,.5,.5],trigger='ai_light',
                    priority=True,route_group='west',motion=False,light=True,
                    ai_confirm_observations=3,ai_confidence=.65,arrival_direction=[.7,-.7],
                    arrival_min_progress=.012,stationary_confirm_s=3)]}

    def test_all_arrival_fields_survive_validation(self):
        config=self.config()
        self.assertEqual(config,validate_config(config))

    def test_bad_direction_confidence_and_confirmation_rejected(self):
        for key,value in [('priority',1),('route_group',{}),('ai_confidence',True),
                          ('ai_confidence',float('nan')),('ai_confirm_observations',1),
                          ('ai_confirm_observations',2.5),('arrival_direction',[0,0]),
                          ('arrival_direction',[2,0]),('stationary_confirm_s',-1),
                          ('arrival_min_progress',float('inf'))]:
            config=self.config();config['zones'][0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                validate_config(config)

    def test_anpr_profile_covers_both_priority_paths(self):
        config=json.loads((Path(__file__).parents[1]/'config/views/anpr/zones.json').read_text())
        regions=[watch.Zone(**item) for item in validate_config(config)['zones']]
        priority=[z for z in regions if z.priority]
        self.assertEqual(3,len(priority))
        strict=next(z for z in priority if z.strict_change)
        self.assertEqual(1,strict.on_frames)
        self.assertEqual('motion',strict.trigger)
        lower=next(z for z in priority if z.route_role=='exit' and not z.strict_change)
        self.assertTrue(lower.ai_enabled)
        self.assertTrue(lower.light)
        self.assertEqual('main',lower.light_severity)
        self.assertTrue(lower.contains_point(106/640,184/360))
        upper=next(z for z in priority if z.route_role=='entrance')
        self.assertTrue(upper.contains_point(477/640,87/360))

    def test_saturated_float32_background_stays_valid(self):
        obj=watcher();obj.semantic=None;obj.args.warmup=0;obj.args.adapt=.02
        obj.zones=[watch.Zone('Corner',(0,0,.5,1),trigger='ai_light',light=True,light_delta=3)]
        frame=watch.np.full((20,20,3),255.,watch.np.float32)
        with mock.patch.object(watch,'notify'):
            for _ in range(60):obj.process(frame)
        self.assertLessEqual(float(obj.bg.max()),255.)
        mask=watch.np.ones((20,20),bool)
        report=measure_local_brightness(frame,watch.np.full((20,20),255.0002,watch.np.float32),mask)
        self.assertFalse(report['hit'])

    def test_camera_connection_grace_is_not_misreported_as_ai_failure(self):
        obj=watcher();obj.semantic=None
        now=watch.time.monotonic()
        obj.started=obj.last_frame_at=now-20
        obj.first_frame_seen=False;obj.args.connect_grace=30
        obj._ai_bad_since=now-10
        with mock.patch.object(watch,'notify'):
            obj.check_health(now)
            self.assertIsNone(obj.alarm)
            obj.check_health(now+11)
        self.assertEqual('fault',obj.alarm)
        self.assertIn('no usable frame',obj.fault_reason)
