"""A moving car's old-position fragment must not steal its established ID."""
from types import SimpleNamespace
import unittest

from object_tracking import ObjectVisitTracker, _Track, _track_score


def zone(name='road', rect=(0.,0.,1.,1.)):
    return SimpleNamespace(name=name,rect=rect,classes=['car','bus','truck','person'],
        ai_enabled=True,ai_confidence=.5,ai_confirm_observations=2,route_group='route',severity='main',
        arrival_direction=None,arrival_min_progress=.012)


def car(box, confidence=.9, label='car'):
    return {'label':label,'confidence':confidence,'bbox':box}


def body(x,y=160,label='car',confidence=.9):
    return car([x-30,y-20,x+30,y+20],confidence,label)


class MotionContinuityTests(unittest.TestCase):
    def test_measured_fragment_keeps_original_id_on_advancing_car(self):
        tracker=ObjectVisitTracker()
        zones=[zone()]
        samples=[(9.2006,[car([276.15,90.65,322.68,117.42],.5545)]),
            (9.5076,[car([291.09,88.02,327.77,120.23],.3958)]),
            (9.8997,[car([291.89,91.19,360.22,124.07],.7466),
                     car([322.11,95.1,355.53,123.18],.3344)])]
        original=None
        for at,objects in samples:
            _,seen=tracker.update(objects,zones,640,360,100+at)
            original=original or seen[0]['track_id']
        # Exact cached boxes: one physical car is split into several boxes.
        objects=[car([327.58,94.78,355.87,124.92],.5298),
                 car([332.39,94.79,380.07,126.02],.4318),
                 car([347.21,100.42,392.51,128.18],.3657),
                 car([433.29,175.56,553.61,226.39],.3514),
                 car([292.65,95.29,356.23,123.39],.3383)]
        _,seen=tracker.update(objects,zones,640,360,110.3055)
        advancing=next(o for o in seen if o['confidence']==.4318)
        old_fragment=next(o for o in seen if o['confidence']==.3383)
        self.assertEqual(original,advancing['track_id'])
        self.assertNotEqual(original,old_fragment['track_id'])
        _,seen=tracker.update([car([357.13,96.09,417.65,134.04],.8451)],zones,640,360,110.8071)
        self.assertEqual(original,seen[0]['track_id'])

    def test_identity_is_global_across_zone_boundaries(self):
        tracker=ObjectVisitTracker()
        zones=[zone('approach',(0,0,.45,1)),zone('barrier',(.45,0,.30,1))]
        identities=[]
        observed_regions=[]
        for i,x in enumerate(range(100,501,40)):
            _,seen=tracker.update([body(x)],zones,640,360,100+i*.3)
            identities.append(seen[0]['track_id'])
            observed_regions.append(seen[0]['regions'])
        self.assertEqual(1,len(set(identities)))
        self.assertIn(['approach'],observed_regions)
        self.assertIn(['barrier'],observed_regions)
        self.assertEqual([],observed_regions[-1])

    def test_braking_lead_and_new_follower_keep_separate_ids(self):
        tracker=ObjectVisitTracker()
        for i,x in enumerate([100,115,130]):
            _,seen=tracker.update([body(x)], [zone()],640,360,100+i*.3)
        lead=seen[0]['track_id']
        _,seen=tracker.update([body(85),body(130)], [zone()],640,360,100.9)
        follower=seen[0]['track_id']
        self.assertNotEqual(lead,follower)
        self.assertEqual(lead,seen[1]['track_id'])
        for at,x in [(101.2,100),(101.5,110),(101.8,110)]:
            _,seen=tracker.update([body(130),body(x)], [zone()],640,360,at)
            self.assertEqual([lead,follower],[o['track_id'] for o in seen])

    def test_opposing_cars_keep_ids_through_crossing_with_input_reordering(self):
        tracker=ObjectVisitTracker()
        first=second=None
        for i,(left,right) in enumerate([(100,220),(120,200),(140,180),(160,160),(180,140),(200,120)]):
            # Slightly separated projected lane centers preserve two visible cars.
            objects=[body(left,150),body(right,175)]
            if i%2:objects.reverse()
            _,seen=tracker.update(objects,[zone()],640,360,100+i*.3)
            ids={int((o['bbox'][1]+o['bbox'][3])/2):o['track_id'] for o in seen}
            if first is None:first,second=ids[150],ids[175]
            self.assertEqual((first,second),(ids[150],ids[175]))

    def test_person_crossing_does_not_take_car_identity(self):
        tracker=ObjectVisitTracker()
        original=None
        for i,x in enumerate([100,120,140,160,180]):
            _,seen=tracker.update([body(x),body(200-i*20,label='person')],[zone()],640,360,100+i*.3)
            original=original or seen[0]['track_id']
            self.assertEqual(original,seen[0]['track_id'])
            self.assertNotEqual(original,seen[1]['track_id'])

    def test_long_missing_vehicle_still_expires(self):
        tracker=ObjectVisitTracker()
        for i,x in enumerate([100,115,130]):
            _,seen=tracker.update([body(x)],[zone()],640,360,100+i*.3)
        original=seen[0]['track_id']
        _,seen=tracker.update([body(130)],[zone()],640,360,106.)
        self.assertNotEqual(original,seen[0]['track_id'])

    def test_uncertain_short_or_stale_motion_keeps_old_position_fallback(self):
        old=(.456078125,.253305556,.56284375,.344638889)
        advancing=(332.39/640,94.79/360,380.07/640,126.02/360)
        fragment=(292.65/640,95.29/360,356.23/640,123.39/360)
        def track(**changes):
            values=dict(confidence=.7466,velocity=(.05861245,.02631508),trail=[(.46,.32),(.48,.33),(.51,.34)])
            values.update(changes)
            return _Track(1,'car',old,100.,**values)
        self.assertGreater(_track_score(track(),advancing,100.4058),_track_score(track(),fragment,100.4058))
        for changes,elapsed in [({'confidence':.49},.4058),({'trail':[(.48,.33),(.51,.34)]},.4058),({},.8)]:
            with self.subTest(changes=changes,elapsed=elapsed):
                tr=track(**changes)
                self.assertGreater(_track_score(tr,fragment,100+elapsed),_track_score(tr,advancing,100+elapsed))


if __name__=='__main__':unittest.main()
