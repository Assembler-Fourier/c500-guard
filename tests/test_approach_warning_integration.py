"""Real Watcher/tracker/visit policy with fake platform, model and flow evidence.

Weak truck boxes are the saved diagnostic pair; frame-flow replay is a separate
private artifact. These tests never open a camera, model, hook, network or audio.
"""
from pathlib import Path
import unittest
from unittest import mock
from object_tracking import ObjectVisitTracker
from test_watch import watch,watcher


class RecordedSemantic:
    def __init__(self,obj,clock):
        self.obj=obj;self.clock=clock;self.tracker=ObjectVisitTracker();self.events=[]
        self.primary={'available':True,'state':'ready','age_s':0.,'observed_at':clock(),
                      'detections':[],'view_diagnostics':[]}

    def set(self,stamp,weak=(),normal=None):
        if normal is not None:
            self.events,seen=self.tracker.update(normal,self.obj.zones,640,360,stamp)
            self.primary.update(observed_at=stamp,detections=seen)
        views={}
        for candidate in weak:
            kind=candidate.get('view','whole')
            views.setdefault(kind,[]).append({'label':candidate['label'],'confidence':candidate['confidence'],
                                              'bbox_frame':candidate['bbox']})
        self.primary['view_diagnostics']=[{'kind':kind,'observed_at':stamp,'epoch':1,'generation':1,
                                          'detections':objects} for kind,objects in views.items()]
        if normal is None:self.primary['observed_at']=stamp

    def update(self,*args,**kwargs):
        result,self.events=self.events,[]
        return result

    def status(self):
        return {**self.primary,'age_s':max(0.,self.clock()-self.primary['observed_at'])}

    def event_context(self,event):return self.tracker.event_context(event,self.obj.zones)
    def mark_emitted(self,event):return self.tracker.mark_emitted(event)
    def reset(self):pass
    def frame_at(self,stamp):return None


class OfferedTruckFlow:
    def __init__(self,*args,**kwargs):self.count=0;self.latched=False;self.last=None
    def update(self,frame,at):
        self.count+=1;self.last=at
        return {'hit':self.count>=3 and not self.latched,'qualified':self.count>=2,
                'pairs':max(0,self.count-1),'observed_at':at,'points':6,'forward_px':13.58,
                'global_motion_px':.001,'support_bbox':[.554376,.313462,.646967,.357817]}
    def mark_emitted(self,stamp):
        if stamp!=self.last:return False
        self.latched=True;return True


WEAK_ONE={'label':'truck','confidence':.4574,'bbox':[285.86,41.63,426.74,132.43],'view':'whole'}
WEAK_TWO={'label':'truck','confidence':.3112,'bbox':[287.24,42.68,452.74,136.78],'view':'whole'}
WAIT_ONE={'label':'bus','confidence':.6446,'bbox':[.70173*640,.17322*360,.84081*640,.52567*360]}
WAIT_TWO={'label':'bus','confidence':.639,'bbox':[.70109*640,.17556*360,.84098*640,.53356*360]}


class ApproachWarningIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.now=99.
        for patch in (mock.patch.object(watch.time,'monotonic',side_effect=lambda:self.now),
                      mock.patch.object(watch,'notify'),
                      mock.patch('approach_warning.ApproachMotion',OfferedTruckFlow)):
            patch.start();self.addCleanup(patch.stop)
        self.obj=watcher(zones_path=Path(__file__).resolve().parents[1]/'config/views/ptz2/zones.json')
        self.obj.args.warmup=0.;self.obj.args.adapt=0.;self.obj._pending_rearm=False
        self.obj._queue_snapshot=mock.Mock(return_value=None);self.obj._alignment_blocked=False
        self.semantic=RecordedSemantic(self.obj,lambda:self.now);self.obj.semantic=self.semantic
        self.frame=watch.np.full((360,640,3),80,watch.np.uint8)
        self.zone=next(z for z in self.obj.zones if z.name=='Top road arrival')
        self.zone.approach_motion=True
        self.obj.process(self.frame,received_at=self.now)

    def step(self,at,weak=None,normal=None,stamp=None):
        self.now=at
        if weak is not None or normal is not None:self.semantic.set(at,weak or [],normal)
        self.obj.process(self.frame,received_at=at if stamp is None else stamp)

    def approach(self):
        self.step(100.,[WEAK_ONE],[]);self.step(100.5,[WEAK_TWO],[]);self.step(100.75)

    def waiting(self,start=106.193):
        self.step(start,normal=[WAIT_ONE]);self.step(start+.472,normal=[WAIT_TWO])

    def test_recorded_weak_pair_and_normal_tracker_keep_one_alarm(self):
        self.approach();self.assertEqual('main',self.obj.alarm)
        ident=self.obj.alarm_id;since=self.obj.alarm_since
        self.assertEqual(2,self.obj.alarm_details['corroboration']['candidate_observations'])
        self.waiting()
        self.assertEqual(ident,self.obj.alarm_id);self.assertEqual(since,self.obj.alarm_since)
        self.assertEqual(1,self.obj._alarm_sequence)
        self.assertEqual(1,self.obj.alarm_details.get('track_id'))

    def test_ack_then_same_waiting_truck_stays_silent_after_quiet(self):
        self.approach();self.now=102.;self.assertTrue(self.obj.acknowledge())
        self.waiting()
        for i in range(40):self.step(107+i*.25,normal=[WAIT_TWO])
        self.assertIsNone(self.obj.alarm);self.assertEqual(1,self.obj._alarm_sequence)

    def test_distinct_queued_vehicle_survives_first_ack_and_quiet(self):
        self.approach();self.waiting();self.now=107.;self.obj.acknowledge()
        follower={'label':'car','confidence':.9,'bbox':[.805*640,.385*360,.865*640,.505*360]}
        for i in range(48):self.step(107.25+i*.25,normal=[WAIT_TWO,follower])
        self.assertEqual('main',self.obj.alarm);self.assertEqual(2,self.obj._alarm_sequence)
        self.assertEqual(2,self.obj.alarm_details['track_id'])

    def test_off_warmup_stale_and_excluded_marker_do_not_offer(self):
        for index,mode in enumerate(('off','warm','stale','excluded')):
            with self.subTest(mode=mode):
                if getattr(self.obj,'_approach_warnings',None) is not None:self.obj._approach_warnings.reset()
                self.obj.armed=mode!='off';self.obj.args.warmup=1000. if mode=='warm' else 0.
                self.zone.exclude_polygons=([[.4,.1],[.72,.1],[.72,.42],[.4,.42]],) if mode=='excluded' else []
                for at,weak in ((100.,WEAK_ONE),(100.5,WEAK_TWO),(100.75,None)):
                    at+=index*3
                    self.step(at,None if weak is None else [weak],[],stamp=at-5 if mode=='stale' else at)
                self.assertIsNone(self.obj.alarm)

    def test_stationary_model_box_plus_forward_flow_does_not_warn(self):
        self.step(100.,[WEAK_ONE],[]);self.step(100.5,[WEAK_ONE],[]);self.step(100.75)
        self.assertIsNone(self.obj.alarm)

    def test_one_source_frame_seen_from_two_views_is_not_a_moving_vehicle_pair(self):
        self.step(100.,[WEAK_ONE,{**WEAK_TWO,'view':'focused'}],[])
        self.step(100.5);self.step(100.75)
        self.assertIsNone(self.obj.alarm)


if __name__=='__main__':unittest.main()
