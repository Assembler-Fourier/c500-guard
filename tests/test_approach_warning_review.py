"""Independent Watcher/coordinator lifecycle review with fake model and flow.

Real process, fusion, VisitMemory and alarm decisions; no camera, model, device,
file writes or actual alarm outputs. Vision sensitivity is tested separately.
"""
from pathlib import Path
import unittest
from unittest import mock

from test_watch import watch, watcher


class OfferedMotion:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.marks = []
        self.latched = False
        self.previous_at = None
        self.__class__.instances.append(self)

    def update(self, frame, at):
        self.calls += 1
        self.previous_at = at
        return {'hit':self.calls >= 3 and not self.latched,
                'qualified':self.calls >= 2, 'pairs':max(0,self.calls-1),
                'support_bbox':[.55,.33,.60,.35], 'observed_at':at,
                'reason':'forward road movement','points':10,'forward_px':5.,
                'global_motion_px':0.}

    def mark_emitted(self, stamp):
        if stamp != self.previous_at or self.calls < 3:
            return False
        self.marks.append(stamp)
        self.latched = True
        return True


class ApproachWarningReviewTests(unittest.TestCase):
    def setUp(self):
        self.now = 99.
        clock = mock.patch.object(watch.time, 'monotonic', side_effect=lambda:self.now)
        clock.start(); self.addCleanup(clock.stop)
        notify = mock.patch.object(watch, 'notify')
        notify.start(); self.addCleanup(notify.stop)
        flow = mock.patch('approach_warning.ApproachMotion', OfferedMotion)
        flow.start(); self.addCleanup(flow.stop)
        OfferedMotion.instances = []
        self.obj = watcher(zones_path=Path(__file__).resolve().parents[1]/'config/views/ptz2/zones.json')
        self.obj.args.warmup = 0.
        self.obj.args.adapt = 0.
        self.obj._pending_rearm = False
        self.obj._queue_snapshot = mock.Mock(return_value=None)
        self.obj._alignment_blocked = False
        self.zone = next(z for z in self.obj.zones if z.name == 'Top road arrival')
        self.zone.approach_motion = True
        self.frame = watch.np.full((360,640,3),80,dtype=watch.np.uint8)
        self.primary = {'available':True,'state':'ready','age_s':0.,'detections':[],
                        'view_diagnostics':[],'observed_at':self.now}
        self.semantic = mock.Mock()
        self.semantic.status.side_effect = lambda:self.primary
        self.semantic.update.return_value = []
        self.obj.semantic = self.semantic
        self.obj.process(self.frame, received_at=self.now)

    def deliver(self, at, box=None, *, epoch=1, ready=True, received_at=None):
        self.now = at
        if box is not None:
            self.primary = {'available':ready,'state':'ready' if ready else 'retrying',
                'age_s':0.,'observed_at':at,'detections':[], 'view_diagnostics':[
                    {'kind':'context','observed_at':at,'epoch':epoch,'generation':1,
                     'detections':[{'label':'truck','confidence':.45,
                         'bbox_frame':[v*d for v,d in zip(box,(640,360,640,360))]}]}]}
        else:
            self.primary['age_s'] = at-self.primary['observed_at']
        self.obj.process(self.frame, received_at=at if received_at is None else received_at)

    def approach(self):
        self.deliver(100., [.45,.12,.67,.36])
        self.deliver(100.5,[.47,.13,.70,.38])
        self.deliver(100.75)

    def test_real_process_accepts_fused_warning_and_only_then_marks_delivery(self):
        self.approach()
        self.assertEqual('main',self.obj.alarm)
        self.assertEqual('motion',self.obj.alarm_details['source'])
        self.assertNotIn('track_id',self.obj.alarm_details)
        self.assertIn('corroboration',self.obj.alarm_details)
        self.assertEqual([100.75],OfferedMotion.instances[-1].marks)
        self.assertEqual(1,self.obj._alarm_sequence)

    def test_quiet_rejection_does_not_consume_offer_and_fresh_pair_retries(self):
        self.obj.acked = True
        self.obj._ack_until = 101.
        self.approach()
        self.assertIsNone(self.obj.alarm)
        self.assertEqual([],OfferedMotion.instances[-1].marks)
        self.deliver(101.01,[.49,.14,.73,.39])
        self.assertEqual('main',self.obj.alarm)
        self.assertEqual([101.01],OfferedMotion.instances[-1].marks)

    def test_same_later_confirmed_ai_identity_enriches_one_provisional_alarm(self):
        self.approach()
        alarm_id = self.obj.alarm_id
        self.now = 101.
        details = {'source':'ai','zone_name':self.zone.name,'route_group':self.zone.route_group,
            'route_role':'entrance','alert_stage':'advance','track_id':123,'label':'truck',
            'confidence':.7,'observations':2,'direction_state':'toward',
            'observed_at':self.now,'first_observed_at':100.5,
            'frame_received_at':self.now,'bbox_normalized':[.47,.13,.70,.38]}
        self.assertTrue(self.obj.raise_alarm('main','confirmed truck',details=details))
        self.assertEqual(alarm_id,self.obj.alarm_id)
        self.assertEqual(1,self.obj._alarm_sequence)
        self.assertEqual(123,self.obj.alarm_details['track_id'])

    def test_old_frame_and_untrusted_alignment_do_not_offer_or_consume(self):
        self.deliver(100., [.45,.12,.67,.36])
        self.deliver(100.5,[.47,.13,.70,.38])
        self.deliver(104.,[.49,.14,.73,.39],received_at=100.75)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual([],OfferedMotion.instances[-1].marks)
        self.obj._alignment_blocked = True
        self.deliver(104.5,[.50,.15,.74,.41])
        self.assertIsNone(self.obj.alarm)
        self.assertEqual({},self.obj._approach_warnings.detectors)
        self.assertFalse(self.zone.approach_motion_active)

    def test_model_generation_change_discards_prior_motion_and_candidate_pair(self):
        self.deliver(100., [.45,.12,.67,.36])
        self.deliver(100.5,[.47,.13,.70,.38])
        old = OfferedMotion.instances[-1]
        self.deliver(100.75,[.49,.14,.73,.39],epoch=2)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual([],old.marks)
        self.assertIsNot(old,OfferedMotion.instances[-1])
        self.assertEqual(1,OfferedMotion.instances[-1].calls)

    def test_model_failure_cannot_reuse_previous_healthy_candidates(self):
        self.deliver(100., [.45,.12,.67,.36])
        self.deliver(100.5,[.47,.13,.70,.38])
        self.deliver(100.75,[.49,.14,.73,.39],ready=False)
        self.assertIsNone(self.obj.alarm)
        self.assertEqual([],OfferedMotion.instances[-1].marks)
        self.assertEqual(0,len(self.obj._approach_warnings.candidates))

    def test_excluded_vehicle_class_cannot_warn_or_hold_route_occupied(self):
        self.zone.classes = ['car']
        self.approach()
        self.assertIsNone(self.obj.alarm)
        self.assertFalse(self.zone.approach_motion_active)
        self.assertEqual([],OfferedMotion.instances[-1].marks)


if __name__ == '__main__':unittest.main()
