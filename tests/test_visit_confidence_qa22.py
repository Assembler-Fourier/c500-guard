"""Synthetic QA regressions: real tracking/memory/alarm decisions, fake outputs only."""
import copy
from pathlib import Path
import struct
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from object_tracking import ObjectVisitTracker, _recent_motion_supports
from visit_memory import VisitMemory, _strong_stopline_proof
# This fixture imports Watcher with its state directory redirected to a tempdir.
from test_watch import watch


class Zone(SimpleNamespace):
    def contains_point(self, x, y):
        left, top, width, height = self.rect
        return left <= x <= left + width and top <= y <= top + height


def zone(**change):
    return Zone(**(dict(name='Synthetic gate', route_group='synthetic-entrance',
        route_role='entrance', rect=(.1, .1, .8, .8), classes=['car'],
        ai_enabled=True, motion=False, motion_active=False, light_active=False,
        severity='main', ai_confidence=.5, arrival_direction=None,
        ai_confirm_observations=2, alert_stage='barrier') | change))


def row(y=.55, *, x=.5, label='car', confidence=.9, **extra):
    return dict(label=label, confidence=confidence,
                bbox=[(x-.1)*100, (y-.15)*100, (x+.1)*100, y*100], **extra)


def ml(rows):
    return dict(available=True, state='ready', age_s=0., detections=rows)


def fake_watcher(zones):
    """No Watcher constructor, workers, model, audio, capture, notification or IO."""
    obj = watch.Watcher.__new__(watch.Watcher)
    obj._state_lock = threading.RLock()
    obj.armed, obj._pending_rearm = True, False
    obj._ack_until, obj._resume_alarm_check = 0., False
    obj.alarm, obj.alarm_id, obj.alarm_reason = None, None, ''
    obj.alarm_details, obj.acked = {}, False
    obj.alarm_since, obj._alarm_sequence, obj._alarm_session = 0., 0, 'synthetic'
    obj.visits, obj.zones = VisitMemory(), zones
    obj.ack_quiet_s, obj.input_ack_min_s = 10., 1.
    obj.buzzer, obj.loudness = Mock(), Mock(kind=None)
    obj._queue_snapshot = Mock(return_value=None)
    obj.log_event, obj._remember_alert = Mock(), Mock()
    obj._mark_alert_acknowledged = Mock()
    obj._consider_already_opening = Mock()
    obj._service_pending, obj._arm_live = set(), {}
    obj._alignment_blocked, obj.alignment = False, {'state': 'tracking'}
    obj.barrier_rules = {'ack_on_raise': True, 'ack_on_lower': True}
    obj.barrier_workflow = Mock()
    obj.barrier_workflow.status.return_value = {'trusted': True}
    obj._tracker_agrees = Mock(return_value=True)
    return obj


class Flow:
    """Use the real semantic caller's accepted-event/mark-emitted contract."""
    def __init__(self, zones):
        self.zones, self.tracker = zones, ObjectVisitTracker()
        self.watcher = fake_watcher(zones)
        self.trace, self.alarms = [], []

    def step(self, at, objects):
        events, observed = self.tracker.update(objects, self.zones, 100, 100, at)
        self.watcher.visits.observe(at, self.zones, ml(observed), width=100, height=100)
        with patch.object(watch.time, 'monotonic', return_value=at), patch.object(watch, 'notify'):
            for kind, reason, event_id in events:
                details = dict(source='ai', **self.tracker.event_context(event_id, self.zones),
                               frame_received_at=at)
                before = self.watcher.alarm_id
                accepted = self.watcher.raise_alarm(kind, reason, details=details)
                if accepted:
                    self.tracker.mark_emitted(event_id)
                if self.watcher.alarm_id and self.watcher.alarm_id != before:
                    self.alarms.append((at, copy.deepcopy(self.watcher.alarm_details)))
                self.trace.append(dict(at=at, event_id=event_id, details=details,
                                       accepted=accepted, alarm_id=self.watcher.alarm_id))
        return observed

    def ack(self, at, mode='manual'):
        obj = self.watcher
        with patch.object(watch.time, 'monotonic', return_value=at):
            if mode == 'automatic':
                return obj._silence_for_barrier_change(dict(type='raised',
                    alarm_id=obj.alarm_id, route=VisitMemory.route_key(obj.alarm_details),
                    alarm_started_at=obj.alarm_since, origin_observed_at=obj.alarm_since+.05,
                    position_since=obj.alarm_since+.1, observed_at=at))
            return obj.acknowledge(via='input-any-key' if mode == 'input' else 'api')


class SeparateFollowerRegressionTests(unittest.TestCase):
    def rapid_prefix(self):
        tracker = ObjectVisitTracker()
        for at, ys in ((0.,[.55,.78,.9]), (.25,[.55,.78,.9]),
                       (.75,[.71,.86]), (1.,[.64,.79])):
            tracker.update([row(y) for y in ys], [zone()], 100,100,at)
        return tracker

    def test_reappearing_lead_keeps_a_distinct_box_beside_moving_follower(self):
        tracker = self.rapid_prefix()
        _, observed = tracker.update([row(.55),row(.57),row(.72)], [zone()],100,100,1.25)
        self.assertEqual([1,2,3], [obj['track_id'] for obj in observed])
        self.assertEqual(3, len(set(obj['track_id'] for obj in observed)))

    def test_recent_person_is_never_reassigned_to_missing_vehicle(self):
        tracker = self.rapid_prefix()
        _, observed = tracker.update([row(.57),row(.72),row(.55,label='person')],
                                     [zone(classes=['car','person'])],100,100,1.25)
        self.assertEqual([2,3,4], [obj['track_id'] for obj in observed])
        self.assertEqual('person', tracker._tracks[4].label)

    def test_missing_covisibility_keeps_geometric_fallback(self):
        tracker = self.rapid_prefix()
        tracker._tracks[2].co_visible_tracks.clear()
        _, observed = tracker.update([row(.57),row(.72)], [zone()],100,100,1.25)
        self.assertEqual([1,3], [obj['track_id'] for obj in observed])

    def test_motion_reservation_requires_fresh_strong_forward_trajectory(self):
        track = self.rapid_prefix()._tracks[2]
        box = (.4,.42,.6,.57)
        self.assertTrue(_recent_motion_supports(track, box, 1.25, 1.))
        for fields in ({'last_seen':.75}, {'raw_confidence':.49996},
                       {'trail':track.trail[:2]}, {'velocity':None},
                       {'velocity':(0.,0.)}, {'velocity':(0.,.2)},
                       {'velocity':(.2,0.)}, {'label':'person'}):
            with self.subTest(fields=fields):
                changed = copy.deepcopy(track)
                for key,value in fields.items():
                    setattr(changed,key,value)
                self.assertFalse(_recent_motion_supports(changed, box, 1.25, 1.))
        self.assertFalse(_recent_motion_supports(track, box, 1.751, 1.))
        self.assertFalse(_recent_motion_supports(track, box, 1.049, 1.))
        self.assertFalse(_recent_motion_supports(track, (.2,.42,.4,.57), 1.25, 1.))
        self.assertFalse(_recent_motion_supports(track, track.bbox, 1.25, 1.))

    def test_covisibility_metadata_is_bounded_to_live_track_ids_and_resets(self):
        tracker = ObjectVisitTracker(lost_after=.5)
        for index in range(150):
            tracker.update([row(),row(.85,x=.2)], [zone()],100,100,float(index))
            for offset in (.25,.5,.75):
                tracker.update([row()], [zone()],100,100,index+offset)
                for track in tracker._tracks.values():
                    self.assertLessEqual(len(track.co_visible_tracks), tracker.max_tracks)
                    self.assertTrue(track.co_visible_tracks <= set(tracker._tracks))
                    self.assertNotIn(track.id, track.co_visible_tracks)
            self.assertIn(1, tracker._tracks)
        tracker.reset()
        _, observed = tracker.update([row()], [zone()],100,100,151.)
        self.assertEqual(set(), tracker._tracks[observed[0]['track_id']].co_visible_tracks)

    def test_recent_moving_followers_keep_ids_past_absent_lead_boxes(self):
        z = zone(confirmation_polygon=[[.3,.2],[.7,.2],[.7,.6],[.3,.6]])
        flow = Flow([z])
        sequence = ((0.,[.55,.78,.9]), (.25,[.55,.78,.9]),
                    (.75,[.71,.86]), (1.,[.64,.79]), (1.25,[.57,.72]),
                    (1.5,[.53,.65]), (1.75,[.60]), (2.,[.56]))
        seen = []
        for at, ys in sequence:
            before = len(flow.alarms)
            observed = flow.step(at, [row(y) for y in ys])
            seen.append((at, [obj['track_id'] for obj in observed]))
            if len(flow.alarms) > before:
                self.assertTrue(flow.ack(at+.1, 'automatic'))
        self.assertEqual([(1.25,[2,3]), (1.5,[2,3]), (1.75,[3]), (2.,[3])], seen[-4:])
        self.assertEqual(3, len(flow.alarms), flow.trace)
        self.assertEqual(3, len({d['visit_id'] for _,d in flow.alarms}))

    def test_first_confirmation_keeps_all_three_covisible_vehicles_separate(self):
        z = zone(confirmation_polygon=[[.3,.2],[.7,.2],[.7,.6],[.3,.6]])
        flow = Flow([z])
        for at, ys in ((0.,[.55,.78,.9]), (.25,[.55,.78,.9]),
                       (.75,[.75,.88]), (1.25,[.70,.83]), (1.75,[.65,.78]),
                       (2.25,[.60,.73]), (2.75,[.55,.68]), (3.25,[.65]),
                       (3.75,[.62]), (4.25,[.59]), (4.75,[.56])):
            before = len(flow.alarms)
            flow.step(at, [row(y) for y in ys])
            if len(flow.alarms) > before:
                self.assertTrue(flow.ack(at+.25, 'automatic'))
        self.assertEqual(3, len(flow.alarms), flow.trace)
        self.assertEqual(3, len({d['track_id'] for _,d in flow.alarms}))
        self.assertEqual(3, len({d['visit_id'] for _,d in flow.alarms}))

    def test_same_tracker_in_another_route_keeps_independent_visit(self):
        flow = Flow([zone(), zone(name='Synthetic exit', route_group='synthetic-exit', route_role='exit')])
        flow.step(0., [row()]); flow.step(.25, [row()])
        self.assertEqual(1, len(flow.alarms))
        self.assertTrue(flow.ack(.5, 'automatic'))
        flow.step(.75, [row()])
        self.assertEqual(2, len(flow.alarms))
        self.assertEqual(flow.alarms[0][1]['track_id'], flow.alarms[1][1]['track_id'])
        self.assertNotEqual(flow.alarms[0][1]['route_group'], flow.alarms[1][1]['route_group'])
        self.assertNotEqual(flow.alarms[0][1]['visit_id'], flow.alarms[1][1]['visit_id'])

    def run_follower(self, *, mode='manual', count=2, blink=False, label='car'):
        z = zone(classes=[label], route_role='pedestrian' if label == 'person' else 'entrance',
            confirmation_polygon=[[.3,.2],[.7,.2],[.7,.6],[.3,.6]],
            arrival_direction=[0,-1], stationary_confirm_s=3., arrival_min_progress=.012,
            ai_confirm_observations=count)
        flow = Flow([z])
        for at in [0., .25] + [float(n) for n in range(1,12)]:
            flow.step(at, [row(label=label)])
            if len(flow.alarms) == 1 and flow.watcher.alarm and mode != 'unanswered':
                self.assertTrue(flow.ack(at+.25, mode))
        self.assertEqual(1, len(flow.alarms))
        lead = flow.alarms[0][1]
        sequence = [(12., [.55,.85]), (12.5,[.55,.78]), (13., [] if blink else [.71]),
                    (13.5,[.64]), (14.,[.57]), (14.5,[.53]), (15.,[.50]), (15.5,[.48])]
        for at, ys in sequence:
            flow.step(at, [row(y, label=label) for y in ys])
        self.assertEqual(2, len(flow.alarms), flow.trace)
        follower = flow.alarms[1][1]
        self.assertNotEqual(lead['track_id'], follower['track_id'])
        self.assertNotEqual(lead['visit_id'], follower['visit_id'])
        self.assertIn(str(follower['track_id']), flow.watcher.visits._visits[lead['visit_id']].separate_tracks)
        self.assertEqual({str(lead['track_id'])}, flow.watcher.visits._visits[lead['visit_id']].track_ids)
        self.assertGreater(flow.alarms[1][0], flow.alarms[0][0] + .25 + 10)
        self.assertEqual(2, sum(c.args == ('main',) for c in flow.watcher.buzzer.set.call_args_list))
        return flow

    def test_known_separate_follower_alarms_after_manual_quiet_expiry(self):
        self.run_follower()

    def test_known_separate_follower_alarms_after_automatic_service_ack(self):
        flow = self.run_follower(mode='automatic')
        self.assertEqual(0., flow.watcher._ack_until)

    def test_unanswered_lead_does_not_consume_follower_event(self):
        self.run_follower(mode='unanswered')

    def test_delayed_three_observation_follower_survives_brief_dropout(self):
        self.run_follower(count=3, blink=True)

    def test_separate_pedestrian_identity_does_not_inherit_manual_ack(self):
        flow = self.run_follower(label='person')
        self.assertTrue(flow.alarms[1][1]['manual_ack_required'])

    def test_same_car_lost_tracker_identity_keeps_ack(self):
        z = zone()
        flow = Flow([z])
        flow.step(0., [row()]); flow.step(.25, [row()])
        self.assertTrue(flow.ack(.5))
        # Tracker lifetime is five seconds; visit clear dwell is six seconds.
        for at in (1., 2., 3., 4., 5., 5.5):
            flow.step(at, [])
        flow.step(5.75, [row()]); flow.step(6., [row()])
        self.assertEqual(1, len(flow.alarms))
        self.assertEqual({'1','2'}, flow.watcher.visits._visits[flow.alarms[0][1]['visit_id']].track_ids)
        for at in (11., 11.25, 12.):
            flow.step(at, [row()])
        self.assertEqual(1, len(flow.alarms))

    def test_one_second_input_guard_and_ten_second_global_quiet_stay_exact(self):
        flow = Flow([zone()])
        flow.step(1., [row()]); flow.step(1.25, [row()])
        self.assertFalse(flow.ack(2.249, 'input'))
        self.assertTrue(flow.ack(2.25, 'input'))
        self.assertEqual(12.25, flow.watcher._ack_until)
        for at in (3., 3.25, 12.249):
            flow.step(at, [row(x=.75)])
        self.assertEqual(1, len(flow.alarms))
        flow.step(12.25, [row(x=.75)])
        self.assertEqual(2, len(flow.alarms))


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def adjacent32(value, steps):
    bits = struct.unpack('I', struct.pack('f', value))[0]
    return struct.unpack('f', struct.pack('I', bits+steps))[0]


class RawConfidenceRegressionTests(unittest.TestCase):
    def events(self, z, score, *, frames=2, legacy=False, moving=False):
        tracker = ObjectVisitTracker()
        found = []
        for index in range(frames):
            detection = row(.8-index*.03 if moving else .55, confidence=round(score, 4))
            if not legacy:
                detection['raw_confidence'] = score
            events, observed = tracker.update([detection], [z], 100, 100, 100.+index*.25)
            found.extend(tracker.event_context(e[2], [z]) for e in events)
            for e in events:
                tracker.mark_emitted(e[2])
        return found, observed

    def test_zone_gate_uses_raw_below_equal_above_float32_boundary(self):
        for threshold in (.35, .5, .55, .6, .7, .9):
            for steps in (-1, 0, 1):
                with self.subTest(threshold=threshold, steps=steps):
                    score = adjacent32(threshold, steps)
                    events, observed = self.events(zone(ai_confidence=threshold), score)
                    self.assertEqual(int(steps >= 0), len(events))
                    self.assertEqual(round(score, 4), observed[-1]['confidence'])
                    if events:
                        self.assertEqual(score, events[0].get('raw_confidence'))
                        self.assertEqual(round(score,4), events[0]['confidence'])

    def test_moving_floor_uses_raw_below_equal_above_float32_boundary(self):
        for threshold in (.35, .5, .55, .6):
            for steps in (-1, 0, 1):
                with self.subTest(threshold=threshold, steps=steps):
                    z = zone(ai_confidence=.9, ai_confidence_moving=threshold,
                             arrival_direction=[0,-1], arrival_min_progress=.012)
                    events, _ = self.events(z, adjacent32(threshold, steps), frames=4, moving=True)
                    self.assertEqual(int(steps >= 0), len(events))

    def test_strong_pair_uses_raw_below_equal_above_float32_boundary(self):
        for threshold in (.6, .7, .9):
            for steps in (-1, 0, 1):
                with self.subTest(threshold=threshold, steps=steps):
                    z = zone(ai_fast_confirm_confidence=threshold, ai_confirm_observations=3,
                        confirmation_polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]])
                    events, _ = self.events(z, adjacent32(threshold, steps))
                    self.assertEqual(int(steps >= 0), len(events))
                    if events:
                        self.assertTrue(_strong_stopline_proof(events[0], z, 100.25, 3.))

    def test_legacy_adapters_keep_display_threshold_and_proof(self):
        for threshold in (.35, .5, .55, .6, .7, .9):
            with self.subTest(threshold=threshold):
                events, _ = self.events(zone(ai_confidence=threshold), threshold, legacy=True)
                self.assertEqual(1, len(events))
                self.assertNotIn('raw_confidence', events[0])

    def test_invalid_explicit_raw_score_cannot_fall_back_to_display(self):
        for score in (True, None, '0.9', float('nan'), float('inf'), -.1, 1.1):
            with self.subTest(score=score):
                tracker = ObjectVisitTracker()
                for at in (100.,100.25):
                    events, observed = tracker.update([row(raw_confidence=score)], [zone()], 100,100,at)
                    self.assertEqual([], events)
                    self.assertEqual([], observed)

    def test_strong_pair_memory_rechecks_each_raw_score_and_last_score_binding(self):
        z = zone(ai_fast_confirm_confidence=.7, ai_confirm_observations=3,
            confirmation_polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]])
        events, _ = self.events(z, f32(.7))
        proof = events[0]
        self.assertTrue(_strong_stopline_proof(proof,z,100.25,3.))
        for index in (0,1):
            for score in (adjacent32(.7,-1),True,None,'0.7',float('nan'),float('inf')):
                with self.subTest(index=index, score=score):
                    broken = copy.deepcopy(proof)
                    broken['strong_confirmation'][index]['raw_confidence'] = score
                    self.assertFalse(_strong_stopline_proof(broken,z,100.25,3.))
        for score in (adjacent32(.7,1), None):
            changed = {**proof,'raw_confidence':score}
            self.assertFalse(_strong_stopline_proof(changed,z,100.25,3.))


if __name__ == '__main__':
    unittest.main()
