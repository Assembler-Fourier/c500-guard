"""Campus fork/parking sequences and current evidence for deferred alarms."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from object_tracking import ObjectVisitTracker


def zone(**overrides):
    values = dict(name='Entrance approach', rect=(.05, .05, .9, .9),
                  ai_enabled=True, classes=['car'], severity='main',
                  ai_confirm_observations=2, ai_confidence=.6,
                  arrival_direction=[1, 0], arrival_min_progress=.012,
                  stationary_confirm_s=3., route_group='entrance',
                  confirmation_polygon=[[.5,.2],[.85,.2],[.85,.8],[.5,.8]],
                  route_role='entrance', operator_hint='Check entrance barrier')
    values.update(overrides)
    return SimpleNamespace(**values)


def car(x=.2, y=.5, *, confidence=.9):
    # Inputs are the wheel/ground point; the box can overlap the gate before
    # the vehicle actually commits to the barrier lane.
    return {'label':'car', 'confidence':confidence,
            'bbox':[(x-.08)*100,(y-.16)*100,(x+.08)*100,y*100]}


class CampusTrafficTests(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectVisitTracker()
        self.zones = [zone()]
        self.now = 100.

    def step(self, x=.2, y=.5, *, after=.5, objects=None):
        self.now += after
        if objects is None:
            objects = [car(x,y)]
        return self.tracker.update(objects, self.zones, 100, 100, self.now)

    def commit(self):
        for x in [.25,.32,.39,.46,.52]:
            events, observed = self.step(x)
            self.assertEqual([], events)
        events, observed = self.step(.55)
        self.assertEqual(1, len(events))
        return events[0][2], observed[0]['track_id']

    def test_shared_curve_toward_motion_remains_tracking_until_commitment(self):
        first_id = None
        for x in [.2,.25,.3,.35,.4,.45]:
            events, observed = self.step(x)
            self.assertEqual([], events)
            first_id = first_id or observed[0]['track_id']
            self.assertEqual(first_id, observed[0]['track_id'])
            self.assertEqual('tracking approach', observed[0]['region_status']['Entrance approach'])
            self.assertEqual([], observed[0]['confirmed_regions'])
        # The box already overlaps the commitment area at .45: only its
        # bottom-center decides whether the vehicle has entered that lane.
        self.assertGreater(car(.45)['bbox'][2], 50)
        self.assertEqual([], self.step(.51)[0])
        events, observed = self.step(.55)
        self.assertEqual(1, len(events))
        self.assertEqual(first_id, observed[0]['track_id'])

    def test_car_waiting_in_shared_road_never_uses_stationary_fallback(self):
        for _ in range(30):
            events, observed = self.step(.3)
            self.assertEqual([], events)
        self.assertEqual('tracking approach', observed[0]['region_status']['Entrance approach'])
        self.assertEqual('waiting', observed[0]['direction']['Entrance approach'])

    def test_parking_turn_never_commits_to_barrier(self):
        for x,y in [(.2,.5),(.27,.5),(.34,.5),(.41,.5),(.43,.6),
                    (.43,.7),(.43,.8),(.36,.8),(.3,.8)]:
            self.assertEqual([], self.step(x,y)[0])
        for _ in range(12):
            self.assertEqual([], self.step(.3,.8)[0])

    def test_existing_zone_without_gate_keeps_normal_confirmation(self):
        self.zones = [zone(confirmation_polygon=None)]
        self.assertEqual([], self.step(.2)[0])
        self.assertEqual(1, len(self.step(.24)[0]))

    def test_stationary_vehicle_first_seen_in_commitment_area_still_alerts(self):
        for _ in range(6):
            self.assertEqual([], self.step(.6)[0])
        events, observed = self.step(.6)
        self.assertEqual(1, len(events))
        self.assertEqual('stationary', observed[0]['direction']['Entrance approach'])

    def test_zero_stationary_timeout_disables_advance_fallback(self):
        self.zones = [zone(stationary_confirm_s=0, alert_stage='advance')]
        for _ in range(30):
            events, observed = self.step(.6)
            self.assertEqual([], events)
        self.assertEqual('waiting', observed[0]['direction']['Entrance approach'])
        events, _ = self.step(.63)
        self.assertEqual(1, len(events), 'Observed arrival movement remains enabled')
        evidence = self.tracker.event_context(events[0][2], self.zones)
        self.assertEqual('advance', evidence['alert_stage'])

    def test_positive_stationary_timeout_remains_backward_compatible(self):
        self.zones = [zone(stationary_confirm_s=3)]
        for _ in range(6):
            self.assertEqual([], self.step(.6)[0])
        events, _ = self.step(.6)
        self.assertEqual('', self.tracker.event_context(events[0][2], self.zones)['alert_stage'])

    def test_one_frame_gate_jitter_does_not_sound(self):
        for x in [.3,.37,.44,.51,.48,.52]:
            self.assertEqual([], self.step(x)[0])
        # New toward movement after the meaningful retreat restores eligibility.
        self.assertEqual(1, len(self.step(.55)[0]))

    def test_pending_arrival_withdraws_after_meaningful_reverse_even_above_origin(self):
        event_id, track_id = self.commit()
        self.assertEqual(event_id, self.step(.63)[0][0][2])
        events, observed = self.step(.59)
        self.assertEqual([], events)
        self.assertEqual(track_id, observed[0]['track_id'])
        self.assertEqual('moving away', observed[0]['region_status']['Entrance approach'])
        self.assertEqual([], observed[0]['confirmed_regions'])
        self.assertIsNone(self.tracker.event_context(event_id, self.zones))
        self.assertFalse(self.tracker.mark_emitted(event_id))
        for _ in range(12):  # An ACK quiet period can finish while it waits here.
            self.assertEqual([], self.step(.59)[0])

    def test_pending_arrival_can_resume_after_observed_turn_back_toward_barrier(self):
        event_id, _ = self.commit()
        self.step(.63)
        self.assertEqual([], self.step(.59)[0])
        self.assertEqual([], self.step(.595)[0])
        events, _ = self.step(.62)
        self.assertEqual(event_id, events[0][2])

    def test_pending_arrival_moving_outside_gate_is_not_deliverable(self):
        event_id, _ = self.commit()
        for y in [.6,.7,.81]:
            events, observed = self.step(.55,y)
        self.assertEqual([], events)
        self.assertEqual('tracking approach', observed[0]['region_status']['Entrance approach'])
        self.assertIsNone(self.tracker.event_context(event_id, self.zones))
        self.assertFalse(self.tracker.mark_emitted(event_id))

    def test_small_ground_jitter_does_not_withdraw_waiting_arrival(self):
        event_id, _ = self.commit()
        for x in [.548,.553,.547,.551,.55]:
            events, _ = self.step(x)
            self.assertEqual(event_id, events[0][2])

    def test_consumed_route_is_not_rearmed_by_gate_exit_or_turnaround(self):
        event_id, track_id = self.commit()
        self.assertTrue(self.tracker.mark_emitted(event_id))
        for x in [.59,.55,.48,.42,.43,.49,.54,.6]:
            events, observed = self.step(x)
            self.assertEqual([], events)
            self.assertEqual(track_id, observed[0]['track_id'])

    def test_evidence_states_which_operator_action_and_gate_confirmed(self):
        event_id, track_id = self.commit()
        evidence = self.tracker.event_context(event_id, self.zones)
        self.assertEqual('entrance', evidence['route_role'])
        self.assertEqual('Check entrance barrier', evidence['operator_hint'])
        self.assertTrue(evidence['confirmation_required'])
        self.assertEqual(2, evidence['observations'])
        self.assertEqual(track_id, evidence['track_id'])

    def test_gate_does_not_override_outer_region_exclusions(self):
        self.zones = [zone(contains_point=lambda x,y: .05 <= x <= .95 and y < .6)]
        for _ in range(12):
            self.assertEqual([], self.step(.6,.7)[0])

    def test_low_confidence_or_absence_invalidates_stale_delivery_evidence(self):
        event_id, _ = self.commit()
        self.assertEqual([], self.step(objects=[car(.55, confidence=.5)])[0])
        self.assertIsNone(self.tracker.event_context(event_id, self.zones))
        self.assertFalse(self.tracker.mark_emitted(event_id))
        self.assertEqual([], self.step(.55)[0])
        self.assertEqual(event_id, self.step(.55)[0][0][2])
        self.assertEqual([], self.step(objects=[])[0])
        self.assertIsNone(self.tracker.event_context(event_id, self.zones))


if __name__ == '__main__':
    unittest.main()
