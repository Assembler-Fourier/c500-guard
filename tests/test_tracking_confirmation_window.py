"""Opt-in brief-glare confirmation; synthetic sensitivity, no recall claim."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from object_tracking import ObjectVisitTracker
from zone_editor import validate_config


def zone(window=1.5):
    return SimpleNamespace(name='Approach',rect=(.2,.2,.6,.6),classes=['car'],
        ai_enabled=True,severity='main',route_group='entrance',
        ai_confidence=.55,ai_confirm_observations=2,ai_confirmation_window_s=window,
        arrival_direction=[1,0],arrival_min_progress=.012,stationary_confirm_s=0)


def car(x,confidence=.7):
    return {'label':'car','confidence':confidence,'bbox':[(x-.05)*640,.3*360,(x+.05)*640,.4*360]}


class ConfirmationWindowTests(unittest.TestCase):
    def replay(self, frames, window=1.5):
        tracker=ObjectVisitTracker();events=[];z=[zone(window)]
        for at,objects in frames:
            found,_=tracker.update(objects,z,640,360,100+at)
            for event in found:
                events.append((at,tracker.event_context(event[2],z)))
                tracker.mark_emitted(event[2])
        return events

    def test_two_qualified_moving_observations_survive_brief_missing_and_weak_frames(self):
        # Rounded overlay55% could be below .55; it does not count here.
        frames=[(0,[car(.4,.70)]),(.5,[car(.43,.548)]),(1,[]),(1.5,[car(.48,.57)])]
        self.assertEqual([],self.replay(frames,0))
        events=self.replay(frames)
        self.assertEqual(1,len(events))
        self.assertEqual(1.5,events[0][0])
        self.assertEqual(2,events[0][1]['observations'])

    def test_confirmation_expires_after_motion_horizon(self):
        self.assertEqual([],self.replay([(0,[car(.4)]),(.5,[]),(1.501,[car(.48)])]))

    def test_missing_or_weak_current_reading_never_emits(self):
        self.assertEqual([],self.replay([(0,[car(.4)]),(.5,[]),(1,[car(.48,.54)]),(1.5,[])]))

    def test_visible_outside_observation_clears_old_confirmation(self):
        z=zone();z.rect=(.2,.2,.25,.6)
        tracker=ObjectVisitTracker()
        for at,objects in [(0,[car(.43)]),(.5,[car(.46)]),(1,[car(.44)])]:
            events,_=tracker.update(objects,[z],640,360,100+at)
            self.assertEqual([],events)

    def test_departure_and_turnback_need_fresh_reversal_evidence(self):
        frames=[(0,[car(.5)]),(.3,[car(.46)]),(.6,[]),(.9,[car(.5)]),(1.2,[car(.5)])]
        self.assertEqual([],self.replay(frames))
        frames.append((1.5,[car(.53)]));frames.append((1.8,[car(.56)]));frames.append((2.1,[car(.59)]))
        self.assertEqual(1,len(self.replay(frames)))

    def test_parked_readings_cannot_use_expired_movement(self):
        frames=[(0,[car(.4)]),(.5,[]),(2,[car(.5)]),(2.5,[]),(3,[car(.501)])]
        self.assertEqual([],self.replay(frames))

    def test_duplicate_frame_timestamp_is_not_a_second_observation(self):
        self.assertEqual([],self.replay([(0,[car(.4)]),(0,[car(.48)])]))

    def test_after_emission_missing_and_recovery_do_not_repeat(self):
        frames=[(0,[car(.4)]),(.5,[]),(1,[car(.44)]),(1.5,[]),(2,[car(.48)]),(2.5,[car(.52)])]
        self.assertEqual(1,len(self.replay(frames)))

    def test_window_is_validated_and_zero_keeps_legacy_policy(self):
        data={'zones':[{'name':'Approach','rect':[.2,.2,.6,.6], 'ai_confirmation_window_s':1.5}]}
        self.assertEqual(1.5,validate_config(data)['zones'][0]['ai_confirmation_window_s'])
        for value in (-.1,1.501,float('nan'),True):
            data['zones'][0]['ai_confirmation_window_s']=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_config(data)


if __name__=='__main__':unittest.main()
