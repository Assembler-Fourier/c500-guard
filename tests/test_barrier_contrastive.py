"""A static opposite-position strip cannot prove a physical arm moved."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from test_watch import watch, watcher, ROOT
import barrier_monitor as bm


class ContrastiveArmTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.monitor = bm.BarrierMonitor(clock=lambda: self.now)
        rng = np.random.default_rng(724)
        def feature():
            item = rng.normal(size=bm.FEATURE_SHAPE).astype(np.float32)
            item -= item.mean()
            return item / np.linalg.norm(item)
        self.background = feature()
        other = feature()
        other -= np.sum(other*self.background)*self.background
        other /= np.linalg.norm(other)
        self.arm = .6*self.background + .8*other
        self.down = feature()
        self.live_down = feature()
        self.monitor._counts[:, :, 0] = 4
        for index in range(2):
            self.monitor._templates[index, 0, 0] = np.stack([self.down, self.background])
            self.monitor._templates[index, 1, 0] = np.stack([self.live_down, self.arm])
        self.frame = rng.integers(40, 200, (360, 640, 3), dtype=np.uint8)

    def feed(self, up):
        with mock.patch.object(bm, '_features', return_value=(np.stack([self.live_down, up]), 20.)):
            for _ in range(3):
                self.now += .2
                self.monitor.update(self.frame, self.now)
        return self.monitor.status()['barriers']

    def test_stationary_background_does_not_become_up_when_down_lighting_changes(self):
        rows = self.feed(self.background)
        for row in rows:
            self.assertEqual('unknown', row['state'])
            self.assertGreater(row['match'], bm.MIN_CORRELATION)
            self.assertGreater(row['match']-row['opposite_match'], bm.MIN_CORRELATION_MARGIN)
            self.assertGreater(row['background_match'], row['match'])
            self.assertIn('opposite-position background', row['reason'])

    def test_real_arm_appearance_can_still_confirm_up(self):
        for row in self.feed(self.arm):
            self.assertEqual('up', row['state'])
            self.assertGreater(row['match'], row['background_match'])

    def test_equally_matching_presence_and_background_is_unknown(self):
        self.monitor._templates[:, 0, 0, 1] = self.arm
        self.assertTrue(all(row['state'] == 'unknown' for row in self.feed(self.arm)))

    def test_normalized_comparison_requires_substantial_improvement_not_just_a_tiny_lead(self):
        self.assertEqual(-1., bm._position_evidence(.54, -.8), 'Absolute correlation floor still applies')
        self.assertEqual(0., bm._position_evidence(.8, .8))
        self.assertEqual(0., bm._position_evidence(1., 1.))
        self.assertLess(bm._position_evidence(.90, .89), bm.MIN_CORRELATION)
        self.assertAlmostEqual(.875, bm._position_evidence(.90, .20))

    def test_different_static_backgrounds_do_not_block_correct_down(self):
        # Both raw strips correlate well, but only DOWN improves upon its
        # opposite-position calibration. The old across-strip .25 margin failed.
        with mock.patch.object(bm, '_features', return_value=(np.zeros((2, 4)), 20.)), \
                mock.patch.object(bm, '_correlation', side_effect=[.998, .79, .61, .999] * 6):
            for _ in range(3):
                self.now += .2
                self.monitor.update(self.frame, self.now)
        for row in self.monitor.status()['barriers']:
            self.assertEqual('down', row['state'])
            self.assertLess(row['match']-row['opposite_match'], bm.MIN_CORRELATION_MARGIN)
            self.assertGreater(row['discriminative_match'], .99)
            self.assertLess(row['opposite_discriminative_match'], 0.)


class DisplayContradictionTests(unittest.TestCase):
    def test_current_angle_conflict_never_draws_fallback_up_or_mutates_classifier(self):
        row = dict(id='entrance', state='up', fresh=True, active_line=[[.6,.1],[.6,.4]],
                   confidence=.8, candidate_position='up')
        status = {'barriers': [row]}
        tracker = SimpleNamespace(update=lambda *_: {'entrance': {
            'theta': 0., 'shown': 0., 'ok': True, 'line': [[.6,.4],[.8,.4]], 'motion': None}})
        obj = SimpleNamespace(_arm_tracker=tracker, _arm_last_run=0., alarm=None,
                              _arm_motion_recorder=None, _ack_on_arm_motion=mock.Mock(),
                              barrier_workflow=SimpleNamespace(status=lambda: status),
                              _barrier_error='', _arm_memory_status=lambda: {})
        with mock.patch.object(watch.time, 'time', return_value=1000.), \
                mock.patch.object(watch.time, 'monotonic', return_value=100.):
            watch.Watcher._observe_arm_angles(obj, np.zeros((4,4,3)), 100., True, status)
            result = watch.Watcher.barrier_status(obj)['barriers'][0]
        self.assertEqual('unknown', result['state'])
        self.assertIsNone(result['active_line'])
        self.assertIsNone(result['live_line'])
        self.assertEqual('up', result['classifier_state'])
        self.assertIn('disagrees', result['reason'])
        self.assertEqual('up', row['state'], 'Presentation must not rewrite service observations')
        obj._ack_on_arm_motion.assert_not_called()


class ConflictAcknowledgementTests(unittest.TestCase):
    def setUp(self):
        self.now, self.wall = 103.25, 1000.
        for patcher in (mock.patch.object(watch.time, 'monotonic', side_effect=lambda:self.now),
                        mock.patch.object(watch.time, 'time', side_effect=lambda:self.wall),
                        mock.patch.object(watch, 'notify')):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.obj = watcher(zones_path=ROOT/'config/views/ptz2/zones.json')
        self.obj.armed = True
        self.obj._pending_rearm = self.obj._alignment_blocked = False
        self.obj.alignment = {'state':'tracking'}
        self.obj.alarm, self.obj.alarm_id, self.obj.alarm_since = 'main', 'current-car', 100.
        self.obj.alarm_reason = 'Exit approach: car'
        self.obj.alarm_details = dict(route_group='campus-exit', route_role='exit', source='ai')
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'trusted':True,'barriers':[],'service':{'routes':{}}}
        self.obj.buzzer.set('main')
        self.obj.loudness = mock.Mock(kind='main')
        self.obj.visits.acknowledge = mock.Mock()
        self.obj._arm_live = {'exit':dict(ok=True,theta=0.,shown=90.,position_conflict=True,at_wall=self.wall)}

    def assert_unanswered(self):
        self.assertEqual('main', self.obj.alarm)
        self.assertEqual('current-car', self.obj.alarm_id)
        self.assertEqual('main', self.obj.buzzer.want)
        self.obj.visits.acknowledge.assert_not_called()
        self.obj.loudness.end.assert_not_called()

    def test_smoothed_up_cannot_override_current_raw_conflict_in_any_ack_path(self):
        self.assertIs(False, self.obj._tracker_agrees('exit','up'))
        movement = dict(type='raised',alarm_id='current-car',route='campus-exit',
                        alarm_started_at=100.,origin_observed_at=101.,position_since=102.,observed_at=103.2)
        self.assertFalse(self.obj._silence_for_barrier_change(movement))
        clear = dict(type='resolve',alarm_id='current-car',route='campus-exit',
                     alarm_started_at=100.,occupied_at=100.,clear_since=101.,observed_at=103.,
                     barrier_observed_at=103.2,barrier_state='up')
        self.assertFalse(self.obj._resolve_service_visit(clear))
        self.obj._arm_last_known = {'exit':('down',101.)}
        status = dict(trusted=True,barriers=[dict(id='exit',state='up',fresh=True,calibrated=True,
                                               observed_at=103.2,position_since=102.)])
        self.assertFalse(self.obj._ack_on_any_arm(status,103.2))
        self.assertEqual({'exit':('down',101.)}, self.obj._arm_last_known)
        self.assert_unanswered()

    def test_conflict_cannot_establish_or_renew_an_origin_for_a_later_transition(self):
        status = dict(trusted=True,barriers=[dict(id='exit',state='up',fresh=True,calibrated=True,
                                               observed_at=103.2,position_since=102.)])
        for known in ({}, {'exit':('up',101.)}):
            self.obj._arm_last_known = dict(known)
            self.assertFalse(self.obj._ack_on_any_arm(status,103.2))
            self.assertEqual(known,self.obj._arm_last_known)
        self.assert_unanswered()

    def test_conflict_veto_lasts_as_long_as_unknown_display_even_without_shown_angle(self):
        self.wall += 1.4
        self.obj._arm_live['exit']['shown'] = None
        self.assertIs(False, self.obj._tracker_agrees('exit','up'))
        self.wall += .2
        self.assertIsNone(self.obj._tracker_agrees('exit','up'))

    def test_conflicting_early_movement_cannot_acknowledge(self):
        self.obj._arm_memory = {'exit':{'state':'down'}}
        reading = dict(ok=True,motion='rising',motion_started=102.,at=103.2,
                       strict='up',position_conflict=True)
        self.assertFalse(self.obj._ack_on_arm_motion('exit',reading))
        self.assert_unanswered()


def private_evidence():
    # Optional, separately authorized evidence; never search a user workspace.
    import os
    supplied = os.environ.get("BARRIERWATCHER_TEST_EVIDENCE")
    folder = Path(supplied) if supplied else None
    return folder if folder and (folder/"templates-before-incident.npz").is_file() else None


@unittest.skipUnless(private_evidence(), 'Private labelled PTZ incident evidence unavailable')
class LabelledPtzDownTests(unittest.TestCase):
    def test_six_user_confirmed_down_frames_never_report_up_from_old_templates(self):
        folder = private_evidence()
        cfg = json.loads((ROOT/'config/views/ptz2/zones.json').read_text(encoding='utf-8-sig'))
        seed = folder/'templates-before-incident.npz'
        with np.load(seed, allow_pickle=False) as archive:
            source_key = str(archive['source_key'].item())
        now = 100.
        monitor = bm.BarrierMonitor(seed_path=seed, source_key=source_key,
                                    lines=cfg['barriers']['lines'], clock=lambda: now)
        labelled = folder/'labelled-down'
        for index in range(6):
            frame = np.asarray(Image.open(labelled/f'scene-{index}.jpg').convert('RGB'))
            for _ in range(3):
                now += .2
                monitor.update(frame, now)
            rows = monitor.status()['barriers']
            self.assertTrue(all(row['state'] != 'up' for row in rows), rows)
            self.assertEqual('unknown', rows[1]['state'])
            self.assertIn('background', rows[1]['reason'])

    def test_new_user_calibrations_recognize_six_independent_confirmed_down_frames(self):
        folder = private_evidence()
        cfg = json.loads((ROOT/'config/views/ptz2/zones.json').read_text(encoding='utf-8-sig'))
        seed = folder/'templates-user-225605.npz'
        with np.load(seed, allow_pickle=False) as archive:
            source_key = str(archive['source_key'].item())
        now = 100.
        monitor = bm.BarrierMonitor(seed_path=seed, source_key=source_key,
                                    lines=cfg['barriers']['lines'], clock=lambda: now)
        labelled = folder/'labelled-down'
        for index in range(6):
            frame = np.asarray(Image.open(labelled/f'scene-{index}.jpg').convert('RGB'))
            for gain, offset in ((1., 0.), (1.08, 6.), (.8, 6.)):
                adjusted = np.clip(frame.astype(float)*gain+offset, 0, 255).astype(np.uint8)
                for _ in range(3):
                    now += .2
                    monitor.update(adjusted, now)
                rows = monitor.status()['barriers']
                self.assertEqual(['down', 'down'], [row['state'] for row in rows], (index, gain, rows))


if __name__ == '__main__':
    unittest.main()
