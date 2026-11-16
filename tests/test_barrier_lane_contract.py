"""Lane identity and current-visit proof using synthetic read-only services."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from barrier_workflow import BarrierWorkflow
from barrier_monitor import BarrierMonitor
from service_cycle import ServiceCycleMonitor


def observation(stamp, occupancy='occupied', barrier='down', **extra):
    return {'barrier': barrier, 'barrier_observed_at': stamp,
            'occupancy': occupancy, 'occupancy_observed_at': stamp,
            'model_ready': True, **extra}


class CurrentVisitProofTests(unittest.TestCase):
    def test_previous_car_occupancy_cannot_resolve_current_alarm(self):
        monitor = ServiceCycleMonitor()
        alarm = dict(id='current-car', route='campus-exit', started_at=10.)
        monitor.update({'campus-exit': observation(9.9)}, alarm, now=10.)
        actions = []
        for stamp in (10.1, 11.1, 12.1, 12.2):
            actions += monitor.update({'campus-exit': observation(stamp, 'clear')}, alarm, now=stamp)
        self.assertEqual([], actions)

    def test_fresh_current_occupancy_and_clear_carry_exact_proof(self):
        monitor = ServiceCycleMonitor()
        alarm = dict(id='current-car', route='campus-exit', started_at=10.)
        monitor.update({'campus-exit': observation(10.)}, alarm, now=10.)
        for stamp in (11., 12., 13.):
            actions = monitor.update({'campus-exit': observation(stamp, 'clear')}, alarm, now=stamp)
        self.assertEqual(1, len(actions))
        proof = actions[0]
        self.assertEqual('resolve', proof['type'])
        self.assertEqual(10., proof['alarm_started_at'])
        self.assertEqual(10., proof['occupied_at'])
        self.assertEqual(11., proof['clear_since'])
        self.assertEqual(13., proof['barrier_observed_at'])
        self.assertEqual('down', proof['barrier_state'])

    def test_old_current_alarm_anchor_expires_during_blind_gap(self):
        monitor = ServiceCycleMonitor()
        alarm = dict(id='current-car', route='campus-exit', started_at=10.)
        monitor.update({'campus-exit': observation(10.)}, alarm, now=10.)
        for stamp in (11., 20., 30., 40.):
            monitor.update({'campus-exit': observation(stamp, barrier='unknown')}, alarm, now=stamp)
        actions = monitor.update({'campus-exit': observation(41., barrier='up')}, alarm, now=41.)
        self.assertEqual([], actions)
        self.assertFalse(monitor.status(41.)['routes']['campus-exit']['raised_for_visit'])

    def test_candidate_before_origin_cannot_create_held_open_reminder(self):
        monitor = ServiceCycleMonitor()
        alarm = dict(id='current-car', route='campus-exit', started_at=10.)
        monitor.update({'campus-exit': observation(10.5)}, alarm, now=10.5)
        actions = monitor.update({'campus-exit': observation(
            11., barrier='up', barrier_position_since=10.2)}, alarm, now=11.)
        self.assertEqual([], actions)
        self.assertFalse(monitor.status(11.)['routes']['campus-exit']['raised_for_visit'])

    def test_barrier_reading_from_before_alarm_cannot_complete_clear_dwell(self):
        monitor = ServiceCycleMonitor()
        alarm = dict(id='current-car', route='campus-exit', started_at=10.)
        monitor.update({'campus-exit': observation(10., barrier_observed_at=9.9)}, alarm, now=10.)
        actions = []
        for stamp in (10.1, 11.1, 12.1):
            actions += monitor.update({'campus-exit': observation(
                stamp, 'clear', barrier_observed_at=9.9)}, alarm, now=stamp)
        self.assertEqual([], actions)


class LaneMappingTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.rows = []
        self.monitor = SimpleNamespace(update=lambda *a, **k: None,
            status=lambda: {'barriers': self.rows},
            road_clear=lambda *a: {'clear': True, 'reason': 'Synthetic verified empty road'})
        self.flow = BarrierWorkflow('synthetic', '.', None, monitor=self.monitor, clock=lambda: self.now)
        self.frame = np.zeros((4, 4, 3), np.uint8)

    @staticmethod
    def zone(name, route, role):
        return SimpleNamespace(name=name, route_group=route, route_role=role,
            polygon=None, rect=(0, 0, 1, 1), exclude_polygons=[],
            motion_active=False, light_active=False,
            mask_for=lambda width, height: np.ones((height, width), bool))

    def step(self, at, zones, alarm, occupied=True):
        self.now = at
        self.rows = [dict(id=arm, route_group=arm, state='down', fresh=True,
                         calibrated=True, observed_at=at, position_since=at)
                     for arm in ('exit', 'entrance')]
        primary = dict(available=True, state='ready', observed_at=at,
                       detections=[{'regions': [z.name for z in zones]}] if occupied else [])
        return self.flow.observe(self.frame, at, primary, zones, True, alarm)

    def test_footpath_cannot_borrow_vehicle_barriers_to_auto_resolve(self):
        zones = [self.zone('Exit', 'campus-exit', 'exit'),
                 self.zone('Entrance', 'campus-entrance', 'entrance'),
                 self.zone('Walkway', 'Pedestrian footpath', 'pedestrian')]
        alarm = dict(id='person', route='Pedestrian footpath', started_at=100.)
        self.step(100., zones, alarm)
        actions = []
        for at in (101., 102., 103.):
            actions += self.step(at, zones, alarm, occupied=False)
        self.assertEqual([], actions)
        status = self.flow.status()
        self.assertEqual('unknown', status['service']['routes']['Pedestrian footpath']['barrier'])
        self.assertIsNone(status['road_readings']['Pedestrian footpath']['barrier_id'])

    def test_pedestrian_named_exit_still_has_no_physical_arm(self):
        zones = [self.zone('Walkway', 'campus-exit', 'pedestrian')]
        self.step(100., zones, dict(id='person', route='campus-exit', started_at=100.))
        self.assertEqual('unknown', self.flow.status()['service']['routes']['campus-exit']['barrier'])

    def test_conflicting_lane_roles_do_not_silently_select_one_arm(self):
        zones = [self.zone('Exit', 'campus-exit', 'exit'),
                 self.zone('Entrance', 'campus-exit', 'entrance')]
        self.step(100., zones, dict(id='car', route='campus-exit', started_at=100.))
        self.assertEqual('unknown', self.flow.status()['service']['routes']['campus-exit']['barrier'])

    def test_vehicle_lane_still_gets_its_own_arm_and_clear_resolution(self):
        zones = [self.zone('Exit', 'campus-exit', 'exit')]
        alarm = dict(id='car', route='campus-exit', started_at=100.)
        self.step(100., zones, alarm)
        for at in (101., 102., 103.):
            actions = self.step(at, zones, alarm, occupied=False)
        self.assertEqual(['resolve'], [a['type'] for a in actions])
        self.assertEqual('exit', self.flow.status()['road_readings']['campus-exit']['barrier_id'])


class BarrierDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.monitor = BarrierMonitor(clock=lambda: self.now)
        self.monitor._counts[:, :, 0] = 1
        self.frame = np.repeat(np.tile((np.arange(640) % 200).astype(np.uint8), (360, 1))[..., None], 3, axis=2)

    def step(self, scores, contrast=20.):
        with mock.patch('barrier_monitor._features', return_value=(np.zeros((2, 4)), contrast)), \
                mock.patch('barrier_monitor._correlation', side_effect=(scores + [0., 0.]) * 2):
            return self.monitor.update(self.frame, self.now)['barriers'][0]

    def test_conflicting_positions_are_unknown_with_comparable_scores(self):
        row = self.step([.9, .9])
        self.assertEqual('unknown', row['state'])
        self.assertIn('conflict', row['reason'])
        self.assertEqual(.9, row['match'])
        self.assertEqual(.9, row['opposite_match'])
        self.assertIsNone(row['candidate_position'])

    def test_confirmation_reports_source_span_without_claiming_both_strips_match(self):
        row = self.step([.9, 0.])
        self.assertEqual('down', row['candidate_position'])
        self.assertEqual('unknown', row['state'])
        self.assertEqual(0., row['confirmation_s'])
        self.now += .4
        row = self.step([.9, 0.])
        self.assertEqual('down', row['state'])
        self.assertEqual(.4, row['confirmation_s'])
        self.assertEqual('Calibrated arm position confirmed', row['reason'])

    def test_low_contrast_reason_does_not_misrepresent_a_match(self):
        row = self.step([.9, 0.], contrast=1.)
        self.assertEqual('unknown', row['state'])
        self.assertIn('contrast', row['reason'])


if __name__ == '__main__':
    unittest.main()
