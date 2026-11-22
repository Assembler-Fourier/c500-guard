"""Stable display endpoints independent of recognition-strip/angle jitter."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arm_angle import display_barrier_row, geometry_from_config


class ArmDisplayGeometryTests(unittest.TestCase):
    def setUp(self):
        self.saved = {'pivot': [.643, .452], 'down_tip': [.497, .439], 'up_tip': [.647, .195]}
        self.geometry = geometry_from_config(self.saved)
        self.old_line = [[.655, .13], [.638, .44]]
        self.row = {'id': 'exit', 'state': 'up', 'fresh': True, 'calibrated': True,
                    'confidence': .92, 'active_line': self.old_line,
                    'candidate_position': 'up', 'reason': 'Calibrated arm position confirmed'}

    def test_settled_endpoint_never_jumps_back_to_old_recognition_strip(self):
        expected = [self.saved['up_tip'], self.saved['pivot']]
        for reading in (None, {'ok': False, 'line': None},
                        {'ok': True, 'theta': 85., 'shown': 85., 'line': [[.62, .20], [.63, .45]]}):
            with self.subTest(reading=reading):
                result = display_barrier_row(self.row, reading, self.geometry, reading_fresh=True)
                self.assertEqual(expected, result['live_line'])
                self.assertEqual(expected, result['active_line'])
                self.assertEqual(90., result['live_angle'])
                self.assertEqual('up', result['state'])
                self.assertEqual(.92, result['confidence'])

    def test_down_endpoint_uses_full_display_geometry_even_when_angle_is_unreadable(self):
        self.row.update(state='down', active_line=[[.497, .439], [.622, .452]])
        result = display_barrier_row(self.row, {'ok': False}, self.geometry, reading_fresh=True)
        self.assertEqual([self.saved['down_tip'], self.saved['pivot']], result['live_line'])
        self.assertEqual(0., result['live_angle'])

    def test_genuine_moving_line_is_provisional_and_never_claims_endpoint(self):
        self.row.update(state='unknown', confidence=0.)
        reading = {'ok': True, 'theta': 40., 'shown': 35., 'motion': 'rising',
                   'line': [[.55, .30], [.643, .452]], 'motion_started': 100., 'at': 101.}
        before = copy.deepcopy((self.row, reading))
        result = display_barrier_row(self.row, reading, self.geometry, reading_fresh=True)
        self.assertEqual(reading['line'], result['live_line'])
        self.assertEqual(35., result['live_angle'])
        self.assertEqual('rising', result['live_motion'])
        self.assertEqual('unknown', result['state'])
        self.assertEqual(0., result['confidence'])
        self.assertIsNone(result['active_line'])
        self.assertEqual(before, (self.row, reading), 'presentation must not rewrite ACK inputs')

    def test_uncertain_static_angle_cannot_invent_a_line_or_use_recognition_fallback(self):
        self.row.update(state='unknown')
        result = display_barrier_row(self.row, {'ok': True, 'theta': 60., 'shown': 60.,
            'line': [[.55, .3], [.643, .452]], 'motion': None}, self.geometry, reading_fresh=True)
        self.assertEqual('unknown', result['state'])
        self.assertIsNone(result['live_line'])
        self.assertIsNone(result['active_line'])

    def test_fresh_conflict_remains_unknown_and_cannot_draw_either_guess(self):
        reading = {'ok': True, 'theta': 0., 'line': [[.497, .439], [.643, .452]],
                   'position_conflict': True}
        result = display_barrier_row(self.row, reading, self.geometry, reading_fresh=True)
        self.assertEqual('unknown', result['state'])
        self.assertEqual('up', result['classifier_state'])
        self.assertEqual(0., result['confidence'])
        self.assertIsNone(result['live_line'])
        self.assertIsNone(result['active_line'])
        self.assertTrue(reading['position_conflict'])

    def test_stale_angle_cannot_override_fresh_endpoint_or_draw_motion(self):
        reading = {'ok': True, 'theta': 40., 'shown': 40., 'motion': 'lowering',
                   'line': [[.55, .3], [.643, .452]], 'position_conflict': True}
        result = display_barrier_row(self.row, reading, self.geometry, reading_fresh=False)
        self.assertEqual('up', result['state'])
        self.assertEqual([self.saved['up_tip'], self.saved['pivot']], result['live_line'])
        self.row['state'] = 'unknown'
        self.assertIsNone(display_barrier_row(self.row, reading, self.geometry)['live_line'])

    def test_stale_or_uncalibrated_strict_reading_never_claims_known_endpoint(self):
        reading = {'ok': True, 'shown': 40., 'motion': 'rising', 'line': [[.55, .3], [.643, .452]]}
        for change in ({'fresh': False}, {'calibrated': False}):
            with self.subTest(change=change):
                result = display_barrier_row({**self.row, **change}, reading, self.geometry, reading_fresh=True)
                self.assertEqual('unknown', result['state'])
                self.assertIsNone(result['live_line'])
                self.assertIsNone(result['active_line'])

    def test_bad_configured_geometry_never_falls_back_to_recognition_lines(self):
        for geometry in ({}, {'pivot': [float('nan'), 0]},
                         {**self.geometry, 'up_tip': [1000, -10]}):
            with self.subTest(geometry=geometry):
                result = display_barrier_row(self.row, None, geometry)
                self.assertIsNone(result['live_line'])
                self.assertIsNone(result['active_line'])
                self.assertEqual('up', result['state'], 'bad drawing geometry does not invalidate recognition')

    def test_legacy_without_display_geometry_retains_valid_endpoint_only(self):
        result = display_barrier_row(self.row, None, None)
        self.assertEqual(self.old_line, result['live_line'])
        self.row.update(state='unknown')
        self.assertIsNone(display_barrier_row(self.row, None, None)['live_line'])


if __name__ == '__main__':
    unittest.main()
