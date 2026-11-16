"""Barrier self-learning must not learn glare, must say what it learned, and must be undoable."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from test_watch import watch, watcher  # noqa: F401 - also puts src on the path
from barrier_workflow import BarrierWorkflow
from barrier_monitor import BarrierMonitor


class Clock:
    def __init__(self):
        self.now = 1000.

    def __call__(self):
        return self.now


def status(confidence=.8):
    return {'barriers': [{'id': 'exit', 'state': 'down', 'fresh': True, 'calibrated': True,
                          'confidence': confidence}]}


class LearningGateTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.store = Path(self.folder.name) / 'barriers.npz'
        self.store.write_bytes(b'version-1')
        self.monitor = mock.Mock()
        self.monitor.storage_path = self.store
        self.monitor._observations = {'exit': {'best': .8, 'candidate_at': 0.}}
        self.monitor.learning_conditions.return_value = {'glare': 0., 'light_range': 3., 'frames': 9}

        def capture(identity, position):
            self.store.write_bytes(b'version-2')
            return {'ok': True}
        self.monitor.capture.side_effect = capture
        self.flow = BarrierWorkflow('test', self.folder.name, None, monitor=self.monitor, clock=Clock())
        self.flow.on_learned = mock.Mock()

    def tearDown(self):
        self.folder.cleanup()

    def test_low_confidence_glare_or_swinging_light_is_never_learned(self):
        self.flow._learn_arms(status(confidence=.42), 2000.)
        self.monitor.capture.assert_not_called()
        for light in ({'glare': .05, 'light_range': 3.}, {'glare': 0., 'light_range': 30.}):
            with self.subTest(light=light):
                self.monitor.learning_conditions.return_value = light
                self.flow._learn_arms(status(), 2000.)
                self.monitor.capture.assert_not_called()
                self.assertIn('glare', self.flow.learning_skips[('exit', 'down')]['reason'])

    def test_learned_example_is_reported_and_can_be_undone(self):
        self.flow._learn_arms(status(), 2000.)
        self.monitor.capture.assert_called_once_with('exit', 'down')
        entry = self.flow.on_learned.call_args.args[0]
        self.assertEqual(('exit', 'down', .8), (entry['arm'], entry['position'], entry['confidence']))
        self.assertEqual(b'version-2', self.store.read_bytes())
        result = self.flow.undo_last_learning()
        self.assertTrue(result['ok'])
        self.assertEqual(b'version-1', self.store.read_bytes())
        self.monitor.reload.assert_called_once()
        self.assertEqual([], self.flow.learned)
        self.assertFalse(self.flow.undo_last_learning()['ok'], 'nothing left to undo')

    def test_refused_capture_leaves_no_backup(self):
        self.monitor.capture.side_effect = None
        self.monitor.capture.return_value = {'ok': False}
        self.flow._learn_arms(status(), 2000.)
        self.assertEqual([], list(self.store.parent.glob('barriers.npz.before-learn-*')))

    def test_failed_backup_prevents_automatic_template_mutation(self):
        with mock.patch('barrier_workflow.shutil.copyfile', side_effect=OSError('disk unavailable')):
            self.flow._learn_arms(status(), 2000.)
        self.monitor.capture.assert_not_called()
        self.assertEqual(b'version-1', self.store.read_bytes())
        self.assertIn('preserve', self.flow.learning_skips[('exit', 'down')]['reason'])

    def test_two_arms_learned_in_same_second_have_two_reversible_backups(self):
        self.monitor._observations['entrance'] = {'best': .8, 'candidate_at': 0.}
        rows = status()['barriers'] + [{**status()['barriers'][0], 'id': 'entrance'}]
        captures = []
        def capture(identity, position):
            captures.append(identity)
            self.store.write_bytes(f'version-{len(captures)+1}'.encode())
            return {'ok': True}
        self.monitor.capture.side_effect = capture
        with mock.patch('barrier_workflow.time.strftime', return_value='same-second'), \
                mock.patch('barrier_workflow.time.time_ns', side_effect=[1000, 1001]):
            self.flow._learn_arms({'barriers': rows}, 2000.)
        self.assertEqual(['exit', 'entrance'], captures)
        self.assertEqual(2, len(list(self.store.parent.glob('barriers.npz.before-learn-*'))))
        self.assertTrue(self.flow.undo_last_learning()['ok'])
        self.assertEqual(b'version-2', self.store.read_bytes())
        self.assertTrue(self.flow.undo_last_learning()['ok'])
        self.assertEqual(b'version-1', self.store.read_bytes())


class GlareMeasureTests(unittest.TestCase):
    def test_headlight_glare_on_the_arm_strips_is_measured(self):
        clock = Clock()
        monitor = BarrierMonitor(clock=clock)
        dark = np.full((360, 640), 40, np.uint8)
        monitor._frames.append((clock(), dark))
        calm = monitor.learning_conditions('exit')
        self.assertEqual((0., 0.), (calm['glare'], calm['light_range']))
        glared = dark.copy()
        glared[:, :] = 250
        monitor._frames.append((clock(), glared))
        lit = monitor.learning_conditions('exit')
        self.assertGreater(lit['glare'], .5)
        self.assertGreater(lit['light_range'], 100)


class WatcherBarrierLearningTests(unittest.TestCase):
    def test_learned_example_is_logged_with_its_picture_and_undo_is_reported(self):
        obj = watcher()
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(watch, 'STATE_DIR', Path(folder)):
            obj.preview_jpeg = b'\xff\xd8 picture'
            obj._barrier_learned({'arm': 'entrance', 'position': 'up', 'match': .86})
            kind, detail = obj.log_event.call_args.args
            self.assertEqual('barrier-learned', kind)
            picture = obj.log_event.call_args.kwargs['picture']
            self.assertTrue((Path(folder) / 'barrier-learning' / picture).is_file())
        obj.barrier_workflow = mock.Mock()
        obj.barrier_workflow.undo_last_learning.return_value = {'ok': True, 'restored': 'x'}
        self.assertTrue(obj.undo_barrier_learning()['ok'])
        self.assertEqual('barrier-learning-undo', obj.log_event.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
