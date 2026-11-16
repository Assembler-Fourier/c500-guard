"""The AI also watches the whole picture (no double counting), and remembers where each object came from.

15 Nov 2026: someone who parks on the right and walks across the lanes to the cabin window was invisible to the AI
until the footpath, because only the two focused crops were inspected.
"""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_watch import watch
from object_tracking import ObjectVisitTracker
from regional_inference import add_context_detections
from semantic_monitor import SemanticMonitor
import settings_store


def det(label, box, confidence=.8):
    return {'label': label, 'confidence': confidence, 'bbox': list(box)}


class ContextMergeTests(unittest.TestCase):
    crops = [(0, 16, 380, 300), (281, 0, 617, 200)]

    def test_objects_where_the_crops_do_not_look_are_added(self):
        person = det('person', (500, 250, 530, 340))                    # main road near the camera
        result = add_context_detections([], [person], self.crops)
        self.assertEqual(1, len(result))
        self.assertEqual('whole picture', result[0]['view'])

    def test_objects_inside_a_crop_are_left_to_the_sharper_view(self):
        self.assertEqual([], add_context_detections([], [det('car', (100, 100, 200, 160))], self.crops))

    def test_no_double_counting_across_a_crop_edge(self):
        focused = [det('car', (330, 250, 380, 300))]                      # clipped at the crop edge
        whole = det('car', (335, 255, 420, 320))                          # centre outside the crop
        self.assertEqual(focused, add_context_detections(focused, [whole], self.crops))

    def test_a_person_beside_a_car_is_still_its_own_object(self):
        focused = [det('car', (420, 230, 520, 330))]
        person = det('person', (470, 240, 500, 335))
        result = add_context_detections([], [focused[0], person], [(0, 0, 10, 10)])
        self.assertEqual(['car', 'person'], [r['label'] for r in result])

    def test_bad_input_is_ignored(self):
        bad = [{'label': 'car'}, det('dog', (500, 250, 530, 340)), det('car', (5, 5, 1, 1)),
               {'label': 'car', 'bbox': ['x', 1, 2, 3]}, det('car', (500, 250, 530, 340), 3)]
        self.assertEqual([], add_context_detections([], bad, self.crops))


class MonitorContextViewTests(unittest.TestCase):
    def zone(self, rect):
        return SimpleNamespace(rect=rect, motion=False, ai_enabled=True, classes=['person', 'car'],
                               severity='main', name='Walkway', route_role='pedestrian')

    def make(self, context):
        with patch.object(SemanticMonitor, '_run', lambda self: None):
            monitor = SemanticMonitor([self.zone((.2, .2, .5, .5))], Path('mock.onnx'), interval=0,
                                      context_view=context)
        self.addCleanup(monitor.stop)
        return monitor

    def test_whole_picture_is_queued_as_an_extra_context_view(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        monitor = self.make(True)
        monitor.update(frame, True)
        _, _, _, views = monitor._input.get_nowait()
        self.assertEqual(2, len(views))
        self.assertEqual(((0, 0, 640, 360), 'context'), (views[1][1], views[1][2]))
        self.assertEqual((360, 640, 3), views[1][0].shape)

    def test_off_by_default(self):
        monitor = self.make(False)
        monitor.update(np.zeros((360, 640, 3), np.uint8), True)
        _, _, _, views = monitor._input.get_nowait()
        self.assertEqual(1, len(views))

    def test_worker_merges_context_objects_outside_the_crops(self):
        monitor = SemanticMonitor([self.zone((.0, .0, .5, .5))], Path('mock.onnx'), interval=60, context_view=True)
        self.addCleanup(monitor.stop)

        class Detector:
            available = True

            def status(self):
                return {'available': True, 'error': ''}

            def detect(self, frame):
                if frame.shape[:2] == (360, 640):                         # the whole picture
                    return [det('person', (500, 250, 530, 340)), det('car', (20, 20, 60, 60))]
                return [det('car', (20, 20, 60, 60))]

        with patch('object_detector.ObjectDetector', return_value=Detector()):
            monitor._stop.set()
            monitor._thread.join(1)
            monitor._stop.clear()
            monitor._thread = threading.Thread(target=monitor._run, daemon=True)
            monitor._thread.start()
            deadline = time.monotonic() + 2
            while not monitor.status().get('available') and time.monotonic() < deadline:
                time.sleep(.01)
            frame = np.zeros((360, 640, 3), np.uint8)
            crop = frame[0:200, 0:340].copy()
            monitor._input.put_nowait((monitor._epoch, monitor._worker_generation, time.monotonic(),
                                       [(crop, (0, 0, 340, 200)), (frame.copy(), (0, 0, 640, 360), 'context')]))
            deadline = time.monotonic() + 2
            while monitor._output.empty() and time.monotonic() < deadline:
                time.sleep(.01)
        _, _, _, objects = monitor._output.get_nowait()
        self.assertEqual([('car', None), ('person', 'whole picture')],
                         sorted((o['label'], o.get('view')) for o in objects))
        self.assertIn('whole picture', monitor._status['inference_mode'])


class OriginTests(unittest.TestCase):
    def test_alarm_evidence_says_where_the_object_came_from(self):
        road = SimpleNamespace(name='Main road', contains_point=lambda x, y: y > .6, classes=['person'],
                               ai_enabled=False, rect=(0, .6, 1, .4))
        path = SimpleNamespace(name='Pedestrian footpath', contains_point=lambda x, y: y < .5,
                               classes=['person'], ai_enabled=True, rect=(0, 0, 1, .5), ai_confidence=0.,
                               route_group='', severity='main')
        tracker = ObjectVisitTracker()
        events = []
        for step in range(8):
            y = 300 - step * 25                                           # walking from the road to the path
            events, observed = tracker.update([det('person', (300, y - 60, 320, y))], [road, path], 640, 360,
                                              100. + step)
        self.assertTrue(events)
        context = tracker.event_context(events[0][2], [road, path])
        self.assertEqual('Main road', context['came_from'])
        self.assertEqual(7.0, context['tracked_s'])
        self.assertEqual(7.0, observed[0]['tracked_s'])


class SettingTests(unittest.TestCase):
    def test_setting_exists_on_by_default_and_can_be_turned_off(self):
        item = next(i for section in settings_store.STATIC for i in section['items']
                    if i.get('key') == 'ai.whole_picture')
        self.assertEqual(('bool', True), (item['kind'], item['default']))
        config, changed = settings_store.apply_changes({'zones': [], 'ai': {}}, {'ai.whole_picture': False})
        self.assertIs(False, config['ai']['whole_picture'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zones.json'
            path.write_text(json.dumps({'zones': [], 'ai': {}}), encoding='utf-8')
            self.assertTrue(watch.Watcher._load_ai_whole_picture(path))
            path.write_text(json.dumps({'zones': [], 'ai': {'whole_picture': False}}), encoding='utf-8')
            self.assertFalse(watch.Watcher._load_ai_whole_picture(path))
            path.write_text(json.dumps({'zones': [], 'ai': {'whole_picture': 'yes'}}), encoding='utf-8')
            self.assertTrue(watch.Watcher._load_ai_whole_picture(path))


if __name__ == '__main__':
    unittest.main()
