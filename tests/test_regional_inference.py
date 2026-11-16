import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from regional_inference import inference_rects, merge_view_detections
from semantic_monitor import SemanticMonitor


def zone(rect, role='', **extra):
    return SimpleNamespace(rect=rect, route_role=role, ai_enabled=True,
                           classes=['car'], severity='main', name=role or 'Gate', **extra)


def detection(box, label='car', confidence=.8):
    return {'bbox': list(box), 'label': label, 'confidence': confidence}


class ViewGeometryTests(unittest.TestCase):
    def test_no_role_configuration_preserves_exact_legacy_crop(self):
        self.assertEqual([(14,10,76,74)], inference_rects([zone((.2,.2,.5,.5))],100,100))

    def test_unknown_or_partial_roles_fall_back_without_losing_regions(self):
        zones = [zone((.1,.2,.2,.4),'exit'), zone((.6,.1,.2,.3),'entrance'),
                 zone((.3,.8,.1,.1))]
        self.assertEqual(1, len(inference_rects(zones,640,360)))
        zones[-1].route_role = 'unrecognized'
        self.assertEqual(1, len(inference_rects(zones,640,360)))

    def test_focused_views_include_entire_footpath_and_body_headroom(self):
        zones = [zone((.10,.42,.31,.42),'exit'),
                 zone((.13,.20,.30,.27),'pedestrian'),
                 zone((.55,.14,.21,.12),'entrance')]
        exit_rect, entrance_rect = inference_rects(zones,640,360)
        self.assertEqual((12,28,326,323), exit_rect)
        self.assertEqual((300,7,537,115), entrance_rect)
        self.assertGreater(exit_rect[2], entrance_rect[0])
        self.assertLess(exit_rect[1], .20 * 360 - 35)
        for z, rect in ((zones[0],exit_rect),(zones[1],exit_rect),(zones[2],entrance_rect)):
            self.assertLessEqual(rect[0], z.rect[0]*640)
            self.assertLessEqual(rect[1], z.rect[1]*360)
            self.assertGreaterEqual(rect[2], (z.rect[0]+z.rect[2])*640)
            self.assertGreaterEqual(rect[3], (z.rect[1]+z.rect[3])*360)

    def test_no_ai_regions_do_not_create_passes(self):
        z = zone((.1,.1,.4,.4),'exit')
        z.ai_enabled = False
        self.assertEqual([], inference_rects([z],640,360))
        z.ai_enabled, z.classes = True, []
        self.assertEqual([], inference_rects([z],640,360))

    def test_fullframe_bounds_clip_and_identical_views_avoid_duplicate_work(self):
        self.assertEqual([(0,0,640,360)], inference_rects(
            [zone((0,0,1,1),'exit'),zone((0,0,1,1),'entrance')],640,360))


class MergeTests(unittest.TestCase):
    def test_offsets_map_into_aligned_full_frame_without_mutating_source(self):
        obj = detection([10,20,30,50])
        result = merge_view_detections([([obj],(100,50,200,150))])
        self.assertEqual([110,70,130,100],result[0]['bbox'])
        self.assertEqual([10,20,30,50],obj['bbox'])

    def test_same_car_from_overlapping_views_counts_once_at_higher_confidence(self):
        views = [([detection([40,20,60,60],confidence=.7)],(0,0,70,100)),
                 ([detection([10,20,30,60],confidence=.9)],(30,0,100,100))]
        result = merge_view_detections(views)
        self.assertEqual(1,len(result))
        self.assertEqual(.9,result[0]['confidence'])
        self.assertEqual([40,20,60,60],result[0]['bbox'])

    def test_vehicle_class_disagreement_merges_but_person_and_bicycle_remain(self):
        objects = [detection([5,5,35,45],label,.9-index*.05)
                   for index,label in enumerate(['car','bus','person','bicycle'])]
        result = merge_view_detections([(objects,(0,0,100,100))])
        self.assertEqual(['car','person','bicycle'],[o['label'] for o in result])

    def test_two_separate_vehicles_remain(self):
        objects = [detection([5,5,30,45]),detection([35,5,60,45])]
        self.assertEqual(2,len(merge_view_detections([(objects,(0,0,100,100))])))

    def test_recorded_entrance_crop_slice_does_not_make_second_exit_car(self):
        # 20:32:03: track35 begins at x371.01, the entrance crop's left edge;
        # its actual recorded 21x39 slice overlaps the simultaneously visible
        # complete track33 box (full box reconstructed from the visible overlay).
        full = detection([331,63,390,102], confidence=.76)
        fragment = detection([.01,62.42,21.04,101.07], confidence=.6286)
        merged = merge_view_detections([([full],(0,77,556,352)),
                                        ([fragment],(371,77,630,206))])
        self.assertEqual([{**full,'bbox':[331,140,390,179]}],merged)

    def test_high_confidence_fragment_still_keeps_complete_vehicle(self):
        whole = detection([30,20,90,80], confidence=.65)
        part = detection([0,20,20,80], confidence=.95)
        self.assertEqual([whole],merge_view_detections([([whole],(0,0,100,100)),
                                                       ([part],(70,0,150,100))]))

    def test_small_complete_or_same_crop_object_is_not_erased(self):
        whole = detection([30,20,90,80])
        part = detection([70,25,85,75],confidence=.9)
        self.assertEqual(2,len(merge_view_detections([([whole,part],(0,0,100,100))])))
        self.assertEqual(2,len(merge_view_detections([([whole],(0,0,100,100)),
                                                   ([detection([10,25,25,75])],(60,0,150,100))])))

    def test_adjacent_vehicle_touching_crop_edge_but_not_contained_stays(self):
        whole = detection([30,20,90,80])
        adjacent = detection([0,20,35,80])
        self.assertEqual(2,len(merge_view_detections([([whole],(0,0,100,100)),
                                                   ([adjacent],(80,0,150,100))])))

    def test_two_clipped_boxes_have_no_complete_witness_and_are_retained(self):
        first = detection([30,20,90,80])
        second = detection([0,20,20,80])
        self.assertEqual(2,len(merge_view_detections([([first],(0,0,90,100)),
                                                   ([second],(70,0,150,100))])))

    def test_edge_fragment_of_another_class_family_remains_separate(self):
        whole = detection([30,20,90,80])
        person = detection([0,20,20,80],label='person')
        self.assertEqual(2,len(merge_view_detections([([whole],(0,0,100,100)),
                                                   ([person],(70,0,150,100))])))

    def test_malformed_results_do_not_poison_valid_boxes(self):
        objects = [detection([float('nan'),0,20,20]), detection([20,0,10,10]),
                   detection([0,0,20,20],confidence=float('inf')),
                   detection([-5,-8,120,140])]
        result = merge_view_detections([(objects,(10,20,110,120))])
        self.assertEqual(1,len(result))
        self.assertEqual([10,20,110,120],result[0]['bbox'])


class WorkerViewTests(unittest.TestCase):
    def make_monitor(self):
        with patch.object(threading.Thread,'start'):
            monitor = SemanticMonitor([zone((.2,.1,.5,.7))],Path('mock.onnx'),interval=60)
        monitor._enabled = True
        self.addCleanup(monitor.stop)
        return monitor

    @staticmethod
    def wait_for(predicate):
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        raise AssertionError('Worker did not reach expected state')

    @staticmethod
    def submit(monitor):
        frame = np.zeros((100,70,3),dtype=np.uint8)
        monitor._input.put_nowait((monitor._epoch,monitor._worker_generation,time.monotonic(),
                                   [(frame,(0,0,70,100)),(frame,(30,0,100,100))]))

    def test_two_passes_produce_one_temporal_observation(self):
        monitor = self.make_monitor()
        class Detector:
            available, calls = True, 0
            def status(self):
                return {'available':self.available,'error':''}
            def detect(self,frame):
                self.calls += 1
                return [detection([40,20,60,60] if self.calls % 2 else [10,20,30,60])]
        detector = Detector()
        with patch('object_detector.ObjectDetector',return_value=detector):
            monitor._thread = threading.Thread(target=monitor._run,daemon=True)
            monitor._thread.start()
            self.wait_for(lambda:monitor.status()['available'])
            self.submit(monitor)
            self.wait_for(lambda:not monitor._output.empty())
            frame = np.zeros((100,100,3),dtype=np.uint8)
            monitor._last_submit = time.monotonic()
            self.assertEqual([],monitor.update(frame,True))
            self.assertEqual(1,monitor.status()['tracked_objects'])
            self.assertEqual(0,monitor.status()['active_visits'])
            self.assertEqual(2,monitor.status()['inference_views'])
            self.submit(monitor)
            self.wait_for(lambda:not monitor._output.empty())
            self.assertEqual(1,len(monitor.update(frame,True)))
            self.assertEqual(4,detector.calls)

    def test_failure_in_second_pass_discards_partial_first_pass(self):
        monitor = self.make_monitor()
        class Detector:
            available, calls = True, 0
            def status(self):
                return {'available':self.available,'error':'second pass failed'}
            def detect(self,frame):
                self.calls += 1
                if self.calls == 2:
                    self.available = False
                return [detection([20,20,40,60])]
        with patch('object_detector.ObjectDetector',return_value=Detector()):
            monitor._thread = threading.Thread(target=monitor._run,daemon=True)
            monitor._thread.start()
            self.wait_for(lambda:monitor.status()['available'])
            self.submit(monitor)
            self.wait_for(lambda:monitor.status()['state']=='retrying')
            self.assertTrue(monitor._output.empty())
            self.assertEqual(0,monitor.status()['inferences'])
            self.assertEqual(1,monitor._worker_generation)

    def test_epoch_change_between_views_abandons_work_without_publishing(self):
        monitor = self.make_monitor()
        class Detector:
            available, calls = True, 0
            def status(self):
                return {'available':True,'error':''}
            def detect(self,frame):
                self.calls += 1
                monitor.reset()
                return [detection([20,20,40,60])]
        detector = Detector()
        with patch('object_detector.ObjectDetector',return_value=detector):
            monitor._thread = threading.Thread(target=monitor._run,daemon=True)
            monitor._thread.start()
            self.wait_for(lambda:monitor.status()['available'])
            self.submit(monitor)
            self.wait_for(lambda:detector.calls==1)
            time.sleep(.02)
            self.assertEqual(1,detector.calls)
            self.assertTrue(monitor._output.empty())
            self.assertEqual(0,monitor.status()['inferences'])

    def test_result_expiring_during_last_pass_is_not_published(self):
        monitor = self.make_monitor()
        monitor.MAX_RESULT_AGE = .025
        class Detector:
            available, calls = True, 0
            def status(self):
                return {'available':True,'error':''}
            def detect(self,frame):
                self.calls += 1
                if self.calls == 2:
                    time.sleep(.04)
                return [detection([20,20,40,60])]
        detector = Detector()
        with patch('object_detector.ObjectDetector',return_value=detector):
            monitor._thread = threading.Thread(target=monitor._run,daemon=True)
            monitor._thread.start()
            self.wait_for(lambda:monitor.status()['available'])
            self.submit(monitor)
            self.wait_for(lambda:detector.calls==2)
            time.sleep(.06)
            self.assertTrue(monitor._output.empty())
            self.assertEqual(0,monitor.status()['inferences'])
            self.assertEqual(0,monitor.status()['recovery_attempts'])


if __name__ == '__main__':
    unittest.main()
