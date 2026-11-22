"""Clean pictures of followed vehicles for the separate car memory."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from test_watch import watch, watcher
from vehicle_sightings import VehicleSightings
from semantic_monitor import SemanticMonitor

ZONES = [SimpleNamespace(name='Exit approach', route_role='exit'),
         SimpleNamespace(name='Top road arrival', route_role='entrance'),
         SimpleNamespace(name='Pedestrian footpath', route_role='pedestrian')]


def car(track, box, regions=('Exit approach',), label='car', confidence=.8):
    return {'track_id': track, 'label': label, 'confidence': confidence, 'box_n': list(box), 'regions': list(regions)}


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


class SightingsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.clock = Clock()
        self.s = VehicleSightings(self.root, clock=self.clock, wall=lambda: 1_700_000_000. + self.clock.now)
        self.frame = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)

    def tearDown(self):
        self.folder.cleanup()

    def feed(self, objects, step=.5):
        self.clock.now += step
        self.s.observe(objects, self.clock.now, self.frame, ZONES)

    def written(self):
        return sorted(p for p in self.root.iterdir() if p.is_dir())

    def test_an_exiting_car_is_written_with_its_best_clean_crops(self):
        for i in range(8):                                  # moving up the exit lane, growing then shrinking
            y = .8 - i * .06
            width = .1 + (.05 if i == 3 else 0)
            self.feed([car(7, (.2, y - .1, .2 + width, y))])
        self.assertEqual([], self.written())                # still being followed
        self.feed([])
        self.clock.now += 5
        self.feed([])
        folders = self.written()
        self.assertEqual(1, len(folders))
        meta = json.loads((folders[0] / 'meta.json').read_text(encoding='utf-8'))
        self.assertEqual(('out', 'car', 7, 4), (meta['direction'], meta['label'], meta['track_id'], meta['crops']))
        self.assertEqual(['Exit approach'], meta['zones'])
        self.assertEqual(4, len(list(folders[0].glob('crop_*.jpg'))))
        from PIL import Image
        with Image.open(folders[0] / 'crop_0.jpg') as first:
            self.assertEqual(int(.15 * 640), first.width)                         # the largest view first

    def test_entering_car_reads_in(self):
        for i in range(5):
            self.feed([car(3, (.6, .1 + i * .05, .75, .2 + i * .05), regions=('Top road arrival',))])
        self.clock.now += 6
        self.feed([])
        meta = json.loads((self.written()[0] / 'meta.json').read_text(encoding='utf-8'))
        self.assertEqual('in', meta['direction'])

    def test_people_parked_cars_and_unsure_detections_are_not_kept(self):
        for _ in range(5):
            self.feed([car(1, (.1, .1, .3, .3), label='person'),
                       car(2, (.1, .1, .3, .3), regions=()),                 # never in an entrance/exit zone
                       car(3, (.1, .1, .3, .3), confidence=.3)])
        self.clock.now += 6
        self.feed([])
        self.assertEqual([], self.written())

    def test_results_not_yet_numbered_by_the_tracker_wait(self):
        self.feed([{'label': 'car', 'confidence': .9, 'box_n': [.1, .1, .3, .3], 'regions': ['Exit approach']}])
        self.assertEqual({}, self.s._tracks)

    def test_same_result_is_counted_once(self):
        self.s.observe([car(5, (.1, .1, .3, .3))], 42., self.frame, ZONES)
        self.s.observe([car(5, (.1, .1, .3, .3))], 42., self.frame, ZONES)
        self.assertEqual(1, len(self.s._tracks[5]['grounds']))
        self.assertEqual(1, len(self.s._tracks[5]['observations']))

    def test_missed_alarm_decisions_are_saved_with_boxes_and_confidence(self):
        for index in range(65):
            obj = car(31, (.70, .40, .80, .56), regions=('Top road arrival',))
            obj['region_status'] = {'Top road arrival': 'moving away'}
            obj['confidence'] = .7 + index / 1000
            self.feed([obj])
        rows = self.finish()[0]['observations']
        self.assertEqual(60, len(rows))
        self.assertEqual(.705, rows[0]['confidence'])
        self.assertEqual(.764, rows[-1]['confidence'])
        self.assertEqual([.70, .40, .80, .56], rows[-1]['bbox'])
        self.assertEqual({'Top road arrival': 'moving away'}, rows[-1]['region_status'])
        self.assertLess(rows[0]['at'], rows[-1]['at'])

    def test_flush_on_off_and_twelve_hour_retention(self):
        self.feed([car(9, (.1, .1, .3, .3))])
        self.s.flush()
        folder = self.written()[0]
        old = time.time() - 13 * 3600
        os.utime(folder, (old, old))
        self.s.wall = time.time
        self.assertEqual(1, self.s.purge())
        self.assertEqual([], self.written())

    def test_errors_never_escape(self):
        self.s.observe([car(1, (.1, .1, .3, .3))], 1., 'not a frame', ZONES)
        self.assertTrue(self.s.last_error)

    def test_off_means_nothing_is_collected(self):
        self.s.enabled = False
        self.feed([car(1, (.1, .1, .3, .3))])
        self.assertEqual({}, self.s._tracks)

    def test_vehicles_cut_off_at_the_edge_or_on_the_camera_icon_give_no_picture(self):
        for box in ((0., .3, .2, .5), (.8, .3, 1., .5), (.3, .8, .5, 1.), (.05, .6, .2, .75)):
            self.feed([car(11, box)])
        self.assertEqual([], self.s._tracks[11]['crops'])

    def test_glare_loses_to_a_clear_view(self):
        glare = self.frame.copy()
        glare[:, :] = 255                                      # headlights straight into the camera
        self.s.max_crops = 1
        self.clock.now += .5
        self.s.observe([car(12, (.3, .3, .6, .6))], self.clock.now, glare, ZONES)   # bigger but blown out
        self.feed([car(12, (.3, .3, .5, .5))])                                       # smaller, clear
        self.assertEqual(1, len(self.s._tracks[12]['crops']))
        self.assertLess(float(self.s._tracks[12]['crops'][0][1].mean()), 250)

    def finish(self):
        self.clock.now += 6
        self.feed([])
        return [json.loads((f / 'meta.json').read_text(encoding='utf-8')) for f in self.written()]

    def test_leaving_through_the_exit_and_up_the_top_road_is_out(self):
        for i in range(4):                                       # exit lanes, moving up
            self.feed([car(20, (.2, .5 - i * .05, .35, .6 - i * .05))])
        for i in range(4):                                       # then the top road (entrance zones), still up
            self.feed([car(20, (.55, .3 - i * .04, .7, .4 - i * .04), regions=('Top road arrival',))])
        self.assertEqual('out', self.finish()[0]['direction'])

    def test_seen_only_moving_away_up_the_top_road_is_out(self):
        for i in range(5):
            self.feed([car(21, (.55, .35 - i * .03, .7, .45 - i * .03), regions=('Top road arrival',))])
        self.assertEqual('out', self.finish()[0]['direction'])

    def test_standing_in_the_exit_lane_has_not_left(self):
        for _ in range(5):
            self.feed([car(22, (.2, .4, .35, .5))])
        self.assertEqual('unclear', self.finish()[0]['direction'])

    def test_a_one_picture_fragment_keeps_only_diagnostics_not_car_memory(self):
        self.feed([car(23, (.2, .4, .35, .5))])
        meta = self.finish()[0]
        self.assertTrue(meta['diagnostic_only'])
        self.assertEqual(0, meta['crops'])
        self.assertEqual(1, len(meta['observations']))
        self.assertEqual([], list(self.written()[0].glob('*.jpg')))
        self.assertEqual(0, self.s.written)
        self.assertEqual(1, self.s.diagnostics_written)

    def test_early_vehicle_without_usable_crop_keeps_confidence_and_rejection(self):
        self.frame[:] = 255  # glare: no crop can establish appearance
        for confidence in (.70, .548, .57):
            obj = car(24, (.60, .25, .72, .40), regions=('Top road arrival',), confidence=confidence)
            obj['region_status'] = {'Top road arrival': 'below confidence threshold' if confidence < .55 else 'confirming'}
            self.feed([obj])
        meta = self.finish()[0]
        self.assertTrue(meta['diagnostic_only'])
        self.assertEqual([.70,.548,.57], [o['confidence'] for o in meta['observations']])
        self.assertEqual('below confidence threshold', meta['observations'][1]['region_status']['Top road arrival'])
        self.assertEqual([], list(self.written()[0].glob('*.jpg')))

    def test_long_wait_does_not_erase_the_original_travel_direction(self):
        for i in range(5):
            self.feed([car(25, (.65,.2+i*.05,.78,.3+i*.05), regions=('Top road arrival',))])
        for _ in range(70):
            self.feed([car(25, (.65,.4,.78,.5), regions=('Top road arrival',))])
        meta = self.finish()[0]
        self.assertEqual('in',meta['direction'])
        self.assertEqual(60,len(meta['observations']))
        self.assertAlmostEqual(.3,meta['first_ground'][1])

    def test_reused_track_id_and_second_do_not_overwrite_an_existing_sighting(self):
        for _ in range(3):
            self.feed([car(26, (.2,.4,.35,.5))])
        record = self.s._tracks[26]
        self.s._write(26,record)
        self.s._write(26,record)
        self.assertEqual(2,len(self.written()))
        self.assertTrue(all((folder/'meta.json').exists() for folder in self.written()))


class MonitorFrameTests(unittest.TestCase):
    def test_analysed_frames_are_kept_only_when_asked(self):
        zone = SimpleNamespace(rect=(.2, .2, .5, .5), motion=False, ai_enabled=True, classes=['car'],
                               severity='main', name='Lane', route_role='exit')
        with mock.patch.object(SemanticMonitor, '_run', lambda self: None):
            monitor = SemanticMonitor([zone], Path('mock.onnx'), interval=0, context_view=True)
        self.addCleanup(monitor.stop)
        frame = np.zeros((360, 640, 3), np.uint8)
        monitor.update(frame, True, stamp=time.monotonic())
        self.assertEqual(0, len(monitor._frames))
        monitor.keep_frames = True
        monitor._input.get_nowait()
        stamp = time.monotonic()
        monitor.update(frame, True, stamp=stamp)
        self.assertEqual((360, 640, 3), monitor.frame_at(stamp).shape)
        self.assertIsNone(monitor.frame_at(stamp - 1))


class WatcherFeedTests(unittest.TestCase):
    def test_watcher_feeds_the_frame_the_ai_saw(self):
        obj = watcher()
        obj._alignment_blocked = False
        obj.sightings = mock.Mock(enabled=True)
        obj.semantic = mock.Mock()
        frame = np.zeros((360, 640, 3), np.uint8)
        obj.semantic.frame_at.return_value = frame
        primary = {'observed_at': 12.5, 'detections': [car(1, (.1, .1, .3, .3))]}
        obj._feed_sightings(primary)
        obj.semantic.frame_at.assert_called_once_with(12.5)
        args = obj.sightings.observe.call_args.args
        self.assertIs(frame, args[2])
        obj._alignment_blocked = True
        obj._feed_sightings(primary)
        self.assertEqual(1, obj.sightings.observe.call_count)

    def test_setting_defaults_on(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zones.json'
            path.write_text(json.dumps({'zones': []}), encoding='utf-8')
            self.assertTrue(watch.Watcher._load_sightings_enabled(path))
            path.write_text(json.dumps({'zones': [], 'memory': {'vehicle_sightings': False}}), encoding='utf-8')
            self.assertFalse(watch.Watcher._load_sightings_enabled(path))


if __name__ == '__main__':
    unittest.main()
