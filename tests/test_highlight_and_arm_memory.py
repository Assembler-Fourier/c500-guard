"""Highlights (what the AI follows, what changed) and the barrier arms' remembered positions.

Display, evidence and history only: none of this starts, stops or quietens an alarm.
"""
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from test_watch import watch, watcher
from approach_light import measure_local_brightness
from object_tracking import ObjectVisitTracker
import dashboard_ui

NODE = shutil.which("node") or str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")


class TrackerHighlightTests(unittest.TestCase):
    def test_observed_objects_carry_a_normalised_box_and_a_short_path(self):
        tracker = ObjectVisitTracker()
        observed = []
        for step in range(14):
            x = 100 + step * 10
            _, observed = tracker.update([{'label': 'car', 'confidence': .9, 'bbox': [x, 100, x + 60, 160]}],
                                         [], 640, 360, 1000. + step * .5)
        car = observed[0]
        self.assertEqual([round(v, 4) for v in car['box_n']],
                         [round(230 / 640, 4), round(100 / 360, 4), round(290 / 640, 4), round(160 / 360, 4)])
        self.assertEqual(10, len(car['trail']))                       # bounded
        self.assertAlmostEqual(car['trail'][-1][0], 260 / 640, places=3)  # newest ground point last
        self.assertLess(car['trail'][0][0], car['trail'][-1][0])


class OverlayObjectTests(unittest.TestCase):
    def test_state_follows_what_the_object_is_doing(self):
        base = {'label': 'car', 'confidence': .87, 'track_id': 7, 'box_n': [.1, .2, .3, .4], 'trail': [[.2, .4]]}
        rows = watch.Watcher._overlay_objects([
            {**base, 'confirmed_regions': ['Exit barrier'], 'regions': ['Exit barrier'],
             'region_status': {'Exit barrier': 'confirmed'}},
            {**base, 'track_id': 8, 'regions': ['Exit approach'], 'region_status': {'Exit approach': 'confirming'}},
            {**base, 'track_id': 9, 'label': 'person', 'regions': []},
            {'label': 'car', 'confidence': .5},                               # no box: skipped
            {**base, 'box_n': ['x', 1, 2, 3]},                                # bad box: skipped
        ])
        self.assertEqual([('alert', 'Exit barrier', 7), ('zone', 'Exit approach', 8), ('seen', '', 9)],
                         [(row['state'], row['zone'], row['id']) for row in rows])
        self.assertEqual('confirmed', rows[0]['status'])
        self.assertEqual([[.2, .4]], rows[0]['trail'])

    def test_at_most_twelve_objects(self):
        rows = watch.Watcher._overlay_objects([{'label': 'car', 'confidence': .9, 'box_n': [.1, .1, .2, .2]}] * 30)
        self.assertEqual(12, len(rows))

    def test_unscoped_road_guess_is_not_drawn_as_lane_traffic(self):
        guess = {'label': 'car', 'confidence': .55, 'track_id': 275,
                 'box_n': [.67, .47, .87, .63], 'regions': [], 'confirmed_regions': []}
        person = {**guess, 'label': 'person', 'track_id': 276}
        lane_car = {**guess, 'regions': ['Entrance waiting'], 'track_id': 277}
        observed = [guess] * 13 + [person, lane_car]
        rows = watch.Watcher._overlay_objects(observed)
        self.assertEqual([276, 277], [row['id'] for row in rows])
        self.assertEqual([], guess['regions'], 'Display filtering must not mutate AI evidence')

    def test_outside_track_keeps_direction_history_when_it_enters_lane(self):
        tracker = ObjectVisitTracker()
        zone = watch.Zone(name='lane', rect=(.3, .1, .5, .8), trigger='ai',
                          arrival_direction=[1, 0], arrival_min_progress=.01)
        _, seen = tracker.update([{'label': 'car', 'confidence': .8, 'bbox': [100, 100, 160, 180]}],
                                 [zone], 640, 360, 100.)
        identity = seen[0]['track_id']
        self.assertEqual([], watch.Watcher._overlay_objects(seen))
        _, seen = tracker.update([{'label': 'car', 'confidence': .8, 'bbox': [170, 100, 230, 180]}],
                                 [zone], 640, 360, 100.5)
        rows = watch.Watcher._overlay_objects(seen)
        self.assertEqual(identity, rows[0]['id'])
        self.assertEqual('lane', rows[0]['zone'])
        self.assertEqual(2, len(rows[0]['trail']))


class ChangeHighlightTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()
        self.zone = self.obj.zones[0]

    def test_changed_patch_is_outlined_briefly_then_dropped(self):
        self.obj._alignment_blocked = False
        self.obj._note_highlight(self.zone, [64, 36, 32, 18], (360, 640), 'change')
        boxes = self.obj._change_boxes()
        self.assertEqual([{'zone': self.zone.name, 'box': [.1, .1, .15, .15], 'kind': 'change',
                           'alarm': bool(getattr(self.zone, 'active', False))}], boxes)
        self.zone._highlight = (time.monotonic() - self.obj.HIGHLIGHT_S - .1,) + self.zone._highlight[1:]
        self.assertEqual([], self.obj._change_boxes())

    def test_nothing_is_outlined_while_alignment_is_lost_or_bounds_are_bad(self):
        self.obj._note_highlight(self.zone, [64, 36, 0, 18], (360, 640), 'change')     # zero width
        self.assertIsNone(getattr(self.zone, '_highlight', None))
        self.obj._note_highlight(self.zone, [64, 36, 32, 18], (360, 640), 'light')
        self.obj._alignment_blocked = True
        self.assertEqual([], self.obj._change_boxes())

    def test_state_lists_objects_and_changes(self):
        state = self.obj.state_dict()
        self.assertIn('objects', state)
        self.assertIn('change_boxes', state)

    def test_alarm_pictures_get_the_highlight(self):
        frame = np.zeros((watch.PROC_H, watch.PROC_W, 3), np.uint8)
        self.obj._alignment_blocked = False
        self.obj.semantic = mock.Mock()
        self.obj.semantic.status.return_value = {'age_s': .2, 'detections': [
            {'label': 'car', 'confidence': .9, 'track_id': 3, 'box_n': [.5, .5, .7, .8], 'trail': [],
             'confirmed_regions': ['Exit barrier'], 'regions': ['Exit barrier']}]}
        self.obj.render_preview(frame)
        image = np.asarray(Image.open(BytesIO(self.obj.preview_jpeg)).convert('RGB')).astype(int)
        edge = image[int(.65 * watch.PROC_H), int(.5 * watch.PROC_W)]       # left edge of the box
        self.assertGreater(edge[0], 180)                                      # red: confirmed arrival
        self.assertLess(edge[2], 140)


class LightBoundsTests(unittest.TestCase):
    def test_light_measurement_reports_where_the_light_appeared(self):
        background = np.full((120, 160), 40, np.uint8)
        current = background.copy()
        current[40:80, 50:110] = 140                     # headlights: 12.5 % of the region, +100 levels
        region = np.ones((120, 160), np.uint8)
        report = measure_local_brightness(current, background, region, min_delta=5)
        self.assertTrue(report['hit'])
        x, y, w, h = report['support_bounds']
        self.assertTrue(48 <= x <= 52 and 38 <= y <= 42 and 56 <= w <= 62 and 36 <= h <= 42, report['support_bounds'])
        self.assertIsNone(measure_local_brightness(background, background, region)['support_bounds'])


class ArmMemoryTests(unittest.TestCase):
    def setUp(self):
        self.obj = watcher()

    def status(self, exit_state, fresh=True, calibrated=True):
        return {'barriers': [{'id': 'exit', 'state': exit_state, 'fresh': fresh, 'calibrated': calibrated},
                             {'id': 'entrance', 'state': 'down', 'fresh': True, 'calibrated': True}]}

    def arm_events(self):
        return [call for call in self.obj.log_event.call_args_list if call.args[0] == 'arm']

    def test_first_reading_is_remembered_without_an_event(self):
        self.obj._note_arm_memory(self.status('down'))
        memory = self.obj._arm_memory_status()['memory']
        self.assertEqual('down', memory['exit']['state'])
        self.assertEqual([], self.arm_events())

    def test_unknown_stale_or_uncalibrated_readings_never_change_the_memory(self):
        self.obj._note_arm_memory(self.status('up'))
        for args in (('unknown',), ('down', False), ('down', True, False)):
            self.obj._note_arm_memory(self.status(*args))
        self.assertEqual('up', self.obj._arm_memory_status()['memory']['exit']['state'])
        self.assertEqual([], self.arm_events())

    def test_a_real_change_is_logged_with_how_long_the_arm_was_held(self):
        self.obj._note_arm_memory(self.status('up'))
        self.obj._arm_memory['exit']['since'] -= 270                # held up for 4.5 minutes
        self.obj._arm_live = {'exit': {'shown': 5., 'at_wall': time.time()}}   # the angle tracker agrees: down
        self.obj._note_arm_memory(self.status('down'))
        events = self.arm_events()
        self.assertEqual(1, len(events))
        self.assertEqual(('exit', 'down'), (events[0].kwargs['arm'], events[0].kwargs['state']))
        self.assertGreaterEqual(events[0].kwargs['previous_for_s'], 270)
        history = self.obj._arm_memory_status()['history']
        self.assertEqual(('exit', 'down'), (history[-1]['arm'], history[-1]['state']))

    def test_barrier_status_includes_memory_and_history(self):
        self.obj.barrier_workflow = mock.Mock()
        self.obj.barrier_workflow.status.return_value = {'configured': True, 'barriers': []}
        self.obj._note_arm_memory(self.status('up'))
        status = self.obj.barrier_status()
        self.assertEqual('up', status['memory']['exit']['state'])
        self.assertIn('history', status)

    def test_garbage_never_raises(self):
        for value in (None, [], {'barriers': None}, {'barriers': [None, 'x', {'id': 'exit'}]}):
            self.obj._note_arm_memory(value)


class PageOverlayTests(unittest.TestCase):
    def script(self):
        page = dashboard_ui.dashboard_page('BarrierWatcher')
        start = page.index('var renderedObjects=')
        end = page.index('function renderOverlay(s,status){')
        return page[start:end]

    def node(self, body):
        if not Path(NODE).exists():
            self.skipTest('Node.js is required for browser-script behaviour checks')
        result = subprocess.run([NODE, '-'], input=body, text=True, capture_output=True, encoding='utf-8', timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_lane_cards_replace_remembered_pills_and_keep_object_overlay(self):
        page = dashboard_ui.dashboard_page('BarrierWatcher')
        self.assertNotIn('id="barrier-pills"', page)
        self.assertIn('pill.className=`arm-pill ${arm.state}`', page)
        self.assertIn('id="lane-${role}-arm"', page)
        self.assertIn("arm.state==='unknown'?'UNKNOWN'", page)
        self.assertIn('data-lane-sound="entrance"', page)
        self.assertIn('<svg id="object-overlay"', page)

    def test_overlay_draws_objects_and_changes_and_clears_on_alignment_loss(self):
        harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const nodes={};const $=id=>nodes[id]||(nodes[id]={innerHTML:''});
const ctx={$,mode:'live',esc:s=>String(s).replace(/[<>&"]/g,''),shortZoneName:s=>s.replace(' barrier',''),
 finite:n=>typeof n==='number'&&Number.isFinite(n),
 JSON,Math,Number,String,Array};
vm.runInNewContext(SOURCE,ctx);
const status={cameraFresh:true,aligned:true,aiFresh:true};
const s={objects:[{id:12,label:'car',confidence:.82,box:[.5,.5,.6,.6],state:'alert',zone:'Exit barrier',trail:[[.5,.6],[.55,.6]]}],
 change_boxes:[{zone:'Exit corner',box:[.1,.1,.2,.2],kind:'change',alarm:true}]};
ctx.renderObjects(s,status);
const svg=nodes['object-overlay'].innerHTML;
assert.match(svg,/<g class="obj alert vehicle">/);
assert.match(svg,/<polyline points="320.0,216.0 352.0,216.0"\/>/);
assert.match(svg,/<rect x="320.0" y="180.0" width="64.0" height="36.0"/);
assert.match(svg,/CAR 82% · Exit/);
assert.match(svg,/<g class="chg alarm">/);
ctx.renderObjects({...s,objects:[],change_boxes:[]},{...status,aligned:false});
assert.equal(nodes['object-overlay'].innerHTML,'');
"""
        self.node(harness.replace('SOURCE', json.dumps(self.script())))


class CameFromNoteTests(unittest.TestCase):
    def test_event_mapping_says_where_the_object_was_first_seen(self):
        page = dashboard_ui.dashboard_page('BarrierWatcher')
        start = page.index('function eventData(event){')
        end = page.index('\n}', start) + 2
        harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const ctx={transport:{path:p=>p},shortZoneName:s=>s,sourceLabel:s=>s,routeRole:()=>'exit',Date,JSON,Math,Number,String,Array};
vm.runInNewContext(SOURCE,ctx);
const d=ctx.eventData({kind:'main',reason:'Pedestrian footpath: AI confirmed person arrival',time:'02:30:00',date:'2026-11-15',
  evidence:{source:'ai',zone_name:'Pedestrian footpath',label:'person',came_from:'outside the zones',tracked_s:12.4}});
assert.equal(d.cameFrom,'Person first seen outside the zones, followed for 12 s.');
const e=ctx.eventData({kind:'main',reason:'x',evidence:{source:'motion'}});
assert.equal(e.cameFrom,'');
"""
        if not Path(NODE).exists():
            self.skipTest('Node.js is required')
        result = subprocess.run([NODE, '-'], input=harness.replace('SOURCE', json.dumps(page[start:end])), text=True,
                                capture_output=True, encoding='utf-8', timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
