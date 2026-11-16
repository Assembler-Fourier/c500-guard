#!/usr/bin/env python3
"""Isolated, bounded replay through the real Watcher.process/alarm policy.

Usage: python tools/replay_audit.py manifest.json --output report.json

Manifest v1: {"version":1, "mode":"image"|"route_only", "config":"zones.json",
 "coordinate_space":"aligned"|"raw", "frames":[{"t":0,"path":"frame.jpg"}],
 "model":"yolox_tiny.onnx", "labels_complete":false, "duration_s":10,
 "labels":[{"id":"car-1","route":"campus-entrance","visible_at":2,
             "end_at":9,"require_alarm":true}], "runtime":{"warmup_s":2}}

Frame alternatives: {"t":1,"video":"clip.mp4","video_t":3.5}; route_only
frames instead supply pixel-xyxy "detections" and an optional image. This mode
uses synthetic detections, never claims model recall, and defaults to gray frames.
Explicit "ack":{"via":"input-any-key"} simulates input on the isolated watcher.
Optional frame "expected_ack":false labels whether ANY acknowledgement is allowed.
Optional frame "barrier_labels":{"exit":"down"} scores real calibrated readings.

Raw input requires "alignment_reference" (local image or reference NPZ). Optional
"barrier_templates" is a local NPZ matching the config and "source_identity".
An aligned input bypasses registration and cannot score alignment. To measure
lock recovery, raw manifests may label "alignment_stops":[{"at":5,"end_at":12}].

Paths resolve relative to the manifest. No camera, network, subprocess, input hook,
audio, Windows volume change, or production state is used. The synchronous replay
does not measure worker scheduling, asynchronous recovery or wall-clock latency.
Annotated recordings are accepted only with "annotated":true and their metrics
are marked diagnostic: overlays can change the detector's answers.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def finite(value, name, minimum=0.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be finite and >= {minimum}')
    return float(value)


def local_path(value, base):
    if not isinstance(value, str) or not value or '://' in value or value.startswith(('\\\\', '//')):
        raise ValueError('Replay inputs must be explicit local filesystem paths')
    result = Path(value)
    return (base / result).resolve() if not result.is_absolute() else result.resolve()


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('version') != 1 or data.get('mode') not in ('image', 'route_only'):
        raise ValueError('Require manifest version 1 and explicit image or route_only mode')
    if data.get('coordinate_space') not in ('aligned', 'raw'):
        raise ValueError('Declare coordinate_space as aligned or raw')
    if data['coordinate_space'] == 'raw' and not data.get('alignment_reference'):
        raise ValueError('Raw replay requires an explicit alignment_reference')
    frames = data.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('frames must be a nonempty list with explicit timestamps')
    previous = -1.
    for frame in frames:
        stamp = finite(frame.get('t'), 'frame.t')
        if stamp <= previous:
            raise ValueError('Frame timestamps must strictly increase')
        previous = stamp
        if data['mode'] == 'image' and not (frame.get('path') or frame.get('video')):
            raise ValueError('Image replay needs a local image or video for every frame')
        if frame.get('path') and frame.get('video'):
            raise ValueError('A frame cannot name both an image and a video')
        if frame.get('video'):
            finite(frame.get('video_t'), 'frame.video_t')
        if data['mode'] == 'image' and 'detections' in frame:
            raise ValueError('Image replay cannot substitute supplied detections')
    for label in data.get('labels', []):
        begin = finite(label.get('visible_at'), 'label.visible_at')
        if finite(label.get('end_at'), 'label.end_at') < begin or not label.get('id') or not label.get('route'):
            raise ValueError('Labels need id, route and an ordered visible_at/end_at window')
        if type(label.get('require_alarm')) is not bool:
            raise ValueError('Label require_alarm must be explicitly true or false')
    if 'duration_s' in data:
        finite(data['duration_s'], 'duration_s', .001)
    data['_base'] = path.parent
    data['_manifest'] = str(path)
    return data


class Clock:
    def __init__(self):
        self.t = 1000.
    def monotonic(self):
        return self.t
    def time(self):
        return 1_800_000_000. + self.t - 1000.
    def __getattr__(self, name):
        return getattr(time, name)


class FakeBuzzer:
    def __init__(self, *args, **kwargs):
        self.want = None
    def set(self, kind):
        self.want = kind
    def shutdown(self):
        self.want = None


class FakeLoudness:
    def __init__(self, *args, **kwargs):
        self.kind = None
    def begin(self, kind, **kwargs):
        self.kind = kind
    def end(self):
        self.kind = None
    def recover_previous(self): pass
    def evidence(self, **kwargs): pass
    def tick(self): pass


class DisabledService:
    """No filesystem, process, window, device or worker operations."""
    enabled = False
    restarts = 0
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def forbidden(*args, **kwargs):
    raise RuntimeError('Replay attempted a forbidden thread, process, network or power operation')


class SynchronousSemantic:
    """Real detector/crops/_confirm; no queues, worker or invented extra observations."""
    def __init__(self, zones, config, manifest, clock):
        from semantic_monitor import SemanticMonitor
        from object_tracking import ObjectVisitTracker
        self.impl = SemanticMonitor.__new__(SemanticMonitor)
        self.impl.zones, self.impl.confidence = zones, .6
        self.impl._tracker, self.impl._status = ObjectVisitTracker(), {}
        self.clock, self.manifest = clock, manifest
        self.interval = finite((config.get('ai') or {}).get('interval_s', .5), 'ai.interval_s')
        self.context_view = (config.get('ai') or {}).get('whole_picture', True) is not False
        self.detector, self.objects, self._frame, self._stamp = None, [], None, None
        self._last = -math.inf
        self._enabled = False
        if manifest['mode'] == 'image':
            from object_detector import ObjectDetector
            model = local_path(manifest['model'], manifest['_base'])
            self.detector = ObjectDetector(model, confidence=self.impl._candidate_confidence(), threads=1, engine='processor')
            if not self.detector.available:
                raise RuntimeError(self.detector.status().get('error', 'Detector failed'))

    def reset(self):
        self.impl._tracker.reset()
        self.impl._status = {}
        self._stamp = None
        self._last = -math.inf

    def update(self, frame, enabled, stamp=None):
        if enabled != self._enabled:
            self.reset()
            self._enabled = enabled
        if not enabled or stamp - self._last < self.interval - 1e-9:
            return []
        self._last, self._stamp, self._frame = stamp, stamp, frame
        objects = self.objects
        if self.detector:
            from regional_inference import inference_rects, merge_view_detections, add_context_detections
            pixels = np.asarray(frame, dtype=np.uint8)
            rects = inference_rects(self.impl.zones, pixels.shape[1], pixels.shape[0])
            objects = merge_view_detections([(self.detector.detect(pixels[y1:y2,x1:x2]), rect)
                         for rect in rects for x1,y1,x2,y2 in [rect]])
            if self.context_view:
                objects = add_context_detections(objects, self.detector.detect(pixels), rects)
            if not self.detector.available:
                raise RuntimeError(self.detector.status().get('error', 'Detector failed'))
        return self.impl._confirm(objects, frame.shape[1], frame.shape[0], stamp)

    def status(self):
        return {'available':True, 'state':'ready' if self._stamp is not None else 'waiting_for_frames',
                'observed_at':self._stamp, 'age_s':self.clock.t-self._stamp if self._stamp else None,
                'detections':copy.deepcopy(self.impl._status.get('detections', []))}

    def mark_emitted(self, event): return self.impl._tracker.mark_emitted(event)
    def event_context(self, event): return self.impl._tracker.event_context(event, self.impl.zones)
    def frame_at(self, stamp): return self._frame if stamp == self._stamp else None


def read_frame(item, base, mode):
    if item.get('path'):
        with Image.open(local_path(item['path'], base)) as image:
            frame = np.asarray(image.convert('RGB'))
    elif item.get('video'):
        import cv2
        capture = cv2.VideoCapture(str(local_path(item['video'], base)))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, item['video_t'] * 1000)
            okay, bgr = capture.read()
            if not okay:
                raise ValueError(f'Cannot decode video frame at {item["video_t"]} seconds')
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
    else:
        frame = np.full((360,640,3), 80, np.uint8)
    if frame.shape != (360,640,3):
        raise ValueError(f'Replay requires 640x360 RGB source frames, got {frame.shape}; precompute explicit resizing')
    return frame


def metrics(manifest, rows, events, complete, barrier_enabled, alignment_enabled):
    alarms = [event for event in events if event['kind'] in ('main', 'early')]
    positives = [x for x in manifest.get('labels', []) if x['require_alarm']]
    matched, used = [], set()
    for label in positives:
        match = next((i for i,event in enumerate(alarms) if i not in used
                      and label['visible_at'] <= event['t'] <= label['end_at']
                      and (event.get('evidence', {}).get('route_group')
                           or event.get('evidence', {}).get('zone_name')) == label['route']), None)
        if match is not None:
            used.add(match)
            matched.append({'label':label['id'],'alarm_id':alarms[match].get('alarm_id'),
                            'delay_s':round(alarms[match]['t']-label['visible_at'],6)})
    duration = manifest.get('duration_s')
    exhaustive = manifest.get('labels_complete') is True
    full_duration = bool(complete and duration and rows and rows[0]['t'] == 0 and rows[-1]['t'] >= duration)
    score_barriers = [(row['barriers'].get(arm) == expected)
                      for row in rows for arm,expected in row.get('barrier_labels', {}).items()]
    ack_labels = [row for row in rows if type(row.get('expected_ack')) is bool]
    recoveries = []
    if alignment_enabled:
        for stop in manifest.get('alignment_stops', []):
            found = next((r for r in rows if stop['at'] <= r['t'] <= stop['end_at'] and r['alignment'] == 'tracking'), None)
            recoveries.append({'stopped_at':stop['at'],'lock_delay_s':None if found is None else round(found['t']-stop['at'],6)})
    return {'recall':len(matched)/len(positives) if positives and complete else None,
            'labelled_arrivals':len(positives),'matched_arrivals':len(matched),'matches':matched,
            'time_to_alarm_s':[x['delay_s'] for x in matched] if positives and complete else None,
            'recall_within_2s':sum(x['delay_s'] <= 2 for x in matched)/len(positives) if positives and complete else None,
            'unmatched_alarm_count':len(alarms)-len(used),
            'false_alarms_per_hour':(len(alarms)-len(used))*3600/duration if exhaustive and full_duration else None,
            'false_alarm_metric_reason':None if exhaustive and full_duration else 'Needs exhaustive labels, explicit duration and complete coverage from t=0 through duration.',
            'wrong_ack_count':sum(row['acknowledged'] and not row['expected_ack'] for row in ack_labels) if ack_labels else None,
            'missed_expected_ack_count':sum(not row['acknowledged'] and row['expected_ack'] for row in ack_labels) if ack_labels else None,
            'barrier_accuracy':sum(score_barriers)/len(score_barriers) if barrier_enabled and score_barriers else None,
            'barrier_label_count':len(score_barriers),
            'alignment_lock_recovery':recoveries if alignment_enabled and manifest.get('alignment_stops') else None}


def run_replay(manifest_path, *, max_frames=2000, max_wall_s=120., state_parent=None):
    manifest = load_manifest(manifest_path)
    config_path = local_path(manifest['config'], manifest['_base'])
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    from zone_editor import validate_config
    validate_config(config)
    started = time.monotonic()
    clock, events, rows = Clock(), [], []
    state_parent = Path(state_parent or tempfile.gettempdir()).resolve()
    state_parent.mkdir(parents=True, exist_ok=True)
    import platform_support
    import secondary_analysis, semantic_monitor, alarm_recorder, arm_angle, vehicle_sightings
    module_name = '_barrierwatcher_isolated_replay'
    prior = sys.modules.get(module_name)
    with tempfile.TemporaryDirectory(prefix='barrier-replay-', dir=state_parent) as scratch_name, ExitStack() as stack:
        scratch = Path(scratch_name)
        stack.enter_context(patch.object(platform_support, 'state_dir', return_value=scratch))
        stack.enter_context(patch.object(threading.Thread, 'start', side_effect=forbidden))
        stack.enter_context(patch.object(subprocess, 'Popen', side_effect=forbidden))
        stack.enter_context(patch.object(socket, 'socket', side_effect=forbidden))
        # Python's Windows platform.system() can launch `ver` during ONNX import.
        # The OS family is already known; keep the process guard intact.
        os_family = {'win32':'Windows','darwin':'Darwin'}.get(sys.platform,'Linux')
        stack.enter_context(patch.object(platform,'system',return_value=os_family))
        if sys.platform == 'win32':
            # ONNX also asks for release; the native API gives the kernel
            # major version without Python's optional subprocess fallback.
            stack.enter_context(patch.object(platform,'release',return_value=str(sys.getwindowsversion().major)))
        spec = importlib.util.spec_from_file_location(module_name, SOURCE/'watch.py')
        watch = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = watch
        try:
            spec.loader.exec_module(watch)
            stack.enter_context(patch.object(watch, 'time', clock))
            for name,replacement in [('Buzzer',FakeBuzzer),('AlarmLoudness',FakeLoudness),
                                     ('FrameReader',DisabledService),('GuardPresence',DisabledService),
                                     ('notify',lambda *a,**k:None),('ensure_sounds',lambda:{}),
                                     ('request_full_off',forbidden)]:
                stack.enter_context(patch.object(watch,name,replacement))
            stack.enter_context(patch.object(watch.Watcher,'_load_armed',return_value=True))
            stack.enter_context(patch.object(watch.Watcher,'_load_audio_settings',return_value={}))
            stack.enter_context(patch.object(platform_support,'SleepInhibitor',DisabledService))
            stack.enter_context(patch.object(secondary_analysis,'SecondaryAnalyzer',DisabledService))
            stack.enter_context(patch.object(semantic_monitor,'SemanticMonitor',return_value=None))
            stack.enter_context(patch.object(alarm_recorder,'AlarmRecorder',DisabledService))
            stack.enter_context(patch.object(arm_angle,'ArmMotionRecorder',DisabledService))
            stack.enter_context(patch.object(vehicle_sightings,'VehicleSightings',DisabledService))
            source = manifest.get('source_identity','offline-replay')
            isolated = copy.deepcopy(config)
            alignment_enabled = manifest['coordinate_space'] == 'raw'
            isolated.setdefault('alignment',{})['enabled'] = alignment_enabled
            copied_config = scratch/'zones.json'
            copied_config.write_text(json.dumps(isolated),encoding='utf-8')
            if alignment_enabled:
                path = local_path(manifest['alignment_reference'],manifest['_base'])
                if path.suffix.lower() == '.npz':
                    with np.load(path,allow_pickle=False) as data: reference=data['reference'].copy()
                else:
                    with Image.open(path) as image: reference=np.asarray(image.convert('RGB'))
                if reference.shape != (360,640,3): raise ValueError('Alignment reference must be 640x360 RGB')
                view = isolated.get('view')
                key = hashlib.sha256((source+(f'|{view}' if view else '')).encode()).hexdigest()[:16]
                np.savez_compressed(scratch/f'alignment-{key}.npz',reference=reference)
            barrier_enabled = bool(manifest.get('barrier_templates'))
            if barrier_enabled:
                lines = (config.get('barriers') or {}).get('lines')
                identity = source + ('|'+json.dumps(lines,sort_keys=True) if lines is not None else '')
                key = hashlib.sha256(identity.encode()).hexdigest()[:16]
                asset = local_path(manifest['barrier_templates'],manifest['_base'])
                with np.load(asset,allow_pickle=False) as stored:
                    if str(stored['source_key'].item()) != key: raise ValueError('Barrier template identity/config does not match manifest')
                shutil.copyfile(asset,scratch/f'barriers-{key}.npz')
            actual_workflow = watch.BarrierWorkflow
            stack.enter_context(patch.object(watch,'BarrierWorkflow',side_effect=lambda source,state_dir,seed_path,**kw:
                actual_workflow(source,state_dir,None,clock=clock.monotonic,**kw)))
            runtime = manifest.get('runtime') or {}
            args=SimpleNamespace(source=source,fps=10.,zones=str(copied_config),silent=True,connect_grace=30.,
                                 lost_after=3.,freeze_after=20.,black_after=5.,
                                 warmup=finite(runtime.get('warmup_s',2.),'warmup_s'),pixel_threshold=22.,adapt=.02)
            watcher = watch.Watcher(args)
            watcher.learning,watcher.recorder,watcher.sightings = None,None,None
            watcher._queue_snapshot=lambda kind:None
            watcher.log_event=lambda kind,detail,**kw:events.append({'t':round(clock.t-1000.,6),'kind':kind,'detail':detail,**copy.deepcopy(kw)})
            if not alignment_enabled:
                watcher.alignment={'state':'tracking','message':'Manifest declares already aligned input; alignment not evaluated'}
                watcher._alignment_blocked=False
            # Restore the class before constructing the explicit synchronous adapter.
            stack.enter_context(patch.object(semantic_monitor,'SemanticMonitor',SynchronousSemantic._semantic_class))
            watcher.semantic=SynchronousSemantic(watcher.zones,config,manifest,clock)
            for item in manifest['frames']:
                if len(rows)>=max_frames or time.monotonic()-started>=max_wall_s: break
                clock.t=1000.+item['t']
                frame=read_frame(item,manifest['_base'],manifest['mode'])
                watcher.frame_count+=1
                watcher.first_frame_seen=True
                watcher.last_frame_at=clock.t
                watcher.semantic.objects=copy.deepcopy(item.get('detections',[]))
                before=len(events)
                aligned=watcher.align_frame(frame) if alignment_enabled else frame
                if aligned is not None:
                    watcher.process(aligned.astype(np.float32),received_at=clock.t)
                else:
                    watcher.semantic.reset()
                watcher.check_health(clock.t)
                if item.get('ack'):
                    watcher.acknowledge(via=str(item['ack'].get('via','api')))
                status=watcher.barrier_workflow.status() if watcher.barrier_workflow else {}
                rows.append({'t':item['t'],'alarm':watcher.alarm,'alarm_id':watcher.alarm_id,
                             'detections':watcher.semantic.status()['detections'],
                             'alignment':watcher.alignment.get('state'),
                             'alignment_detail':copy.deepcopy(watcher.alignment),
                             'barriers':{x['id']:x['state'] for x in status.get('barriers',[])},
                             'barrier_labels':item.get('barrier_labels',{}),'expected_ack':item.get('expected_ack'),
                             'acknowledged':any(e['kind']=='ack' or e['kind'].startswith('auto-barrier-') for e in events[before:])})
            complete=len(rows)==len(manifest['frames'])
            report={'version':1,'mode':manifest['mode'],'complete':complete,'frames_processed':len(rows),
                    'frames_requested':len(manifest['frames']),'elapsed_wall_s':round(time.monotonic()-started,3),
                    'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    'isolated':True,'external_operations':'Thread/process/socket starts forbidden; fake audio/input/device services; no Watcher.run.',
                    'interpretation':'Synthetic detection policy replay' if manifest['mode']=='route_only' else
                        'Annotated image diagnostic; overlays may distort detection' if manifest.get('annotated') else 'Image detector and watcher-policy replay',
                    'limitations':['Synchronous inference excludes worker scheduling/recovery and real wall-clock alarm latency.',
                                   'No full-night or unlabelled accuracy claim; null metrics mean not established.',
                                   'Sound, Windows volume, capture device and UI are fake/absent.'],
                    'metrics':metrics(manifest,rows,events,complete,barrier_enabled,alignment_enabled),
                    'events':events,'observations':rows}
            return report
        finally:
            if prior is None:sys.modules.pop(module_name,None)
            else:sys.modules[module_name]=prior


# Capture the real class once, before the Watcher constructor's no-worker patch.
from semantic_monitor import SemanticMonitor
SynchronousSemantic._semantic_class = SemanticMonitor


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-frames',type=int,default=2000)
    parser.add_argument('--max-wall-s',type=float,default=120.)
    args=parser.parse_args(argv)
    if not 1<=args.max_frames<=10000 or not 0<args.max_wall_s<=3600:
        parser.error('Limits: 1..10000 frames, >0..3600 wall seconds')
    output=args.output.resolve()
    forbidden_roots=[Path('C:/c500-guard').resolve(),Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'c500-guard']
    if any(output==p or p in output.parents for p in forbidden_roots):
        parser.error('Output must not be in production code or runtime state')
    output.parent.mkdir(parents=True,exist_ok=True)
    report=run_replay(args.manifest,max_frames=args.max_frames,max_wall_s=args.max_wall_s,state_parent=output.parent)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'report':str(output),'complete':report['complete'],'metrics':report['metrics']},indent=2))
    return 0 if report['complete'] else 2


if __name__=='__main__':
    raise SystemExit(main())
