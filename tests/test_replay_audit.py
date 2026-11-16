"""Offline harness safety/metrics tests. All inputs and state are disposable."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('replay_audit_under_test',ROOT/'tools/replay_audit.py')
replay=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=replay
spec.loader.exec_module(replay)


class ReplayAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='replay-audit-tests-')
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.config={'zones':[{'name':'Gate','rect':[.1,.1,.8,.8],'trigger':'ai','motion':False,
                       'light':False,'classes':['car','person'],'route_group':'entry','route_role':'entrance',
                       'ai_confidence':.6,'ai_confirm_observations':2}],
                     'alignment':{'enabled':False},'ai':{'interval_s':.5},
                     'barriers':{'reminder':False}}
        (self.folder/'zones.json').write_text(json.dumps(self.config))

    def manifest(self,**changes):
        car={'label':'car','confidence':.9,'bbox':[160,100,220,180]}
        value={'version':1,'mode':'route_only','config':'zones.json','coordinate_space':'aligned',
               'runtime':{'warmup_s':0},'duration_s':2.,'labels_complete':True,
               'frames':[{'t':0.,'detections':[]},{'t':.5,'detections':[car]},
                         {'t':1.,'detections':[car]},{'t':1.5,'detections':[car]},
                         {'t':2.,'detections':[car]}],
               'labels':[{'id':'arrival','route':'entry','visible_at':.5,'end_at':2.,'require_alarm':True}]}
        value.update(changes)
        path=self.folder/'manifest.json'
        path.write_text(json.dumps(value))
        return path

    def run_manifest(self,path,**kwargs):
        return replay.run_replay(path,state_parent=self.folder,**kwargs)

    def test_real_watcher_policy_accepts_once_and_measures_labelled_delay(self):
        report=self.run_manifest(self.manifest())
        self.assertTrue(report['complete'])
        self.assertEqual(1.,report['metrics']['recall'])
        self.assertEqual([.5],report['metrics']['time_to_alarm_s'])
        self.assertEqual(0.,report['metrics']['false_alarms_per_hour'])
        self.assertEqual(1,len([x for x in report['events'] if x['kind']=='main']))
        self.assertEqual('Synthetic detection policy replay',report['interpretation'])
        self.assertEqual([],list(self.folder.glob('barrier-replay-*')))

    def test_unlabelled_duration_and_exhaustiveness_are_not_invented(self):
        path=self.manifest(labels=[],labels_complete=False)
        data=json.loads(path.read_text());data.pop('duration_s');path.write_text(json.dumps(data))
        metrics=self.run_manifest(path)['metrics']
        self.assertIsNone(metrics['recall'])
        self.assertIsNone(metrics['false_alarms_per_hour'])
        self.assertIsNone(metrics['wrong_ack_count'])
        self.assertIsNone(metrics['barrier_accuracy'])
        self.assertIsNone(metrics['alignment_lock_recovery'])

    def test_explicit_negative_label_exposes_false_alarm(self):
        report=self.run_manifest(self.manifest(labels=[{'id':'parking','route':'entry',
                  'visible_at':0.,'end_at':2.,'require_alarm':False}]))
        self.assertEqual(1800.,report['metrics']['false_alarms_per_hour'])
        self.assertIsNone(report['metrics']['recall'])

    def test_pedestrian_without_route_group_matches_zone_name(self):
        self.config['zones'][0].pop('route_group')
        self.config['zones'][0]['route_role']='pedestrian'
        (self.folder/'zones.json').write_text(json.dumps(self.config))
        path=self.manifest(labels=[{'id':'pedestrian','route':'Gate','visible_at':.5,
                                   'end_at':2.,'require_alarm':True}])
        data=json.loads(path.read_text())
        for frame in data['frames']:
            for obj in frame['detections']:obj['label']='person'
        path.write_text(json.dumps(data))
        report=self.run_manifest(path)
        self.assertEqual(1.,report['metrics']['recall'])
        self.assertEqual(0.,report['metrics']['false_alarms_per_hour'])

    def test_explicit_disallowed_ack_scores_actual_policy_ack(self):
        path=self.manifest()
        data=json.loads(path.read_text())
        data['frames'][3]['ack']={'via':'api'}
        data['frames'][3]['expected_ack']=False
        path.write_text(json.dumps(data))
        report=self.run_manifest(path)
        self.assertEqual(1,report['metrics']['wrong_ack_count'])
        self.assertTrue(any(x['kind']=='ack' for x in report['events']))

    def test_truncated_run_does_not_claim_recall_or_hourly_rate(self):
        report=self.run_manifest(self.manifest(),max_frames=2)
        self.assertFalse(report['complete'])
        self.assertIsNone(report['metrics']['recall'])
        self.assertIsNone(report['metrics']['false_alarms_per_hour'])

    def test_image_mode_rejects_injected_detections_and_missing_sources(self):
        for frames in ([{'t':0.,'path':'x.jpg','detections':[]}],[{'t':0.}]):
            with self.subTest(frames=frames),self.assertRaises(ValueError):
                replay.load_manifest(self.manifest(mode='image',frames=frames))

    def test_timestamp_order_is_strict(self):
        with self.assertRaisesRegex(ValueError,'strictly increase'):
            replay.load_manifest(self.manifest(frames=[{'t':1.},{'t':1.}]))

    def test_paths_reject_network_and_raw_requires_reference(self):
        for value in ('http://localhost/scene.jpg','\\\\server\\share\\x.jpg','//server/x'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                replay.local_path(value,self.folder)
        with self.assertRaisesRegex(ValueError,'alignment_reference'):
            replay.load_manifest(self.manifest(coordinate_space='raw'))

    def test_no_real_platform_state_or_external_services_are_started(self):
        import platform_support
        path=self.manifest()
        with patch.object(platform_support,'state_dir',side_effect=AssertionError('Production state accessed')), \
             patch.object(replay.subprocess,'Popen',side_effect=AssertionError('Process started')), \
             patch.object(replay.socket,'socket',side_effect=AssertionError('Network opened')):
            report=self.run_manifest(path)
        self.assertTrue(report['isolated'])

    def test_image_mode_runs_detector_and_watcher_without_route_substitution(self):
        image=self.folder/'empty.png'
        Image.fromarray(np.full((360,640,3),80,np.uint8)).save(image)
        class Detector:
            available=True
            calls=0
            def __init__(self,*args,**kwargs):
                self.threads=kwargs['threads']
                self.engine=kwargs['engine']
            def detect(self,frame):
                type(self).calls+=1
                return []
        frames=[{'t':t,'path':'empty.png'} for t in (0.,.5,1.)]
        path=self.manifest(mode='image',model='fake-model.onnx',frames=frames,labels=[],duration_s=1.)
        with patch('object_detector.ObjectDetector',Detector):
            report=self.run_manifest(path)
        self.assertEqual(4,Detector.calls)
        self.assertEqual('Image detector and watcher-policy replay',report['interpretation'])
        self.assertEqual(0.,report['metrics']['false_alarms_per_hour'])

    def test_raw_reference_runs_real_alignment_and_scores_explicit_lock_label(self):
        # Leave distributed static landmarks outside the motion/AI region.
        self.config['zones'][0]['rect']=[.4,.4,.1,.1]
        (self.folder/'zones.json').write_text(json.dumps(self.config))
        rng=np.random.default_rng(1)
        image=self.folder/'reference.png'
        Image.fromarray(rng.integers(0,255,(360,640,3),dtype=np.uint8)).save(image)
        path=self.manifest(coordinate_space='raw',alignment_reference='reference.png',
                           frames=[{'t':0.,'path':'reference.png'},{'t':.5,'path':'reference.png'}],
                           labels=[],duration_s=.5,alignment_stops=[{'at':0.,'end_at':.5}])
        report=self.run_manifest(path)
        self.assertEqual('tracking',report['observations'][-1]['alignment'],report['observations'][-1]['alignment_detail'])
        self.assertEqual(0.,report['metrics']['alignment_lock_recovery'][0]['lock_delay_s'])


if __name__=='__main__':unittest.main()
