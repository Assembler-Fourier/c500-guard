"""Guided camera setup: fake state/DOM/transport only; no camera or control calls."""
import re
import unittest
import test_live_dashboard_freshness as freshness


class CameraSetupUITests(unittest.TestCase):
    js=freshness.LiveDashboardFreshnessTests.js

    @classmethod
    def setUpClass(cls):
        cls.page=freshness.page_source()

    def section(self,start,end):
        return self.page.split(start,1)[1].split(end,1)[0]

    def helpers(self):
        return self.section('// CAMERA_SETUP_HELPERS_START','// CAMERA_SETUP_HELPERS_END')

    def flow(self):
        return self.section('// CAMERA_SETUP_FLOW_START','// CAMERA_SETUP_FLOW_END')

    def baseline(self):
        return """
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const context={live:true,connected:true,cameraFresh:true,pictureFresh:true,aligned:true};
const checks={view:true,exit:true,entrance:true,pedestrian:true};
const ready={armed:false,camera_view:'PTZ 2',camera_setup:{version:1,view:'PTZ 2',reference_id:'profile:geometry:1',state:'ready',score_pct:100,stable_s:5,required_stable_s:5,ready_to_save:true,blockers:[],guidance:{action:'good',headline:'All areas visible',detail:'Keep this view'},coverage:{required_visible:8,total_required:8,pct:100},alignment:{inliers:38},reference:{saved:true},calibration:{pending:false,error:'',revision:4}}};
"""+self.helpers()

    def dom(self):
        return """
let now=1000,lastStateAt=1000,page='setup',setupTab='camera',production=true,commandBusy=false,connectionOk=true,shutdownRequested=false,active=true;
const performance={now:()=>now};let state=structuredClone(ready);const setupVisible=tab=>page==='setup'&&setupTab===tab;
const currentState=()=>state,deriveStatus=()=>context,barrierContext=()=>context;
const nodes=new Map();const $=id=>{if(!nodes.has(id))nodes.set(id,{textContent:'',hidden:false,checked:false,disabled:false,className:'',title:'',setAttribute(){},replaceChildren(...nodes){this.children=nodes}});return nodes.get(id)};
const document={hidden:false,createElement:()=>({textContent:''})};
const storage=new Map();const localStorage={getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value)};
let renders=0;const render=()=>renders++;
const clearTimeout=()=>{},setTimeout=()=>1;
"""+self.flow()

    def test_current_readiness_requires_all_checks_and_exact_stability(self):
        self.js(self.baseline()+"""
const good=cameraSetupPresentation(ready,context,checks);
assert.equal(good.canFinish,true);assert.equal(good.canCapture,true);assert.equal(good.score,100);
assert.equal(cameraSetupPresentation(ready,{...context,aligned:false},checks).ready,false);
assert.equal(cameraSetupPresentation({...ready,armed:true},context,checks).canCapture,false);
assert.equal(cameraSetupPresentation({...ready,armed:true},context,checks).canFinish,true,'Review does not capture or pause');
for(const key of Object.keys(checks))assert.equal(cameraSetupPresentation(ready,context,{...checks,[key]:false}).canFinish,false);
for(const patch of [{stable_s:4.99},{ready_to_save:false},{state:'adjust'},{reference_id:''},{calibration:{pending:true}}])assert.equal(cameraSetupPresentation({...ready,camera_setup:{...ready.camera_setup,...patch}},context,checks).canCapture,false);
""")

    def test_stale_missing_or_malformed_telemetry_never_claims_setup_score(self):
        self.js(self.baseline()+"""
for(const key of ['live','connected','cameraFresh','pictureFresh']){const result=cameraSetupPresentation(ready,{...context,[key]:false},checks);assert.equal(result.score,null);assert.equal(result.ready,false);assert.equal(result.stable,0);assert.equal(result.guidance,null);}
for(const data of [null,{}, {...ready.camera_setup,version:2},{...ready.camera_setup,state:'waiting'}]){const result=cameraSetupPresentation({...ready,camera_setup:data},context,checks);assert.equal(result.score,null);assert.equal(result.ready,false);}
for(const score of ['100',NaN,Infinity,-1,101])assert.equal(cameraSetupPresentation({...ready,camera_setup:{...ready.camera_setup,score_pct:score}},context,checks).score,null);
""")

    def test_review_is_bound_to_view_and_reference_and_never_automatically_captures(self):
        self.js(self.baseline()+self.dom()+"""
let calls=0;const transport={command:()=>{calls++;throw new Error('No backend action expected')}};
syncSetupReview(cameraSetupPresentation(state,context,checks).identity);setupChecks={...checks};finishCameraSetup();
assert.equal(calls,0);assert.equal(storage.size,1);assert.equal(setupReviewPersistent,true);
const original=setupIdentity;syncSetupReview('changed geometry');assert.equal(setupChecks.exit,false);assert.equal(setupCompletedAt,'');
syncSetupReview(original);assert.equal(setupChecks.exit,true);assert.ok(setupCompletedAt);
state.camera_setup.reference_id='different';finishCameraSetup();assert.equal(storage.size,1);assert.equal(calls,0);
""")

    def test_guided_capture_requires_paused_empty_reviewed_current_identity(self):
        self.js(self.baseline()+self.dom()+"""
const calls=[];const transport={command:async(...args)=>{calls.push(args);return {ok:true,pending:true}}};
syncSetupReview(cameraSetupPresentation(state,context,checks).identity);setupChecks={...checks};
await captureSetupReference();assert.equal(calls.length,0,'Empty road confirmation required');
$('setup-empty-confirm').checked=true;state.armed=true;await captureSetupReference();assert.equal(calls.length,0);
state.armed=false;await captureSetupReference();assert.equal(calls.length,1);
assert.deepEqual(calls[0],['alignment/calibrate',{guided:true,reference_id:'profile:geometry:1'}]);
assert.ok(setupCaptureRequest,'Accepted HTTP response is not completed save');assert.match($('setup-action-status').textContent,/requested/);
await captureSetupReference();assert.equal(calls.length,1,'No duplicate capture while queued');
""")

    def test_success_requires_new_completed_backend_revision(self):
        self.js(self.baseline()+self.dom()+"""
syncSetupReview(cameraSetupPresentation(state,context,checks).identity);setupChecks={...checks};
setupCaptureRequest={view:'PTZ 2',revision:4,submittedAt:1000};now=lastStateAt=1100;
renderCameraSetup(state,context);assert.ok(setupCaptureRequest);assert.match($('setup-action-status').textContent,/waiting/);
state.camera_setup.calibration={pending:true,error:'',revision:5};renderCameraSetup(state,context);assert.ok(setupCaptureRequest);
state.camera_setup.calibration.pending=false;renderCameraSetup(state,context);assert.equal(setupCaptureRequest,null);assert.match($('setup-action-status').textContent,/New reference saved/);
setupCaptureRequest={view:'PTZ 2',revision:5,submittedAt:1000};state.camera_setup.calibration={pending:false,error:'Keep lanes clear',revision:5};renderCameraSetup(state,context);assert.equal(setupCaptureRequest,null);assert.match($('setup-action-status').textContent,/not saved.*Keep lanes clear/);
""")

    def test_raw_preview_never_uses_canonical_zone_overlay(self):
        preview='function renderSetupPreview('+self.section('function renderSetupPreview(','// SETUP_PREVIEW_END')
        self.js("""
let page='setup',setupTab='camera',setupPreviewMode='raw',setupRawFrame='blob:raw',setupRawAt=1000,displayedFrame='blob:aligned';const setupVisible=tab=>page==='setup'&&setupTab===tab;
let now=1000;const performance={now:()=>now};const barrierContext=()=>({live:true,connected:true,cameraFresh:true,pictureFresh:true,aligned:true});
const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{src:null,innerHTML:'',hidden:false,attributes:new Set(),getAttribute(name){return this[name]},removeAttribute(name){delete this[name]},toggleAttribute(name,value){if(value)this.attributes.add(name);else this.attributes.delete(name)}});return nodes.get(id)};
$('route-overlay').innerHTML='<path d="canonical"/>';
"""+preview+"""
renderSetupPreview({},{});assert.equal($('setup-feed').src,'blob:raw');assert.equal($('setup-overlay').innerHTML,'');assert.equal($('setup-overlay').attributes.has('hidden'),true);assert.match($('setup-preview-caption').textContent,/Original camera input/);
now=3001;renderSetupPreview({},{});assert.equal($('setup-feed').hidden,true);assert.equal($('setup-feed').src,undefined);
setupPreviewMode='aligned';renderSetupPreview({},{});assert.equal($('setup-feed').src,'blob:aligned');assert.equal($('setup-overlay').innerHTML,'<path d="canonical"/>');
""")

    def test_no_unverified_canonical_outlines_on_aligned_preview(self):
        preview='function renderSetupPreview('+self.section('function renderSetupPreview(','// SETUP_PREVIEW_END')
        self.js("""
const page='setup',setupTab='camera',setupVisible=tab=>page==='setup'&&setupTab===tab,displayedFrame='blob:aligned',barrierContext=()=>({live:true,connected:true,cameraFresh:true,pictureFresh:true,aligned:false});
const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{innerHTML:'',getAttribute(){},toggleAttribute(){}});return nodes.get(id)};$('route-overlay').innerHTML='<path d="obsolete"/>';
"""+preview+"""
renderSetupPreview({},{});assert.equal($('setup-overlay').innerHTML,'');assert.match($('setup-preview-caption').textContent,/outlines hidden/);
""")

    def test_raw_requests_are_serial_visibility_scoped_and_cancelable(self):
        self.js(self.baseline()+self.dom()+"""
let requests=0,resolve;const revoked=[];const URL={createObjectURL:()=> 'blob:raw',revokeObjectURL:value=>revoked.push(value)};
const transport={path:value=>value};const fetch=async()=>{requests++;return new Promise(r=>resolve=r)};
class Image{decode(){return Promise.resolve()}}
setupPreviewMode='raw';const first=readSetupRawPreview();await readSetupRawPreview();assert.equal(requests,1);
stopSetupRawPreview();page='live';resolve({ok:true,blob:async()=>({})});await first;
assert.equal(setupRawFrame,null);assert.deepEqual(revoked,['blob:raw']);assert.equal(setupRawController,null);
await readSetupRawPreview();assert.equal(requests,1);page='setup';document.hidden=true;await readSetupRawPreview();assert.equal(requests,1);
""")

    def test_markup_has_labeled_steps_score_explanation_and_deliberate_save(self):
        from html.parser import HTMLParser
        class Parser(HTMLParser):
            def __init__(self):super().__init__();self.ids=[]
            def handle_starttag(self,tag,attrs):self.ids.extend(v for k,v in attrs if k=='id')
        parser=Parser();parser.feed(self.page)
        for name in ['setup-check-view','setup-check-exit','setup-check-entrance','setup-check-pedestrian','setup-empty-confirm','setup-finish','setup-capture','setup-stability']:
            self.assertEqual(parser.ids.count(name),1)
        self.assertIn('Measured setup checks, not detection accuracy.',self.page)
        self.assertIn('for="setup-stability"',self.page)
        self.assertRegex(self.page,r'id="setup-action-status" class="setup-action-status[^"]*" role="status" aria-live="polite"')
        self.assertNotIn('alignment/calibrate',self.section('async function readFrame(',"document.querySelectorAll('[data-page]')"))
        monitor=self.section('<section id="page-live"','<section id="page-events"')
        self.assertNotIn('id="framing-coach"',monitor)

    def test_raw_preview_recovers_after_offline_without_duplicate_or_hidden_reads(self):
        self.js(self.baseline()+self.dom()+"""
let requests=0,resolve;const transport={path:value=>value};
const fetch=async()=>{requests++;return new Promise(r=>resolve=r)};
const URL={createObjectURL:()=> 'blob:raw',revokeObjectURL:()=>{}};
class Image{decode(){return Promise.resolve()}}
setupPreviewMode='raw';connectionOk=false;await readSetupRawPreview();assert.equal(requests,0);assert.equal(setupRawTimer,null);
connectionOk=true;ensureSetupRawPreview();ensureSetupRawPreview();assert.equal(requests,1,'Recovery starts exactly one request');
stopSetupRawPreview();resolve({ok:true,blob:async()=>({})});await new Promise(r=>globalThis.setTimeout(r,0));
assert.equal(setupRawController,null);document.hidden=true;ensureSetupRawPreview();assert.equal(requests,1);
document.hidden=false;page='live';ensureSetupRawPreview();assert.equal(requests,1);
page='setup';setupRawTimer=123;ensureSetupRawPreview();assert.equal(requests,1,'A scheduled retry must not be duplicated');
setupRawTimer=null;shutdownRequested=true;ensureSetupRawPreview();assert.equal(requests,1);
""")

    def test_whole_production_script_is_valid_javascript(self):
        import subprocess
        scripts=re.findall(r'<script[^>]*>(.*?)</script>',self.page,re.S)
        self.assertTrue(scripts)
        for script in scripts:
            result=subprocess.run([freshness.NODE,'--input-type=module','--check'],input=script,text=True,encoding='utf-8',capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
