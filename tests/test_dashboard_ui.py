"""Template escaping, browser-script syntax, and control/polling behavior."""
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dashboard_ui

NODE = shutil.which("node") or str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")


class PageParser(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.ids = []
        self.labels = []
        self.elements = {}
        self.remote_assets = []
        self.scripts = []
        self.current_script = None
        self.feed(content)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.append(attrs["id"])
            self.elements[attrs["id"]] = attrs
        if tag == "label" and "for" in attrs:
            self.labels.append(attrs["for"])
        if tag == "script":
            self.current_script = {"attributes": attrs, "text": ""}
            self.scripts.append(self.current_script)
        if tag in ("script", "img", "link"):
            url = attrs.get("src", attrs.get("href", ""))
            if url.startswith(("http:", "https:", "//")):
                self.remote_assets.append(url)

    def handle_data(self, data):
        if self.current_script is not None:
            self.current_script["text"] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.current_script = None


class DashboardTests(unittest.TestCase):
    def config(self, name="Approach"):
        return {"zones": [{"name": name, "rect": [.25, .25, .2, .2],
                           "motion": True, "light": False}]}

    def node(self, script):
        if not Path(NODE).exists():
            self.skipTest("Node.js is required for browser-script behavior checks")
        result = subprocess.run([NODE, "-"], input=script, text=True,
                                capture_output=True, encoding="utf-8", timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_zone_labels_cannot_break_out_of_embedded_json(self):
        name = '</script><script>alert("x")</script>&\u2028\u2029'
        page = PageParser(dashboard_ui.zone_editor_page(self.config(name)))
        self.assertEqual(len(page.scripts), 3)
        data = next(s for s in page.scripts if s["attributes"].get("id") == "initial-config")
        self.assertNotIn("<", data["text"])
        self.assertEqual(json.loads(data["text"])["zones"][0]["name"], name)

    def test_pages_have_unique_ids_accessible_labels_and_no_remote_assets(self):
        for content in (dashboard_ui.PAGE, dashboard_ui.zone_editor_page(self.config())):
            page = PageParser(content)
            self.assertEqual(len(page.ids), len(set(page.ids)))
            self.assertTrue(set(page.labels).issubset(page.ids))
            self.assertFalse(page.remote_assets)
            self.assertIn('name="viewport"', content)

    def test_all_browser_javascript_compiles(self):
        scripts = []
        for content in (dashboard_ui.PAGE, dashboard_ui.zone_editor_page(self.config())):
            for script in PageParser(content).scripts:
                if script["attributes"].get("type") != "application/json":
                    scripts.append(script["text"])
        self.node("const vm=require('node:vm');\n" +
                  "for(const source of " + json.dumps(scripts) + ")new vm.Script(source);\n")

    def test_drag_precision_can_be_saved_without_native_step_mismatch(self):
        page = PageParser(dashboard_ui.zone_editor_page(self.config()))
        for coordinate in ("x", "y", "w", "h"):
            self.assertEqual(page.elements[coordinate]["step"], "any")
        self.assertIn("'/scene.jpg'", dashboard_ui.EDITOR_SCRIPT)

    def test_freshness_is_not_presented_as_end_to_end_delay_and_warmup_is_visible(self):
        self.node("const vm=require('node:vm'),assert=require('node:assert/strict');\n" +
                  "const context={window:{},document:{},setTimeout,clearTimeout,performance};\n" +
                  "vm.runInNewContext(" + json.dumps(dashboard_ui.CORE_SCRIPT) + ",context);\n" + r"""
const ui=context.window.C500UI;
assert.equal(ui.delayLabel({last_frame_age:.1,local_pipeline_ms:20,end_to_end_delay_s:null}),'Not measured');
assert.equal(ui.delayLabel({end_to_end_delay_s:0}),'0.0 s');
assert.equal(ui.deriveStatus({armed:false,alarm:'main'}).mode,'off');
assert.match(ui.deriveStatus({armed:false}).detail,/silent/);
assert.match(ui.deriveStatus({armed:true,warmup_remaining_s:2.3}).detail,/3 s/);
assert.equal(ui.deriveStatus({armed:true,alarm:'fault',warmup_remaining_s:3}).mode,'fault');
assert.match(ui.deriveStatus({armed:true},false).title,/offline/i);
""")

    def test_quiet_countdown_preserves_monitoring_and_fault_visibility(self):
        self.node("const vm=require('node:vm'),assert=require('node:assert/strict');\n" +
                  "const context={window:{},document:{},setTimeout,clearTimeout,performance};\n" +
                  "vm.runInNewContext(" + json.dumps(dashboard_ui.CORE_SCRIPT) + ",context);\n" + r"""
const ui=context.window.C500UI;
let state={armed:true,alarm:null,acked:true,ack_remaining_s:4.8,ack_quiet_s:5,frames:10,last_frame_age:.1,alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1}};
assert.equal(ui.deriveStatus(state).title,'Running · acknowledged');
assert.equal(ui.deriveStatus(state).mode,'on');
assert.match(ui.deriveStatus(state).detail,/5 s.*Detection continues/);
state.ack_remaining_s=2.1;assert.match(ui.deriveStatus(state).detail,/3 s/);
state.fault_reason='Camera disconnected';assert.match(ui.deriveStatus(state).detail,/Camera disconnected/);
state.ack_remaining_s=0;assert.match(ui.deriveStatus(state).title,/Fault/);
state.fault_reason=null;assert.equal(ui.deriveStatus(state).title,'Running · alerts armed');
state.ack_remaining_s=4;state.armed=false;assert.equal(ui.deriveStatus(state).title,'Running · alerts paused');
assert.equal(ui.quietRemaining(state),0);
assert.match(ui.deriveStatus({...state,armed:true},false).title,/offline/i);
assert.equal(ui.currentAlarmId({armed:true,alarm:'main',alarm_id:'episode-1'}),'episode-1');
for(const state of [{armed:true,alarm:'main'},{armed:false,alarm:'main',alarm_id:'old'},{armed:true,alarm:null,alarm_id:'old'}])assert.equal(ui.currentAlarmId(state),null);
""")

    def test_poller_never_overlaps_and_stops_scheduling(self):
        self.node("const vm=require('node:vm'),assert=require('node:assert/strict');\n" +
                  "const context={window:{},document:{},setTimeout,clearTimeout,performance};\n" +
                  "vm.runInNewContext(" + json.dumps(dashboard_ui.CORE_SCRIPT) + ",context);\n" + r"""
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
(async()=>{
 let active=0,peak=0,calls=0;
 const poller=context.window.C500UI.makePoller(async()=>{calls++;active++;peak=Math.max(peak,active);await sleep(25);active--;},10);
 poller.start();for(let i=0;i<20;i++)poller.kick();await sleep(90);poller.stop();
 const stoppedCalls=calls;await sleep(60);assert.equal(calls,stoppedCalls);assert.equal(peak,1);assert.ok(calls>=2);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")

    def test_http_error_message_is_retained(self):
        self.node("const vm=require('node:vm'),assert=require('node:assert/strict');\n" +
                  "const context={window:{},document:{},setTimeout,clearTimeout,performance,AbortController," +
                  "fetch:async()=>({ok:false,status:409,json:async()=>({error:'Disarm before calibrating.'})})};\n" +
                  "vm.runInNewContext(" + json.dumps(dashboard_ui.CORE_SCRIPT) + ",context);\n" +
                  "assert.rejects(context.window.C500UI.requestJSON('/alignment/calibrate'),/Disarm before calibrating/).catch(e=>{console.error(e);process.exitCode=1;});")

    def dashboard_fixture(self):
        return r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
class Element{
 constructor(){this.children=[];this.style={};this.hidden=false;this.disabled=false;this.textContent='';this.listeners={};this.classList={toggle(){},remove(){},add(){}};}
 focus(){this.focused=true;}append(...items){this.children.push(...items);}replaceChildren(){this.children=[];}setAttribute(){}addEventListener(name,callback){this.listeners[name]=callback;}querySelector(){return new Element();}
 set innerHTML(value){throw Error('Unsafe HTML rendering used');}
}
const elements=new Map(),documentListeners={},windowListeners={};
const context={window:{addEventListener(name,callback){windowListeners[name]=callback;}},document:{hidden:false,addEventListener(name,callback){documentListeners[name]=callback;},getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(){return new Element();},createTextNode(text){return {textContent:text,nodeType:3};}},setTimeout(){return 1;},clearTimeout(){},performance,AbortController,fetch:()=>new Promise(()=>{})};
vm.createContext(context);
"""

    def test_dashboard_controls_fail_closed_and_names_use_text_nodes(self):
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
vm.runInContext(`latest={armed:false,frames:12,last_frame_age:.2,zones:[{name:'<img onerror=alert(1)>',changed_pct:1}],ml:{available:true,state:'ready',age_s:.1}};online=true;render();`,context);
assert.equal(elements.get('power').disabled,false);assert.equal(elements.get('ack').disabled,true);assert.equal(elements.get('calibrate').disabled,false);
assert.equal(elements.get('mode-title').textContent,'Running · alerts paused');
const title=elements.get('zones').children[0].children[0].children[0];
assert.equal(title.children[1].nodeType,3);assert.equal(title.children[1].textContent,'<img onerror=alert(1)>');
assert.equal(elements.get('add-lighting').disabled,true);
vm.runInContext(`latest.alignment={state:'tracking',calibrated:true};render();`,context);
assert.equal(elements.get('add-lighting').disabled,false);
for(const state of ['holding','calibration_required','calibrating','disabled']){
 vm.runInContext(`latest.alignment.state=${JSON.stringify(state)};render();`,context);
 assert.equal(elements.get('add-lighting').disabled,true);
}
vm.runInContext(`latest.alignment={state:'calibration_required',calibrated:true,references:2,learning:true,lighting_status:'Checking'};render();`,context);
assert.equal(elements.get('add-lighting').disabled,true);assert.equal(elements.get('lighting-status').textContent,'Checking');
vm.runInContext(`latest.alignment={};latest.armed=true;latest.alarm='main';latest.alarm_id='episode-1';render();`,context);
assert.equal(elements.get('ack').disabled,false);assert.equal(elements.get('calibrate').disabled,true);assert.equal(elements.get('add-lighting').disabled,true);
vm.runInContext(`online=false;render();`,context);
assert.equal(elements.get('power').disabled,true);assert.equal(elements.get('ack').disabled,true);assert.equal(elements.get('calibrate').disabled,true);
assert.equal(elements.get('offline').hidden,false);
vm.runInContext(`online=true;latest.control_warning='Could not save OFF state.';render();`,context);
assert.equal(elements.get('control-warning').hidden,false);assert.equal(elements.get('control-warning').textContent,'Could not save OFF state.');
vm.runInContext(`latest.control_warning='';render();`,context);assert.equal(elements.get('control-warning').hidden,true);
""")

    def test_browser_back_restoration_restarts_preview_and_disables_old_controls(self):
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
vm.runInContext(`latest={armed:true,frames:12,last_frame_age:.1};online=true;render();`,context);
assert.equal(elements.get('power').disabled,false);
windowListeners.pagehide();let requests=0;
context.fetch=()=>{requests++;return new Promise(()=>{});};
windowListeners.pageshow({persisted:false});assert.equal(requests,0);
windowListeners.pageshow({persisted:true});assert.ok(requests>0);assert.equal(elements.get('power').disabled,true);
assert.match(elements.get('mode-title').textContent,/offline/i);
""")

    def test_full_off_is_distinct_from_alarm_pause_and_is_accessible(self):
        page = PageParser(dashboard_ui.DASHBOARD_BODY)
        self.assertEqual(page.elements['full-off']['type'], 'button')
        self.assertEqual(page.elements['full-off']['aria-describedby'], 'full-off-description')
        self.assertEqual(page.elements['shutdown-notice']['role'], 'status')
        self.assertIn('C500 ON', dashboard_ui.DASHBOARD_BODY)
        self.assertIn('C500 OFF', dashboard_ui.DASHBOARD_BODY)
        self.assertIn('pause / resume alerts', dashboard_ui.DASHBOARD_BODY)
        self.assertNotIn('monitoring on / off', dashboard_ui.DASHBOARD_BODY)
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
vm.runInContext(`latest={armed:false,frames:12,last_frame_age:.2};online=true;render();`,context);
assert.equal(elements.get('power').textContent,'Arm alerts');
assert.equal(elements.get('full-off').disabled,false);
assert.match(elements.get('mode-title').textContent,/Running.*paused/);
assert.match(elements.get('control-description').textContent,/keyboard helper.*running/);
vm.runInContext(`latest.armed=true;render();`,context);
assert.equal(elements.get('power').textContent,'Pause alerts');
assert.equal(elements.get('full-off').disabled,false);
assert.equal(elements.get('mode-title').textContent,'Running · alerts armed');
vm.runInContext(`online=false;render();`,context);
assert.equal(elements.get('full-off').disabled,true);
""")

    def test_full_off_acceptance_stops_polling_and_stale_responses_cannot_restore_controls(self):
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
(async()=>{
 const requests=[];let resolveState;
 context.fetch=(path,options)=>{requests.push({path,options});
  if(path==='/state')return new Promise(resolve=>{resolveState=()=>resolve({ok:true,status:200,json:async()=>({armed:true,frames:25,last_frame_age:.1})});});
  return Promise.resolve({ok:true,status:202,json:async()=>({ok:true,status:'stopping'})});
 };
 vm.runInContext(`latest={armed:true,frames:12,last_frame_age:.2};online=true;render();window.stops={state:0,frames:0,starts:0};statePoller.stop=()=>window.stops.state++;statePoller.start=()=>window.stops.starts++;stopFrames=()=>window.stops.frames++;`,context);
 const staleTick=vm.runInContext('stateTick()',context);
 await elements.get('full-off').listeners.click();
 assert.equal(requests.length,1,'one click must only arm the confirmation, never stop the program');
 assert.equal(elements.get('full-off').textContent,'Confirm stop');
 await elements.get('full-off').listeners.click();
 assert.equal(requests[1].path,'/power/off');assert.equal(requests[1].options.method,'POST');
 assert.equal(requests[1].options.headers['X-Via'],'web');
 assert.equal(context.window.stops.state,1);assert.equal(context.window.stops.frames,1);
 assert.equal(elements.get('shutdown-notice').hidden,false);assert.equal(elements.get('shutdown-notice').focused,true);
 assert.equal(elements.get('mode-title').textContent,'Full shutdown requested');
 assert.match(elements.get('mode-detail').textContent,/completion is not verified/i);
 assert.doesNotMatch(elements.get('mode-detail').textContent,/fully off|zero processes/i);
 for(const id of ['power','ack','calibrate','add-lighting','full-off'])assert.equal(elements.get(id).disabled,true,id);
 assert.equal(elements.get('fresh-badge').textContent,'Preview stopped');
 resolveState();await staleTick;
 assert.equal(elements.get('mode-title').textContent,'Full shutdown requested');
 assert.match(elements.get('mode-detail').textContent,/completion is not verified/i);
 assert.doesNotMatch(elements.get('mode-detail').textContent,/fully off|zero processes/i);
 assert.equal(elements.get('full-off').disabled,true);
 await vm.runInContext("fullStop();stateTick();perform('/arm','wrong');refreshAlerts(99);resumeFrames();",context);
 windowListeners.pageshow({persisted:true});
 assert.equal(requests.length,2);assert.equal(context.window.stops.starts,0);
 assert.equal(elements.get('offline').hidden,true);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")

    def test_failed_or_unexpected_full_off_reply_does_not_claim_shutdown(self):
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
(async()=>{
 vm.runInContext(`latest={armed:true,frames:12,last_frame_age:.2};online=true;render();window.stops=0;statePoller.stop=()=>window.stops++;stopFrames=()=>window.stops++;`,context);
 for(const reply of [
  {ok:false,status:503,json:async()=>({ok:false,error:'Power task is unavailable.'})},
  {ok:true,status:200,json:async()=>({ok:true})}
 ]){
  context.fetch=async()=>reply;
  await vm.runInContext('fullStop()',context);
  assert.match(elements.get('action-message').textContent,/Shutdown was not confirmed/);
  assert.match(elements.get('action-message').textContent,/C500 OFF/);
  assert.equal(elements.get('mode-title').textContent,'Running · alerts armed');
  assert.equal(elements.get('full-off').disabled,false);
  assert.equal(elements.get('power').disabled,false);
  assert.equal(context.window.stops,0);
 }
 context.fetch=async()=>{throw Error('Connection closed');};
 await vm.runInContext('fullStop()',context);
 assert.match(elements.get('action-message').textContent,/Connection closed/);
 assert.equal(elements.get('full-off').disabled,false);assert.equal(context.window.stops,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")

    def test_quiet_controls_cannot_extend_pause_and_do_not_disarm(self):
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
vm.runInContext(`latest={armed:true,frames:12,last_frame_age:.2,alarm:null,ack_remaining_s:4.2,ack_quiet_s:5};online=true;render();`,context);
assert.equal(elements.get('power').disabled,false);assert.equal(elements.get('power').textContent,'Pause alerts');
assert.equal(elements.get('ack').disabled,true);assert.equal(elements.get('calibrate').disabled,true);
assert.equal(elements.get('ack-quiet').hidden,false);assert.equal(elements.get('ack-countdown').textContent,'5 s quiet');
assert.equal(elements.get('ack-progress').value,4.2);assert.equal(elements.get('ack-progress').max,5);
assert.equal(elements.get('mode-title').textContent,'Running · acknowledged');
vm.runInContext(`latest.ack_remaining_s=0;render();`,context);
assert.equal(elements.get('ack-quiet').hidden,true);assert.equal(elements.get('ack').disabled,true);
assert.equal(elements.get('ack').textContent,'Acknowledge alarm');
vm.runInContext(`latest.ack_remaining_s=5;online=false;render();`,context);
assert.equal(elements.get('ack-quiet').hidden,true);assert.equal(elements.get('ack').disabled,true);
""")

    def test_ack_is_bound_to_displayed_alarm_and_stale_reply_is_not_success(self):
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
(async()=>{
 const requests=[];let acknowledged=true;
 context.fetch=async(path,options)=>{requests.push({path,options});return {ok:true,status:200,json:async()=>({ok:true,acknowledged,ack_remaining_s:5})};};
 vm.runInContext(`latest={armed:true,alarm:'main',alarm_id:'episode-1'};online=true;render();`,context);
 await vm.runInContext('acknowledge()',context);
 assert.equal(requests.length,1);assert.equal(requests[0].path,'/ack');
 assert.equal(requests[0].options.method,'POST');assert.equal(requests[0].options.headers['Content-Type'],'application/json');
 assert.deepEqual(JSON.parse(requests[0].options.body),{alarm_id:'episode-1'});
 assert.equal(elements.get('action-message').textContent,'Alarm acknowledged.');
 vm.runInContext(`latest.alarm=null;latest.ack_remaining_s=5;render();`,context);
 await vm.runInContext('acknowledge()',context);assert.equal(requests.length,1);
 vm.runInContext(`latest.alarm='main';latest.alarm_id='episode-2';latest.ack_remaining_s=0;render();`,context);
 acknowledged=false;await vm.runInContext('acknowledge()',context);
 assert.equal(requests.length,2);assert.deepEqual(JSON.parse(requests[1].options.body),{alarm_id:'episode-2'});
 assert.match(elements.get('action-message').textContent,/already acknowledged or changed/);
 assert.notEqual(elements.get('action-message').textContent,'Alarm acknowledged.');
})().catch(error=>{console.error(error);process.exitCode=1;});
""")

    def test_health_warnings_and_alert_history_render_as_text(self):
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
vm.runInContext(`latest={armed:true,frames:5,last_frame_age:.1,alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1},loop_age_s:.2,input_ack_after_s:1,audio:{available:true,muted:true,volume:.8,device:'Speakers'},input_helper:{connected:true}};online=true;render();`,context);
assert.equal(elements.get('health-warning').hidden,false);assert.match(elements.get('health-warning').textContent,/muted/);
assert.equal(elements.get('health-speaker').textContent,'Muted');assert.equal(elements.get('input-ack-after').textContent,1);
vm.runInContext(`latest.audio.muted=false;latest.audio.volume=.15;render();`,context);assert.match(elements.get('health-warning').textContent,/15%/);
vm.runInContext(`latest.audio.volume=.9;render();`,context);assert.equal(elements.get('health-warning').hidden,true);
assert.equal(elements.get('health-badge').textContent,'Healthy');
vm.runInContext(`latest.input_helper={connected:false,age_s:null};render();`,context);assert.match(elements.get('health-warning').textContent,/Ctrl\+Alt\+F4/);
vm.runInContext(`latest.input_helper={connected:true};latest.loop_age_s=40;render();`,context);assert.match(elements.get('health-loop').textContent,/Stalled 40/);
vm.runInContext(`online=false;render();`,context);assert.equal(elements.get('health-warning').hidden,true);
vm.runInContext(`renderAlerts([{kind:'main',time:'19:04:22',date:'2026-09-11',reason:'<b>Gate</b>',snapshot:'20260911-190422-abcdef1-main.jpg',acked:{via:'input-mouse',after_s:1.4}},{kind:'fault',time:'19:05:00',reason:'Feed lost',snapshot:null,acked:null}]);`,context);
const rows=elements.get('alerts').children;assert.equal(rows.length,2);
const link=rows[0].children[0];assert.equal(link.href,'/alerts/20260911-190422-abcdef1-main.jpg');assert.equal(link.children[0].src,link.href);
assert.equal(rows[0].children[1].children[1].textContent,'<b>Gate</b>');
assert.match(rows[0].children[1].children[2].textContent,/mouse click after 1.4 s/);
assert.equal(rows[1].children[0].textContent,'No image');assert.match(rows[1].children[1].children[2].textContent,/Not acknowledged/);
vm.runInContext(`renderAlerts([]);`,context);assert.match(elements.get('alerts').children[0].textContent,/No alerts/);
""")

    def test_holding_alignment_is_healthy_and_lost_reference_is_named(self):
        self.node("const vm=require('node:vm'),assert=require('node:assert/strict');\n" +
                  "const context={window:{},document:{},setTimeout,clearTimeout,performance};\n" +
                  "vm.runInNewContext(" + json.dumps(dashboard_ui.CORE_SCRIPT) + ",context);\n" + r"""
const ui=context.window.C500UI;
assert.equal(ui.alignmentInfo({state:'holding',message:'Holding'}).healthy,true);
assert.equal(ui.alignmentInfo({state:'holding'}).label,'Holding');
assert.equal(ui.alignmentInfo({state:'calibration_required',calibrated:true}).label,'Alignment lost');
assert.equal(ui.alignmentInfo({state:'calibration_required',calibrated:false}).label,'Reference needed');
""")

    def test_dashboard_does_not_duplicate_native_input_acknowledgement(self):
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
assert.equal(documentListeners.keydown,undefined);
assert.equal(documentListeners.pointerdown,undefined);
""")

    def editor_fixture(self, zones=None):
        config = self.config()
        if zones is not None:
            config['zones'] = zones
        return r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const drawing=[];
class Element{
 constructor(){this.children=[];this.style={};this.dataset={};this.hidden=false;this.disabled=false;this.textContent='';this.checked=false;this._value='';this.listeners={};this.classList={toggle(){},remove(){},add(){}};}
 set value(value){this._value=String(value);}get value(){return this._value;}
 focus(){this.focused=true;}append(...items){this.children.push(...items);}replaceChildren(){this.children=[];}setAttribute(){}addEventListener(name,callback){this.listeners[name]=callback;}querySelector(){return new Element();}
 getBoundingClientRect(){return {left:0,top:0,width:640,height:360};}focus(){}setPointerCapture(){}
 getContext(){return new Proxy({}, {get:(object,key)=>(...args)=>{drawing.push({key,args});},set:()=>true});}
 set innerHTML(value){throw Error('Unsafe HTML rendering used');}
}
const elements=new Map(),windowListeners={};
const context={window:{devicePixelRatio:1,addEventListener(name,callback){windowListeners[name]=callback;}},document:{hidden:false,addEventListener(){},getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(){return new Element();}},setTimeout(){return 1;},clearTimeout(){},performance,AbortController,fetch:()=>new Promise(()=>{})};
vm.createContext(context);
""" + ('context.document.getElementById("initial-config").textContent=' +
       json.dumps(json.dumps(config)) + ';\n' +
       'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.EDITOR_SCRIPT) + ',context);\n')

    def test_custom_outline_drag_and_cancel_preserve_geometry_and_ignored_parking(self):
        self.node(self.editor_fixture([{
            'name': 'Inbound', 'rect': [.2, .2, .4, .4],
            'polygon': [[.2, .2], [.6, .2], [.6, .6], [.2, .6]],
            'exclude_polygons': [[[.7, .7], [.8, .7], [.8, .8], [.7, .8]]],
            'classes': ['car', 'truck'], 'trigger': 'ai'}]) + r"""
vm.runInContext('frameReady=true',context);
const pointer=elements.get('overlay').listeners;
const parking=vm.runInContext('JSON.stringify(zones[0].exclude_polygons)',context);
pointer.pointerdown({button:0,pointerId:1,clientX:128,clientY:72,preventDefault(){}});
pointer.pointermove({clientX:160,clientY:100});
assert.equal(vm.runInContext('zones[0].polygon[0][0]',context),.25);
pointer.pointercancel({type:'pointercancel'});
assert.equal(vm.runInContext('zones[0].polygon[0][0]',context),.2);
pointer.pointerdown({button:0,pointerId:1,clientX:128,clientY:72,preventDefault(){}});
pointer.pointermove({clientX:160,clientY:100});pointer.pointerup({type:'pointerup'});
assert.equal(vm.runInContext('zones[0].polygon[0][0]',context),.25);
assert.equal(vm.runInContext('JSON.stringify(zones[0].exclude_polygons)',context),parking);
assert.equal(elements.get('exclusion-note').hidden,false);
assert.ok(drawing.some(item=>item.key==='setLineDash'));
// Dragging a corner through the opposite side must retain the last valid outline.
pointer.pointerdown({button:0,pointerId:1,clientX:160,clientY:100,preventDefault(){}});
const valid=vm.runInContext('JSON.stringify(zones[0].polygon)',context);
pointer.pointermove({clientX:512,clientY:144});pointer.pointerup({type:'pointerup'});
assert.equal(vm.runInContext('JSON.stringify(zones[0].polygon)',context),valid);
""")

    def test_polygon_resize_keyboard_move_and_corner_edit_stay_in_sync(self):
        self.node(self.editor_fixture() + r"""
elements.get('shape').value='polygon';elements.get('shape').listeners.change();
assert.equal(vm.runInContext('zones[0].polygon.length',context),4);
elements.get('x').value='.3';elements.get('x').listeners.change();
assert.equal(vm.runInContext('zones[0].polygon[0][0]',context),.3);
elements.get('w').value='.4';elements.get('w').listeners.change();
assert.equal(vm.runInContext('zones[0].polygon[1][0]',context),.7);
elements.get('overlay').listeners.keydown({key:'ArrowRight',shiftKey:true,preventDefault(){}});
assert.ok(Math.abs(vm.runInContext('zones[0].polygon[0][0]',context)-.31)<1e-9);
elements.get('add-corner').listeners.click();
assert.equal(vm.runInContext('zones[0].polygon.length',context),5);
assert.equal(elements.get('vertex-label').textContent,'Corner 2 of 5');
elements.get('remove-corner').listeners.click();
assert.equal(vm.runInContext('zones[0].polygon.length',context),4);
const before=vm.runInContext('JSON.stringify(zones[0].polygon)',context);
elements.get('vertex-x').value='1.5';elements.get('vertex-x').listeners.change();
assert.equal(vm.runInContext('JSON.stringify(zones[0].polygon)',context),before);
assert.match(elements.get('message').textContent,/inside the image/);
""")

    def test_class_filters_do_not_expand_on_unrelated_edits_and_require_a_choice(self):
        self.node(self.editor_fixture([{'name': 'Inbound', 'rect': [.2, .2, .4, .4],
                                       'classes': ['car'], 'trigger': 'ai'}]) + r"""
elements.get('name').value='Entry lane';elements.get('name').listeners.change();
assert.equal(vm.runInContext('JSON.stringify(zones[0].classes)',context),'["car"]');
elements.get('class-person').checked=true;elements.get('class-person').listeners.change();
assert.equal(vm.runInContext('JSON.stringify(zones[0].classes)',context),'["person","car"]');
elements.get('class-vehicle').checked=false;elements.get('class-vehicle').listeners.change();
assert.equal(vm.runInContext('JSON.stringify(zones[0].classes)',context),'["person"]');
elements.get('class-person').checked=false;elements.get('class-person').listeners.change();
assert.match(elements.get('message').textContent,/at least one/);
assert.equal(vm.runInContext('JSON.stringify(zones[0].classes)',context),'["person"]');
elements.get('trigger').value='motion';elements.get('trigger').listeners.change();
assert.equal(vm.runInContext('zones[0].trigger',context),'motion');
assert.equal(elements.get('class-settings').hidden,true);
assert.equal(elements.get('motion-settings').hidden,false);
""")

    def test_editor_save_transmits_polygon_filters_and_parking_without_replacing_metadata(self):
        self.node(self.editor_fixture([{'name': 'Pedestrian path', 'rect': [.2, .2, .4, .4],
                                       'polygon': [[.2, .2], [.6, .2], [.6, .6], [.2, .6]],
                                       'classes': ['person'], 'trigger': 'ai',
                                       'exclude_polygons': [[[.5, .4], [.6, .4], [.6, .5], [.5, .5]]],
                                       'on_frames': 4, 'off_frames': 12}]) + r"""
(async()=>{
 const requests=[];context.fetch=async(path,options)=>{requests.push({path,options});return {ok:true,status:200,json:async()=>({ok:true})};};
 vm.runInContext('online=true;armed=false;updateControls()',context);
 await elements.get('zone-form').listeners.submit({preventDefault(){}});
 const request=requests.find(r=>r.path==='/zones');assert.ok(request);
 const zone=JSON.parse(request.options.body).zones[0];
 assert.equal(zone.trigger,'ai');assert.deepEqual(zone.classes,['person']);assert.equal(zone.polygon.length,4);
 assert.equal(zone.exclude_polygons.length,1);assert.equal(zone.on_frames,4);assert.equal(zone.off_frames,12);
 assert.match(elements.get('message').textContent,/Alerts remain paused; the program stays running/);
 assert.equal(vm.runInContext('armed',context),false);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")

    def test_ai_zone_rows_display_class_filters_instead_of_motion_percentage(self):
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`latest={frames:10,last_frame_age:.1,alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1}};online=true;renderZones([{name:'Pedestrian path',trigger:'ai',classes:['person'],changed_pct:99,ai_active:true}],true)`,context);
const row=elements.get('zones').children[0];
assert.equal(row.children[1].textContent,'People');assert.equal(row.children[1].children[0].textContent,'AI objects only');
assert.equal(row.children[0].children[1].textContent,'Present');
vm.runInContext(`renderZones([{name:'Inbound',trigger:'ai',classes:['car','truck'],light_raises:'early'}],false)`,context);
const off=elements.get('zones').children[0];
assert.equal(off.children[1].textContent,'Vehicles');assert.equal(off.children[1].children[0].textContent,'AI objects only');
assert.equal(off.children[0].children[1].textContent,'Alerts paused');
""")

    def test_scene_lighting_diagnostic_reports_visibility_without_claiming_day_or_night(self):
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`latest={armed:false,frames:5,last_frame_age:.1,lighting:{mode:'dim',state:'near_black',message:'View is too dark'}};online=true;render()`,context);
assert.equal(elements.get('scene-light-badge').textContent,'Dim view');
assert.equal(elements.get('scene-light-state').textContent,'View too dark');
assert.equal(elements.get('scene-light-description').textContent,'View is too dark');
vm.runInContext(`latest.lighting={mode:'bright',state:'glare',message:'Strong glare'};render()`,context);
assert.equal(elements.get('scene-light-state').textContent,'Glare detected');
vm.runInContext(`latest.lighting={mode:'normal',state:'adjusting',message:'Brightness settling'};render()`,context);
assert.equal(elements.get('scene-light-state').textContent,'Adjusting');
vm.runInContext(`latest.lighting.state='stable';render()`,context);
assert.equal(elements.get('scene-light-state').textContent,'Steady');
vm.runInContext(`online=false;render()`,context);
assert.equal(elements.get('scene-light-badge').textContent,'Offline');
assert.equal(elements.get('scene-light-state').textContent,'Status unavailable');
""")

    def test_priority_routes_and_history_are_independent_of_alarm_controls(self):
        page = PageParser(dashboard_ui.DASHBOARD_BODY)
        # What happened tonight is the operator's first question, so the alert
        # history is open; the deeper diagnostics stay out of the way.
        self.assertIn('open', page.elements['history-disclosure'])
        for section in ('diagnostics', 'zone-disclosure'):
            self.assertNotIn('open', page.elements[section])
        self.assertIn('priority-routes', page.ids)
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`latest={armed:true,frames:10,last_frame_age:.1,alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1},zones:[{name:'Lower-left arrival',priority:true,route_group:'left-arrival',trigger:'ai_light',classes:['car'],light_delta:8},{name:'Upper curved arrival',priority:true,route_group:'right-arrival',trigger:'ai',classes:['car'],arrival_direction:[-1,1]},{name:'Other lane',trigger:'ai',classes:['car']}]};online=true;render();`,context);
const cards=elements.get('priority-routes').children;assert.equal(cards.length,2);
assert.equal(cards[0].children[1].textContent,'Lower-left arrival');assert.match(cards[0].children[0].children[0].textContent,/VEHICLE \+ LIGHT/);
assert.equal(cards[1].children[1].textContent,'Upper curved arrival');assert.equal(cards[1].children[3].children[0].textContent,'Direction-aware entry');
assert.equal(cards[0].children[0].children[1].textContent,'Watching');
assert.equal(elements.get('zone-count').textContent,'3 saved zones');
vm.runInContext(`latest.zones[0].light_active=true;latest.alarm='main';latest.alarm_id='light-one';latest.reason='Lower-left arrival: local light +12';render();`,context);
assert.equal(elements.get('priority-routes').children[0].children[0].children[1].textContent,'Light detected');
assert.equal(elements.get('event-title').textContent,'Local light change');assert.equal(elements.get('ack').disabled,false);
vm.runInContext(`latest.armed=false;render();`,context);
assert.equal(elements.get('event-title').textContent,'Buzzer is off');assert.equal(elements.get('ack').disabled,true);
assert.equal(elements.get('priority-routes').children[0].children[0].children[1].textContent,'Alerts paused');
vm.runInContext(`online=false;render();`,context);
assert.equal(elements.get('priority-routes').children[0].children[0].children[1].textContent,'Offline');
assert.equal(elements.get('event-title').textContent,'Connection unavailable');
""")

    def test_route_status_does_not_present_stale_detection_as_live(self):
        self.node("const vm=require('node:vm'),assert=require('node:assert/strict');\n" +
                  "const context={window:{},document:{},setTimeout,clearTimeout,performance};\n" +
                  "vm.runInNewContext(" + json.dumps(dashboard_ui.CORE_SCRIPT) + ",context);\n" + r"""
const ui=context.window.C500UI,zone={trigger:'ai',ai_active:true},state={armed:true,frames:10,last_frame_age:.1,alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1}};
assert.equal(ui.routeStatus(zone,state).label,'Vehicle present');
assert.equal(ui.routeStatus(zone,{...state,ack_remaining_s:3}).label,'Present · quiet');
assert.equal(ui.routeStatus(zone,{...state,last_frame_age:9}).label,'View unavailable');
assert.equal(ui.routeStatus(zone,{...state,alignment:{state:'calibration_required'}}).label,'Alignment paused');
assert.equal(ui.routeStatus(zone,{...state,ml:{available:false,state:'failed'}}).label,'Detection paused');
assert.equal(ui.routeStatus(zone,state,false).label,'Offline');
assert.equal(ui.routeStatus({...zone,trigger:'ai_light'},{...state,ml:{available:false}}).label,'AI waiting');
assert.equal(ui.alertSource({armed:true,alarm:'main',alarm_source:'light',reason:'Arrival'}).kind,'light');
assert.equal(ui.alertSource({armed:true,alarm:'main',reason:'Car at the right gate'}).kind,'ai');
assert.equal(ui.alertSource({armed:true,alarm:'fault',reason:'Camera lost'}).kind,'fault');
assert.equal(ui.alertSource({armed:false,alarm:'main',reason:'Car'}),null);
""")

    def test_ai_light_editor_saves_light_and_ai_filters_without_raw_motion(self):
        self.node(self.editor_fixture([{'name': 'Arrival', 'rect': [.2, .2, .4, .4],
                                       'trigger': 'ai', 'classes': ['car'], 'motion': True,
                                       'light': False, 'priority': True, 'route_group': 'left-arrival',
                                       'arrival_direction': [1, -1], 'arrival_min_progress': .012,
                                       'stationary_confirm_s': 3}]) + r"""
elements.get('trigger').value='ai_light';elements.get('trigger').listeners.change();
assert.equal(elements.get('motion-settings').hidden,true);assert.equal(elements.get('light-settings').hidden,false);
assert.equal(elements.get('light').checked,true);assert.equal(elements.get('light').disabled,true);
assert.equal(elements.get('light_delta').disabled,false);assert.equal(elements.get('class-settings').hidden,false);
assert.equal(vm.runInContext('zones[0].motion',context),false);assert.equal(vm.runInContext('zones[0].light',context),true);
elements.get('ai_confidence').value=70;elements.get('ai_confidence').listeners.change();
elements.get('ai_confirm_observations').value=3;elements.get('ai_confirm_observations').listeners.change();
assert.equal(vm.runInContext('zones[0].ai_confidence',context),.7);assert.equal(vm.runInContext('zones[0].ai_confirm_observations',context),3);
assert.equal(vm.runInContext('zones[0].priority',context),true);assert.equal(vm.runInContext('zones[0].route_group',context),'left-arrival');
assert.equal(vm.runInContext('JSON.stringify(zones[0].arrival_direction)',context),'[1,-1]');
assert.equal(vm.runInContext('zones[0].arrival_min_progress',context),.012);assert.equal(vm.runInContext('zones[0].stationary_confirm_s',context),3);
elements.get('ai_confirm_observations').value=1;elements.get('ai_confirm_observations').listeners.change();
assert.match(elements.get('message').textContent,/2 to 8/);assert.equal(vm.runInContext('zones[0].ai_confirm_observations',context),3);
""")

    def test_missing_health_measurements_are_not_reported_healthy(self):
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`latest={armed:false,frames:5,last_frame_age:.1,alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1},loop_age_s:null,audio:{available:false},input_helper:{connected:true}};online=true;render();`,context);
assert.equal(elements.get('health-badge').textContent,'Checks incomplete');assert.equal(elements.get('health-loop').textContent,'—');
vm.runInContext(`latest.loop_age_s=.2;latest.audio={available:true,muted:false,volume:.9};render();`,context);
assert.equal(elements.get('health-badge').textContent,'Healthy');
""")

    def test_shared_entrance_road_shows_tracking_without_claiming_an_alarm(self):
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`latest={armed:true,frames:10,last_frame_age:.1,alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1,detections:[{label:'car',region_status:{Entrance:'tracking approach'}}]},zones:[{name:'Entrance',route_role:'entrance',priority:true,trigger:'ai',classes:['car'],confirmation_polygon:[[.2,.2],[.4,.2],[.4,.4]],operator_hint:'Check the entrance waiting lane.'}]};online=true;render();`,context);
let card=elements.get('priority-routes').children[0];
assert.equal(card.children[0].children[0].textContent,'CAMPUS ENTRANCE');
assert.equal(card.children[0].children[1].textContent,'Tracking approach');
assert.match(card.children[2].textContent,/No buzzer from this zone yet/);
assert.match(card.children[3].children[0].textContent,/confirmation area/);
assert.match(card.children[4].textContent,/Check the entrance waiting lane/);
assert.equal(elements.get('ack').disabled,true);assert.equal(elements.get('event-action').hidden,true);
vm.runInContext(`latest.ml.detections[0].region_status.Entrance='moving away';render();`,context);
assert.equal(elements.get('priority-routes').children[0].children[0].children[1].textContent,'Moving away · ignored');
// A second arriving vehicle must be visible even while another is being ignored.
vm.runInContext(`latest.ml.detections.push({region_status:{Entrance:'confirmed'}});render();`,context);
assert.equal(elements.get('priority-routes').children[0].children[0].children[1].textContent,'Arrival confirmed');
vm.runInContext(`latest.last_frame_age=10;render();`,context);
assert.equal(elements.get('priority-routes').children[0].children[0].children[1].textContent,'View unavailable');
""")

    def test_current_event_uses_role_and_evidence_without_treating_light_as_a_car(self):
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`latest={armed:true,alarm:'main',alarm_id:'event1',alarm_source:'light',reason:'Exit road brightness changed',alarm_evidence:{source:'light',zone_name:'Exit road',route_role:'exit',measurement:{positive_delta:4.7}},zones:[{name:'Exit road',route_role:'exit',operator_hint:'Use the exit barrier control.'}]};online=true;render();`,context);
assert.equal(elements.get('event-title').textContent,'Campus exit · local light change');
assert.match(elements.get('event-evidence').textContent,/\+4.7 brightness/);
assert.match(elements.get('event-evidence').textContent,/identity not confirmed/);
assert.match(elements.get('event-action').textContent,/Check the exit road/);
assert.doesNotMatch(elements.get('event-action').textContent,/Use the exit barrier/);
vm.runInContext(`latest.alarm_source='ai';latest.alarm_evidence={source:'ai',zone_name:'Entrance',route_role:'entrance',label:'car',confidence:.79,observations:3,direction_state:'toward',confirmation_required:true,operator_hint:'Check the entrance waiting lane.'};render();`,context);
assert.equal(elements.get('event-title').textContent,'Campus entrance · ai object detection');
assert.match(elements.get('event-evidence').textContent,/79% model confidence.*3 observations.*audible confirmation area/);
assert.equal(elements.get('event-action').textContent,'Check the entrance waiting lane.');
vm.runInContext(`latest.alarm=null;latest.alarm_id=null;latest.ack_remaining_s=5;render();`,context);
assert.equal(elements.get('event-action').hidden,true);assert.equal(elements.get('event-evidence').hidden,true);
assert.equal(elements.get('ack').disabled,true);assert.equal(elements.get('power').textContent,'Pause alerts');
""")

    def test_editor_preserves_nested_confirmation_area_and_site_metadata_on_save(self):
        self.node(self.editor_fixture([{'name': 'Entrance road', 'rect': [.2, .2, .4, .4],
                                       'polygon': [[.2, .2], [.6, .2], [.6, .6], [.2, .6]],
                                       'confirmation_polygon': [[.3, .3], [.5, .3], [.5, .5], [.3, .5]],
                                       'route_role': 'entrance', 'route_group': 'entrance',
                                       'operator_hint': 'Check the entrance waiting lane.',
                                       'trigger': 'ai', 'classes': ['car']}]) + r"""
(async()=>{
 const requests=[];context.fetch=async(path,options)=>{requests.push({path,options});return {ok:true,status:200,json:async()=>({ok:true})};};
 assert.equal(elements.get('confirmation-note').hidden,false);
 assert.match(elements.get('route-setting').textContent,/Campus entrance.*Check the entrance/);
 assert.ok(drawing.some(item=>item.key==='fillText'&&item.args[0]==='WAIT ALERT'));
 const gate=vm.runInContext('JSON.stringify(zones[0].confirmation_polygon)',context);
 elements.get('x').value='.21';elements.get('x').listeners.change();
 assert.equal(vm.runInContext('JSON.stringify(zones[0].confirmation_polygon)',context),gate);
 vm.runInContext('online=true;armed=false;updateControls()',context);
 await elements.get('zone-form').listeners.submit({preventDefault(){}});
 const zone=JSON.parse(requests.find(r=>r.path==='/zones').options.body).zones[0];
 assert.equal(JSON.stringify(zone.confirmation_polygon),gate);
 assert.equal(zone.route_role,'entrance');assert.equal(zone.route_group,'entrance');
 assert.equal(zone.operator_hint,'Check the entrance waiting lane.');
 assert.equal(zone.rect[0],.21);assert.equal(vm.runInContext('armed',context),false);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")

    def test_history_retains_detection_evidence_without_live_action_instructions(self):
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`renderAlerts([{kind:'main',reason:'Entrance car',evidence:{source:'ai',label:'car',confidence:.8,observations:3,confirmation_required:true,operator_hint:'Operate the barrier.'},acked:{via:'input-any-key',after_s:2}}])`,context);
const body=elements.get('alerts').children[0].children[1];
assert.match(body.children[2].textContent,/80% model confidence/);
assert.doesNotMatch(body.children[2].textContent,/Operate the barrier/);
assert.match(body.children[3].textContent,/Silenced by a key press/);
""")

    def test_top_road_advance_warning_does_not_claim_barrier_commitment(self):
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`latest={armed:true,frames:10,last_frame_age:.1,alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1,detections:[]},zones:[{name:'Top road arrival',route_role:'entrance',alert_stage:'advance',priority:true,trigger:'ai',classes:['car'],confirmation_polygon:[[.2,.2],[.4,.2],[.4,.4]],operator_hint:'Check approaching traffic at the top road.'}]};online=true;render();`,context);
let card=elements.get('priority-routes').children[0];
assert.match(card.children[2].textContent,/before their turn decision/);
assert.equal(card.children[3].children[0].textContent,'Advance warning · top road');
assert.match(card.children[4].textContent,/Check approaching traffic.*later choose parking or charging/);
assert.match(elements.get('entrance-guide').textContent,/advance vehicle or local light warning/);
assert.match(elements.get('ai-description').textContent,/Top-road vehicle or local light alerts/);
vm.runInContext(`latest.ml.detections=[{region_status:{'Top road arrival':'confirmed'}}];latest.alarm='main';latest.alarm_id='top1';latest.alarm_source='ai';latest.alarm_evidence={zone_name:'Top road arrival',source:'ai',route_role:'entrance',alert_stage:'advance',confirmation_required:true};render();`,context);
assert.equal(elements.get('priority-routes').children[0].children[0].children[1].textContent,'Top-road arrival');
assert.match(elements.get('event-title').textContent,/Top-road warning/);
assert.match(elements.get('event-evidence').textContent,/destination not determined/);
assert.equal(elements.get('event-action').textContent,'Check approaching traffic at the top road.');
assert.equal(elements.get('ack').disabled,false);
vm.runInContext(`latest.alarm=null;latest.alarm_id=null;latest.ack_remaining_s=5;latest.ml.detections[0].region_status['Top road arrival']='already alerted';render();`,context);
assert.equal(elements.get('priority-routes').children[0].children[0].children[1].textContent,'Visit already alerted');
assert.equal(elements.get('event-action').hidden,true);
""")

    def test_advance_stage_and_confirmation_area_survive_editor_save(self):
        self.node(self.editor_fixture([{'name': 'Top road arrival', 'rect': [.2, .2, .4, .4],
                                       'confirmation_polygon': [[.3, .3], [.5, .3], [.5, .5], [.3, .5]],
                                       'route_role': 'entrance', 'alert_stage': 'advance',
                                       'route_group': 'entrance', 'priority': True,
                                       'operator_hint': 'Check the top road.',
                                       'light_polygon': [[.3, .3], [.5, .3], [.5, .5], [.3, .5]],
                                       'trigger': 'ai_light', 'classes': ['car']},
                                      {'name': 'Entrance waiting', 'rect': [.2, .2, .4, .4],
                                       'confirmation_polygon': [[.3, .3], [.5, .3], [.5, .5], [.3, .5]],
                                       'route_role': 'entrance', 'trigger': 'ai', 'classes': ['car']}]) + r"""
(async()=>{
 const requests=[];context.fetch=async(path,options)=>{requests.push({path,options});return {ok:true,status:200,json:async()=>({ok:true})};};
 drawing.length=0;vm.runInContext('draw()',context);
 const labels=drawing.filter(item=>item.key==='fillText');
 assert.equal(labels.filter(item=>item.args[0]==='TOP ALERT').length,1);
 assert.equal(labels.filter(item=>item.args[0]==='WAIT ALERT').length,0);
 const routeLabels=labels.filter(item=>['Top road arrival','Entrance waiting'].includes(item.args[0]));
 assert.equal(routeLabels.length,2);assert.ok(Math.abs(routeLabels[0].args[2]-routeLabels[1].args[2])>=19);
 assert.match(elements.get('confirmation-note').textContent,/advance warning area on the top road.*later turn to charging/);
 assert.match(elements.get('confirmation-note').textContent,/light backup also watches the saved gold band/);
 assert.match(elements.get('route-setting').textContent,/before the parking or entrance turn decision/);
 elements.get('name').value='Upper road arrival';elements.get('name').listeners.change();
 vm.runInContext('online=true;armed=false;updateControls()',context);
 await elements.get('zone-form').listeners.submit({preventDefault(){}});
 const zone=JSON.parse(requests.find(r=>r.path==='/zones').options.body).zones[0];
 assert.equal(zone.alert_stage,'advance');assert.equal(zone.route_role,'entrance');
 assert.equal(zone.priority,true);assert.equal(zone.confirmation_polygon.length,4);
 assert.deepEqual(zone.light_polygon,zone.confirmation_polygon);
 assert.equal(zone.operator_hint,'Check the top road.');assert.equal(zone.name,'Upper road arrival');
})().catch(error=>{console.error(error);process.exitCode=1;});
""")

    def test_top_road_light_backup_names_the_road_without_claiming_a_car(self):
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`latest={armed:true,frames:10,last_frame_age:.1,alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1,detections:[]},zones:[{name:'Top road arrival',route_role:'entrance',alert_stage:'advance',priority:true,trigger:'ai_light',classes:['car'],light_polygon:[[.2,.2],[.4,.2],[.4,.4]]}]};online=true;render();`,context);
assert.match(elements.get('priority-routes').children[0].children[2].textContent,/AI vehicle or local brightness change in the gold top-road band/);
vm.runInContext(`latest.zones[0].light_active=true;latest.alarm='main';latest.alarm_id='toplight1';latest.alarm_source='light';latest.alarm_evidence={source:'light',zone_name:'Top road arrival',route_role:'entrance',alert_stage:'advance',measurement:{positive_delta:5}};render();`,context);
assert.match(elements.get('priority-routes').children[0].children[2].textContent,/top-road alert band.*vehicle is not confirmed/);
assert.equal(elements.get('event-title').textContent,'Top-road warning · local light change');
assert.match(elements.get('event-action').textContent,/Check the top road.*does not confirm a car/);
assert.match(elements.get('event-evidence').textContent,/vehicle identity not confirmed/);
""")

    def test_noise_map_explains_the_light_filter_without_claiming_object_attention(self):
        page = PageParser(dashboard_ui.DASHBOARD_BODY)
        self.assertEqual(page.elements['view-ignored']['aria-describedby'], 'noise-map-description')
        self.assertIn('Light-change noise map', dashboard_ui.DASHBOARD_BODY)
        self.assertIn('filters only the light-change channel', dashboard_ui.DASHBOARD_BODY)
        self.assertIn('does not show YOLO attention or ignored objects', dashboard_ui.DASHBOARD_BODY)
        self.assertNotIn('movement the AI ignores', dashboard_ui.DASHBOARD_SCRIPT)
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
const captions=[],canvas=context.document.getElementById('ai-canvas');
canvas.getBoundingClientRect=()=>({width:640,height:360});
canvas.getContext=()=>({clearRect(){},fillRect(){},fillText(text){captions.push(text);}});
vm.runInContext(`latest={armed:false,frames:10,last_frame_age:.1,alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1},noise_grid:[[1,5]]};online=true;render();setView('ignored');`,context);
assert.match(captions.at(-1),/Light-change noise map.*estimated flicker on 50%/);
assert.doesNotMatch(captions.at(-1),/AI ignores|wind|rain|ignored objects/);
assert.match(elements.get('decisions').children.map(row=>row.textContent).join(' '),/light channel only/);
vm.runInContext(`latest.alignment.state='lost';render();`,context);
assert.match(captions.at(-1),/Detection unavailable.*alignment/);
""")

    def test_paused_dashboard_keeps_stale_ai_and_alignment_failure_visible(self):
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`latest={armed:false,frames:10,last_frame_age:.1,alignment:{state:'tracking'},loop_age_s:.1,audio:{available:true,muted:false,volume:.8},input_helper:{connected:true},ml:{available:true,state:'ready',age_s:.1,detections:[{label:'car',confidence:.9,bbox:[1,2,30,40]}]},zones:[{name:'Gate',trigger:'ai_light',priority:true,ai_objects:1,ai_active:true,light_threshold:12,light_delta:2}]};online=true;render();`,context);
assert.equal(elements.get('ai-badge').textContent,'Ready');
assert.equal(elements.get('detections').hidden,false);
const boxes=[],canvas=context.document.getElementById('ai-canvas');
canvas.getBoundingClientRect=()=>({width:640,height:360});
canvas.getContext=()=>({clearRect(){},fillRect(){},fillText(){},measureText(){return {width:40};},strokeRect(...args){boxes.push(args);}});
vm.runInContext("setView('ai')",context);
assert.equal(boxes.length,1,'current object is drawn');
for(const update of ["latest.ml.age_s=3", "latest.ml.age_s=99", "latest.ml.age_s=null", "delete latest.ml.age_s", "latest.ml.age_s=.1;latest.alignment.state='calibration_required'"]){
 boxes.length=0;
 vm.runInContext(update+';render();',context);
 assert.equal(boxes.length,0,'stale or unaligned objects must not be drawn');
 assert.equal(elements.get('mode-title').textContent,'Running · alerts paused');
 assert.match(elements.get('mode-detail').textContent,/detection unavailable/i);
 assert.equal(elements.get('ai-badge').textContent,'Detection unavailable');
 assert.equal(elements.get('hero-ai').textContent,'Detection unavailable');
 assert.equal(elements.get('overlay-ai').textContent,'AI detection unavailable');
 assert.equal(elements.get('detections').hidden,true);
 assert.equal(elements.get('health-warning').hidden,false);
 assert.match(elements.get('health-warning').textContent,/detection unavailable/i);
 const decisions=elements.get('decisions').children.map(row=>row.textContent).join(' ');
 assert.doesNotMatch(decisions,/seen → confirmed/);
}
assert.match(elements.get('alignment-description').textContent,/Detection unavailable/);
assert.match(elements.get('zone-meters').children[0].textContent,/Detection unavailable.*alignment/);
assert.equal(elements.get('priority-routes').children[0].children[0].children[1].textContent,'Alignment paused');
vm.runInContext(`latest.alignment.state='tracking';latest.last_frame_age=9;render();`,context);
assert.match(elements.get('mode-detail').textContent,/fresh camera frames/);
assert.equal(elements.get('overlay-ai').textContent,'AI detection unavailable');
""")

    def test_ai_only_hides_raw_motion_and_light_but_preserves_them_for_switching_back(self):
        self.node(self.editor_fixture([{'name': 'Entry', 'rect': [.2, .2, .4, .4],
                                       'trigger': 'hybrid', 'classes': ['car'],
                                       'light': True, 'light_delta': 18, 'motion': True}]) + r"""
assert.equal(elements.get('light').disabled,false);
elements.get('trigger').value='ai';elements.get('trigger').listeners.change();
assert.equal(elements.get('motion-settings').hidden,true);assert.equal(elements.get('light-settings').hidden,true);
assert.equal(elements.get('light').disabled,true);assert.equal(elements.get('light_delta').disabled,true);
assert.equal(vm.runInContext('zones[0].light',context),true);assert.equal(vm.runInContext('zones[0].light_delta',context),18);
elements.get('trigger').value='hybrid';elements.get('trigger').listeners.change();
assert.equal(elements.get('light-settings').hidden,false);assert.equal(elements.get('light').disabled,false);
assert.equal(elements.get('light').checked,true);assert.equal(elements.get('light_delta').value,'18');
""")


if __name__ == "__main__":
    unittest.main()
