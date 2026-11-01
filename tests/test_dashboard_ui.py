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

NODE = shutil.which("node") or "node"


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
let state={armed:true,alarm:null,acked:true,ack_remaining_s:4.8,ack_quiet_s:5};
assert.equal(ui.deriveStatus(state).title,'ON · acknowledged');
assert.equal(ui.deriveStatus(state).mode,'on');
assert.match(ui.deriveStatus(state).detail,/5 s.*Detection continues/);
state.ack_remaining_s=2.1;assert.match(ui.deriveStatus(state).detail,/3 s/);
state.fault_reason='Camera disconnected';assert.match(ui.deriveStatus(state).detail,/Camera disconnected/);
state.ack_remaining_s=0;assert.match(ui.deriveStatus(state).title,/Fault/);
state.fault_reason=null;assert.equal(ui.deriveStatus(state).title,'ON · monitoring');
state.ack_remaining_s=4;state.armed=false;assert.match(ui.deriveStatus(state).title,/OFF/);
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
 append(...items){this.children.push(...items);}replaceChildren(){this.children=[];}setAttribute(){}addEventListener(name,callback){this.listeners[name]=callback;}querySelector(){return new Element();}
 set innerHTML(value){throw Error('Unsafe HTML rendering used');}
}
const elements=new Map(),documentListeners={},windowListeners={};
const context={window:{addEventListener(name,callback){windowListeners[name]=callback;}},document:{hidden:false,addEventListener(name,callback){documentListeners[name]=callback;},getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(){return new Element();},createTextNode(text){return {textContent:text,nodeType:3};}},setTimeout(){return 1;},clearTimeout(){},performance,AbortController,fetch:()=>new Promise(()=>{})};
vm.createContext(context);
"""

    def test_dashboard_controls_fail_closed_and_names_use_text_nodes(self):
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
vm.runInContext(`latest={armed:false,frames:12,last_frame_age:.2,zones:[{name:'<img onerror=alert(1)>',changed_pct:1}],ml:{available:true,state:'ready'}};online=true;render();`,context);
assert.equal(elements.get('power').disabled,false);assert.equal(elements.get('ack').disabled,true);assert.equal(elements.get('calibrate').disabled,false);
assert.match(elements.get('mode-title').textContent,/OFF/);
const title=elements.get('zones').children[0].children[0].children[0];
assert.equal(title.children[1].nodeType,3);assert.equal(title.children[1].textContent,'<img onerror=alert(1)>');
assert.equal(elements.get('add-lighting').disabled,false);
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

    def test_quiet_controls_cannot_extend_pause_and_do_not_disarm(self):
        self.node(self.dashboard_fixture() +
                  "vm.runInContext(" + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ",context);\n" + r"""
vm.runInContext(`latest={armed:true,frames:12,last_frame_age:.2,alarm:null,ack_remaining_s:4.2,ack_quiet_s:5};online=true;render();`,context);
assert.equal(elements.get('power').disabled,false);assert.match(elements.get('power').textContent,/off/);
assert.equal(elements.get('ack').disabled,true);assert.equal(elements.get('calibrate').disabled,true);
assert.equal(elements.get('ack-quiet').hidden,false);assert.equal(elements.get('ack-countdown').textContent,'5 s quiet');
assert.equal(elements.get('ack-progress').value,4.2);assert.equal(elements.get('ack-progress').max,5);
assert.equal(elements.get('mode-title').textContent,'ON · acknowledged');
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
vm.runInContext(`latest={armed:true,frames:5,last_frame_age:.1,loop_age_s:.2,input_ack_after_s:1,audio:{available:true,muted:true,volume:.8,device:'Speakers'},input_helper:{connected:true}};online=true;render();`,context);
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


if __name__ == "__main__":
    unittest.main()
