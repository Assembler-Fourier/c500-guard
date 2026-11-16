"""Production-page behavior with fake clocks and transport; never contacts the watcher."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src" / "dashboard_ui.py"
NODE = shutil.which("node") or str(
    Path(os.environ["USERPROFILE"])
    / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
)


def page_source():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "PAGE_TEMPLATE"
                        for target in node.targets))


class LiveDashboardFreshnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = page_source()

    def source(self, start, end):
        return self.page.split(start, 1)[1].split(end, 1)[0]

    def js(self, source):
        result = subprocess.run([NODE, "--input-type=module"],
                                input="import assert from 'node:assert/strict';\n" + source,
                                text=True, encoding="utf-8", capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_capture_rejection_retains_the_backend_reason(self):
        transport = "function createTransport(" + self.source(
            "function createTransport(", "return {path,read,command};") + "return {path,read,command};\n}"
        self.js(transport + r"""
for (const [data,expected] of [
 [{ok:false,reason:'Keep the labelled barrier still in five fresh aligned frames'}, /five fresh aligned frames/],
 [{ok:false,error:'Clear lanes required',reason:'Other detail'}, /Clear lanes required/],
 [{ok:false,message:'Disconnected'}, /Disconnected/],
 [{ok:false}, /Command rejected \(409\)/],
]) {
 const transport=createTransport({production:true,fetcher:async()=>({ok:false,status:409,json:async()=>data})});
 await assert.rejects(transport.command('barrier-calibration',{}),expected);
}
""")

    def test_stalled_state_expires_all_status_not_just_barrier_pills(self):
        timer = next(line for line in self.page.splitlines() if line.startswith("setInterval(()=>"))
        self.js("""
let now=1000,active=true,mode='live',connectionOk=true,lastStateAt=1000,production=true;
let callback,renders=0,clockUpdates=0,barrierConnected;
const performance={now:()=>now};const setInterval=fn=>{callback=fn};
const updateClock=()=>clockUpdates++;const currentState=()=>({ack_remaining_s:0});
const deriveStatus=(_s,connected)=>({connected});
const renderBarriers=(_s,status)=>{barrierConnected=status.connected};
const render=()=>{renders++};const activateScenario=()=>assert.fail('No simulated changes');
""" + timer + """
callback();assert.equal(connectionOk,true);assert.equal(renders,0);
now=2799;callback();assert.equal(connectionOk,true);
now=2800;callback();assert.equal(connectionOk,false);
assert.equal(barrierConnected,false);assert.equal(renders,1);
callback();assert.equal(renders,1,'A stalled connection causes one refresh, not a 10 Hz rebuild');
active=false;now=4000;callback();assert.equal(clockUpdates,4);
""")

    def test_restored_page_rejects_cached_controls_before_new_reads(self):
        restore = next(line for line in self.page.splitlines() if line.startswith("window.addEventListener('pageshow'"))
        self.js("""
let listener,active=false,displayedFrame='blob:old',connectionOk=true,lastStateAt=200,lastFrameAt=200,lastAckAt=200;
const page='monitor';
let events=[];const window={addEventListener:(_name,fn)=>listener=fn};
const render=()=>events.push(['render',connectionOk,displayedFrame]);
const readState=()=>events.push(['state',connectionOk]);
const readFrame=()=>events.push(['frame',connectionOk]);const readConfiguration=()=>events.push(['config',connectionOk]);
""" + restore + """
listener({persisted:false});assert.deepEqual(events,[]);
listener({persisted:true});assert.equal(active,true);assert.equal(connectionOk,false);
assert.equal(displayedFrame,null);assert.equal(lastFrameAt,0);assert.equal(lastStateAt,0);
assert.deepEqual(events,[['render',false,null],['state',false],['frame',false],['config',false]]);
""")

    def test_stale_click_cannot_start_a_command_before_timer_refresh(self):
        command = "async function runCommand(" + self.source("async function runCommand(", "async function applyZones(")
        self.js("""
let now=3000,commandBusy=false,connectionOk=true,shutdownRequested=false,lastStateAt=1000,lastAckAt=1000;
let calls=0,renders=0,liveState={armed:false,alarm_id:'visit-1',last_ack:{via:'input-mouse'}};
const performance={now:()=>now},finite=Number.isFinite,render=()=>renders++,toast=()=>{};
const transport={command:async()=>{calls++;return {ok:true,armed:false,ack_remaining_s:10}}};
""" + command + """
await runCommand('arm');assert.equal(calls,0);assert.equal(liveState.armed,false);
now=2000;await runCommand('ack',{alarm_id:'visit-1'});assert.equal(calls,1);
assert.equal(lastStateAt,1000,'An ACK receipt cannot renew cached camera/AI health');
assert.equal(lastAckAt,2000,'Quiet time starts from the ACK receipt');
assert.equal(liveState.alarm_id,null);assert.equal(liveState.ack_remaining_s,10);
assert.equal(liveState.last_ack,null,"A new ACK receipt must not display the previous alarm's input method");
""")

    def test_replaced_state_request_cannot_overwrite_newer_state(self):
        request = "async function readState(" + self.source("async function readState(", "async function readEvents(")
        self.js("""
let active=true,stateController=null,liveState={},connectionOk=false,lastStateAt=0,lastAckAt=0;
let historyPoints=[],lastHistoryFrame=null,lastEventVersion=1,lastEventsAt=1000;
let pending=[],renders=0,timers=[],cleared=[];
const performance={now:()=>1000},finite=Number.isFinite;
const setTimeout=(fn,ms)=>{timers.push([fn,ms]);return timers.length};const clearTimeout=id=>cleared.push(id);
const getJSON=()=>new Promise((resolve,reject)=>pending.push({resolve,reject}));
const render=()=>renders++;const readEvents=()=>{},maybeReloadForUpdate=()=>{};
""" + request + """
const old=readState(),oldController=stateController;
const latest=readState(),newController=stateController;
assert.notEqual(oldController,newController);
timers[0][0]();assert.equal(oldController.signal.aborted,true);
assert.equal(newController.signal.aborted,false,'The old timeout must not abort a newer request');
pending[1].resolve({frames:22,alerts_version:1});await latest;
pending[0].resolve({frames:11,alerts_version:1});await old;
assert.equal(liveState.frames,22);assert.equal(renders,1);
assert.equal(connectionOk,true,'A successful state response must remain connected');
assert.equal(timers.filter(([,ms])=>ms===600).length,1,'Only the current request schedules its successor');
""")

    def test_frame_decode_timeout_releases_polling_and_discards_the_late_image(self):
        frame = "async function readFrame(" + self.source(
            "async function readFrame(", "document.querySelectorAll('[data-page]')")
        self.js("""
let active=true,frameRunning=false,frameController=null,displayedFrame='blob:initial',lastFrameAt=1000;
let page='monitor',mode='live',connectionOk=true,now=1000,fetches=0;
const performance={now:()=>now},timers=[],revoked=[],decodes=new Map(),nodes=new Map();
const setTimeout=(fn,ms)=>{timers.push([fn,ms]);return timers.length};const clearTimeout=()=>{};
const transport={path:p=>p},fetch=async()=>({ok:true,blob:async()=>({id:++fetches})});
const URL={createObjectURL:blob=>'blob:'+blob.id,revokeObjectURL:url=>revoked.push(url)};
class Image{decode(){return new Promise(resolve=>decodes.set(this.src,resolve));}}
const $=id=>{if(!nodes.has(id))nodes.set(id,{open:false,classList:{remove(){}}});return nodes.get(id);};
const editor={updateImage:()=>assert.fail('Monitor must not edit a zone')};
const currentState=()=>({}),deriveStatus=()=>({cameraFresh:true}),render=()=>{};
const flush=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
""" + frame + """
const old=readFrame();await flush();assert.equal(frameRunning,true);
assert.equal(decodes.has('blob:1'),true);
now=5000;timers[0][0]();await flush();
assert.equal(frameRunning,false,'A stalled decode must release the polling lock when its deadline expires');
assert.equal(frameController,null);assert.equal(displayedFrame,'blob:initial');
assert.equal(revoked.filter(url=>url==='blob:1').length,1);
assert.equal(timers.filter(([,ms])=>ms===180).length,1,'Schedule only one replacement attempt');
await old;
const next=readFrame();await flush();now=5100;decodes.get('blob:2')();await next;
assert.equal(displayedFrame,'blob:2');assert.equal(lastFrameAt,5100);
decodes.get('blob:1')();await flush();
assert.equal(displayedFrame,'blob:2');assert.equal(lastFrameAt,5100);
assert.equal(revoked.filter(url=>url==='blob:1').length,1);
""")

    def test_pre_restore_frame_cannot_replace_the_new_live_picture(self):
        frame = "async function readFrame(" + self.source(
            "async function readFrame(", "document.querySelectorAll('[data-page]')")
        lifecycle = "\n".join(line for line in self.page.splitlines()
                              if line.startswith("window.addEventListener('pagehide',()=>{active=false")
                              or line.startswith("window.addEventListener('pageshow'"))
        self.js("""
let active=true,frameRunning=false,frameController=null,stateController=null,configurationTimer=null;
const stopVehicleStatus=()=>{};
let displayedFrame='blob:initial',lastFrameAt=1000,lastStateAt=1000,lastAckAt=1000;
let page='monitor',mode='live',connectionOk=true,now=1000,fetches=0;
const performance={now:()=>now},timers=[],revoked=[],decodes=new Map(),listeners={},nodes=new Map();
const setTimeout=(fn,ms)=>{timers.push([fn,ms]);return timers.length};const clearTimeout=()=>{};
const transport={path:p=>p},fetch=async()=>({ok:true,blob:async()=>({id:++fetches})});
const URL={createObjectURL:blob=>'blob:'+blob.id,revokeObjectURL:url=>revoked.push(url)};
class Image{decode(){return new Promise(resolve=>decodes.set(this.src,resolve));}}
const $=id=>{if(!nodes.has(id))nodes.set(id,{open:false,classList:{remove(){}}});return nodes.get(id);};
const editor={updateImage:()=>assert.fail('Monitor must not edit a zone')};
const currentState=()=>({}),deriveStatus=()=>({cameraFresh:true}),render=()=>{};
const readState=()=>{connectionOk=true},readConfiguration=()=>{};
const window={addEventListener:(name,fn)=>listeners[name]=fn};
const flush=async()=>{await Promise.resolve();await Promise.resolve();await Promise.resolve();};
""" + frame + lifecycle + """
const oldRequest=readFrame();await flush();const oldController=frameController;
assert.equal(decodes.has('blob:1'),true,'Old frame is waiting to decode');
listeners.pagehide();assert.equal(oldController.signal.aborted,true);
listeners.pageshow({persisted:true});await flush();const newController=frameController;
assert.equal(decodes.has('blob:2'),true,'Restore must start a new frame request immediately');
assert.notEqual(oldController,newController);
timers[0][0]();assert.equal(newController.signal.aborted,false,'Old timeout cannot abort the restored request');
now=2000;decodes.get('blob:2')();await flush();
assert.equal(displayedFrame,'blob:2');assert.equal($('feed').src,'blob:2');assert.equal(lastFrameAt,2000);
now=3000;decodes.get('blob:1')();await oldRequest;
assert.equal(displayedFrame,'blob:2');assert.equal($('feed').src,'blob:2');assert.equal(lastFrameAt,2000);
assert.equal(revoked.filter(url=>url==='blob:1').length,1,'Discard the obsolete decoded image');
assert.equal(revoked.includes('blob:2'),false,'Keep the current picture usable');
assert.equal(timers.filter(([,ms])=>ms===180).length,1,'An obsolete request must not add a polling chain');
""")


if __name__ == "__main__":
    unittest.main()
