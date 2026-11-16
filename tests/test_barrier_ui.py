"""Behavior checks for the camera barrier UI embedded in the production page."""
import ast
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src" / "dashboard_ui.py"
NODE = shutil.which("node") or str(
    Path(os.environ["USERPROFILE"])
    / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
)


def read_page():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    return next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PAGE_TEMPLATE" for target in node.targets)
    )


class BarrierUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = read_page()
        cls.helpers = cls.page.split("// BARRIER_STATE_HELPERS_START", 1)[1].split(
            "// BARRIER_STATE_HELPERS_END", 1
        )[0]
        cls.drawer = cls.page.split("// BARRIER_DRAWER_START", 1)[1].split(
            "$('open-barriers').addEventListener", 1
        )[0]
        start = cls.page.index("function createTransport(")
        end = cls.page.index("return {path,read,command};", start)
        cls.transport = cls.page[start : cls.page.index("\n}", end) + 2]

    def js(self, script):
        result = subprocess.run(
            [NODE, "--input-type=module"],
            input="import assert from 'node:assert/strict';\n" + self.helpers + "\n" + script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    fixture = """
const context={live:true,connected:true,cameraFresh:true,aligned:true,pictureFresh:true};
const arm={id:'exit',name:'Untrusted custom label',state:'up',fresh:true,calibrated:true,samples:{up:true,down:true},confidence:.92,up_for_s:12,active_line:[[.1,.2],[.3,.4]]};
const read=(item=arm,ctx=context)=>barrierSnapshot({barriers:[item]},ctx)[0];
"""

    def test_only_verified_fresh_position_becomes_known(self):
        self.js(self.fixture + """
assert.equal(read().state,'up');
assert.equal(read({...arm,state:'DOWN'}).state,'down');
for(const key of Object.keys(context)){
 const result=read(arm,{...context,[key]:false});
 assert.equal(result.state,'unknown',key);assert.equal(result.line,null,key);
}
for(const patch of [{fresh:false},{fresh:'true'},{calibrated:false},{samples:{up:true,down:false}},{state:'moving'},{state:null}]){
 const result=read({...arm,...patch});assert.equal(result.state,'unknown');assert.equal(result.line,null);
}
""")

    def test_missing_and_duplicate_ids_cannot_invent_barrier_state(self):
        self.js(self.fixture + """
for(const input of [null,{}, {barriers:'bad'},{barriers:[null,{id:'other',state:'up'}]}]){
 const rows=barrierSnapshot(input,context);assert.equal(rows.length,2);
 assert.deepEqual(rows.map(row=>row.name),['Exit','Entrance']);
 assert.ok(rows.every(row=>row.state==='unknown'));
}
assert.equal(barrierSnapshot({barriers:[arm,arm]},context)[0].state,'unknown');
assert.equal(read().name,'Exit');
""")

    def test_invalid_line_is_not_drawn_and_geometry_is_copied(self):
        self.js(self.fixture + """
for(const line of [null,[],[[.1,.2]],[[.1,.2],[.1,.2]],[[0,0],[1.1,1]],[[NaN,0],[.2,.2]],[[0,0],[Infinity,.2]],[[0,0],['.2',.2]],[[0,0],[1,1],[.5,.5]],[[0,0,0],[1,1]]]){
 const result=read({...arm,active_line:line});assert.equal(result.line,null);assert.equal(barrierLinesMarkup([result]),'');
}
const copied=read();copied.line[0][0]=.9;assert.equal(arm.active_line[0][0],.1);
const markup=barrierLinesMarkup([read()]);assert.match(markup,/M64.00,72.00 L192.00,144.00/);
assert.match(markup,/Exit barrier UP/);assert.doesNotMatch(markup,/Untrusted/);
""")

    def test_unknown_state_clears_confidence_and_up_duration(self):
        self.js(self.fixture + """
const unknown=read({...arm,fresh:false});assert.equal(unknown.confidence,null);assert.equal(unknown.upFor,null);
assert.equal(read({...arm,state:'down'}).upFor,null);
for(const confidence of [NaN,Infinity,-.1,1.1,'0.9'])assert.equal(read({...arm,confidence}).confidence,null);
for(const up_for_s of [NaN,Infinity,-1,'12'])assert.equal(read({...arm,up_for_s}).upFor,null);
""")

    def test_capture_payload_cannot_command_physical_hardware(self):
        self.js("""
assert.deepEqual(barrierRequest('capture','exit','up'),{action:'capture',id:'exit',position:'up'});
assert.deepEqual(barrierRequest('capture','entrance','down'),{action:'capture',id:'entrance',position:'down'});
assert.deepEqual(barrierRequest('reset','exit','up'),{action:'reset',id:'exit'});
for(const args of [['open','exit'],['close','entrance'],['capture','other','up'],['capture','exit','moving'],['capture','exit',null],['reset','../../power']])assert.throws(()=>barrierRequest(...args));
""")

    def test_transport_requires_success_receipt_and_never_controls_preview(self):
        self.js(self.transport + """
const calls=[];
const transport=createTransport({production:true,fetcher:async(...args)=>{calls.push(args);return {ok:true,json:async()=>({ok:true})};}});
await transport.command('barrier-calibration',barrierRequest('capture','exit','up'));
assert.equal(calls.length,1);assert.equal(calls[0][0],'/barrier-calibration');
assert.equal(calls[0][1].method,'POST');assert.equal(calls[0][1].headers['X-Via'],'web');
assert.deepEqual(JSON.parse(calls[0][1].body),{action:'capture',id:'exit',position:'up'});
for(const response of [{ok:false,json:async()=>({ok:false,error:'Arm view is ambiguous'})},{ok:true,json:async()=>({captured:true})},{ok:true,json:async()=>{throw Error('invalid');}}]){
 const failed=createTransport({production:true,fetcher:async()=>response});
 await assert.rejects(failed.command('barrier-calibration',{}));
}
let sent=false;const preview=createTransport({production:false,fetcher:async()=>{sent=true;}});
await assert.rejects(preview.command('barrier-calibration',{}));assert.equal(sent,false);
""")

    def drawer_fixture(self):
        return """
const nodes=new Map();const $=id=>{if(!nodes.has(id))nodes.set(id,{textContent:''});return nodes.get(id);};
let production=true,connectionOk=true,shutdownRequested=false,mode='live',displayedFrame='blob:actual',lastFrameAt=performance.now(),lastStateAt=performance.now();
let current={barriers:null};const currentState=()=>current;let status={cameraFresh:true,aligned:true};const deriveStatus=()=>status;
let renders=0;const render=()=>{renders++};let calls=[];let next=async()=>({ok:true});
const transport={command:async(...args)=>{calls.push(args);return next()}};
""" + self.drawer

    def test_capture_rejection_is_not_reported_as_saved_or_retried(self):
        self.js(self.drawer_fixture() + """
next=async()=>{throw Error('Arm view is ambiguous')};
await saveBarrierView('capture','exit','up');
assert.equal(calls.length,1);assert.equal($('barrier-feedback').textContent,'Arm view is ambiguous');assert.equal(barrierBusy,false);
assert.doesNotMatch($('barrier-feedback').textContent,/saved/);
""")

    def test_capture_is_explicit_nonoverlapping_and_requires_current_picture(self):
        self.js(self.drawer_fixture() + """
assert.equal(calls.length,0);
let resolve;next=()=>new Promise(done=>{resolve=done});
const pending=saveBarrierView('capture','exit','up');
assert.equal(barrierBusy,true);await saveBarrierView('capture','entrance','down');assert.equal(calls.length,1);
resolve({ok:true});await pending;assert.equal(barrierBusy,false);assert.equal($('barrier-feedback').textContent,'Exit UP view saved.');
for(const modeKey of ['picture','alignment','connection','stateage','shutdown','preview']){
 displayedFrame='blob:actual';status={cameraFresh:true,aligned:true};connectionOk=true;lastStateAt=performance.now();shutdownRequested=false;production=true;
 if(modeKey==='picture')displayedFrame=null;if(modeKey==='alignment')status.aligned=false;if(modeKey==='connection')connectionOk=false;if(modeKey==='stateage')lastStateAt=performance.now()-2500;if(modeKey==='shutdown')shutdownRequested=true;if(modeKey==='preview')production=false;
 await saveBarrierView('capture','exit','down');assert.equal(calls.length,1,modeKey);
}
""")

    def test_drawer_does_not_add_camera_requests_or_polling(self):
        body = self.page.split("// BARRIER_DRAWER_START", 1)[1].split("// BARRIER_DRAWER_END", 1)[0]
        self.assertNotRegex(body, r"\b(?:fetch|setInterval|setTimeout|readState|readFrame)\(")
        self.assertIn("$('barrier-capture-image').src=displayedFrame", body)
        self.assertIn("performance.now()-lastStateAt<1800", body)
        self.assertIn("barrierReturnFocus.focus()", body)
        self.assertIn("event.key!=='Tab'", body)
        self.assertIn("showModal()", body)
        self.assertIn("window.addEventListener('pagehide'", body)

    def test_calibration_markup_has_distinct_labeled_controls(self):
        class Collector(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = {}
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if attrs.get("id", "").startswith(("barrier-", "capture-", "reset-", "open-barriers", "close-barriers")):
                    self.ids.setdefault(attrs["id"], []).append((tag, attrs))
        collector = Collector()
        collector.feed(self.page)
        self.assertTrue(all(len(nodes) == 1 for nodes in collector.ids.values()))
        tag, attrs = collector.ids["barrier-drawer"][0]
        self.assertEqual(tag, "dialog")
        self.assertEqual(attrs["aria-labelledby"], "barrier-title")
        for arm in ("exit", "entrance"):
            for position in ("up", "down"):
                _, button = collector.ids[f"capture-{arm}-{position}"][0]
                self.assertIn("disabled", button)
                self.assertEqual(button["data-barrier-position"], position)
                self.assertIn(arm, button["aria-label"])
        self.assertIn("it does not move the barrier", self.page)

    def test_complete_javascript_bundle_parses(self):
        for script in re.findall(r"<script\b[^>]*>(.*?)</script>", self.page, re.S):
            if not script.strip():
                continue
            result = subprocess.run(
                [NODE, "--check", "--input-type=module"], input=script, text=True,
                encoding="utf-8", capture_output=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
