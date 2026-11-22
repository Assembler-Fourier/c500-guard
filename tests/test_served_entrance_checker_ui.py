"""Exercise the actual dashboard_page response, never the dormant legacy page."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import dashboard_ui
import test_live_dashboard_freshness as freshness


class ServedEntranceCheckerTests(unittest.TestCase):
    js = freshness.LiveDashboardFreshnessTests.js

    @classmethod
    def setUpClass(cls):
        cls.page = dashboard_ui.dashboard_page('Isolated UI check')

    def harness(self):
        start = self.page.index('function entranceCheckStatus(')
        end = self.page.index('\nasync function getJSON(', start)
        return """
let mode='live',connectionOk=true;
let status={tone:'healthy',label:'Ready',cameraFresh:true,aiFresh:true,aligned:true,inputReady:true};
const deriveStatus=()=>status,finite=Number.isFinite,icon=()=>'',historyPoints=[];
const esc=v=>String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{});return nodes.get(id);};
const ml={provider:'Intel graphics (OpenVINO)',inference_ms:132,
 entrance_verifier:{state:'ready',model:'YOLOX-S entrance check',provider:'Intel graphics (OpenVINO)',precision:'f32',inference_ms:64}};
const aiReading=()=>({value:'Unavailable'});
""" + self.page[start:end]

    def test_actual_served_page_mounts_checker_status_and_accurate_compute_label(self):
        self.assertIn('<title>Isolated UI check</title>', self.page)
        self.assertIn('function entranceCheckStatus(', self.page)
        self.assertIn('note:entranceCheck', self.page)
        self.assertEqual(dashboard_ui.PAGE, dashboard_ui.dashboard_page())
        self.js(self.harness() + """
renderSystem({ml});let html=$('system-grid').innerHTML;
assert.ok(html.includes('Entrance check ready'));
assert.ok(html.includes('GPU · f32 · 64 ms'));
assert.ok(html.includes('GPU processing'));
assert.ok(html.includes('132 ms total'));
assert.ok(!html.includes('CPU processing'));
""")

    def test_optional_failure_is_visible_while_primary_status_stays_truthful(self):
        self.js(self.harness() + """
ml.entrance_verifier={state:'unavailable',error:'Missing <model> "file"'};
renderSystem({ml});const html=$('system-grid').innerHTML;
assert.ok(html.includes('Entrance check unavailable'));
assert.ok(html.includes('Primary ready'));
assert.ok(html.includes('Primary detector continues'));
assert.ok(html.includes('Missing &lt;model&gt; &quot;file&quot;'));
assert.ok(!html.includes('Entrance check ready'));
assert.ok(!html.includes('Missing <model>'));
""")

    def test_cached_ready_checker_and_latency_do_not_appear_current(self):
        self.js(self.harness() + """
connectionOk=false;status={...status,aiFresh:false,cameraFresh:false};
renderSystem({ml});let html=$('system-grid').innerHTML;
assert.ok(html.includes('Entrance check status unavailable'));
assert.ok(!html.includes('Entrance check ready'));
assert.ok(!html.includes('64 ms'));
assert.ok(!html.includes('132 ms total'));
connectionOk=true;status={...status,aiFresh:true,cameraFresh:true};
ml.entrance_verifier={state:'disabled'};renderSystem({ml});html=$('system-grid').innerHTML;
assert.ok(html.includes('Entrance check off'));
delete ml.entrance_verifier;renderSystem({ml});html=$('system-grid').innerHTML;
assert.ok(!html.includes('Entrance check ready'));
assert.ok(!html.includes('Entrance check unavailable'));
""")

    def test_cpu_fallback_and_mixed_provider_labels_are_accurate(self):
        self.js(self.harness() + """
assert.equal(aiComputeLabel({provider:'CPU'}),'CPU processing');
assert.equal(aiComputeLabel({provider:'CPU',entrance_verifier:ml.entrance_verifier}),'CPU + GPU processing');
assert.equal(aiComputeLabel({provider:'CPU',entrance_verifier:{state:'unavailable',provider:'Intel graphics (OpenVINO)'}}),'CPU processing');
assert.equal(aiComputeLabel({}),'AI processing');
""")

    def test_recovery_countdown_is_visible_without_claiming_secondary_ready(self):
        self.js(self.harness() + """
ml.entrance_verifier={state:'cooldown',retry_in_s:4.2,error:'Slow <check> recovering'};
renderSystem({ml});let html=$('system-grid').innerHTML;
assert.ok(html.includes('Entrance check retrying'));
assert.ok(html.includes('Retry in 5 s'));assert.ok(html.includes('Primary ready'));
assert.ok(!html.includes('Entrance check ready'));
assert.ok(html.includes('Slow &lt;check&gt; recovering'));
ml.entrance_verifier={state:'ready',provider:'Intel graphics (OpenVINO)'};
renderSystem({ml});html=$('system-grid').innerHTML;
assert.ok(html.includes('Entrance check ready'));assert.ok(!html.includes('Retry in'));
""")

    def test_deferred_and_stale_retries_are_not_presented_as_current_checks(self):
        self.js(self.harness() + """
ml.entrance_verifier={state:'deferred',error:'Preserve fresh primary'};
renderSystem({ml});let html=$('system-grid').innerHTML;
assert.ok(html.includes('Entrance check deferred'));
assert.ok(html.includes('Primary detector continues'));
ml.entrance_verifier={state:'cooldown',retry_in_s:5};status.aiFresh=false;
renderSystem({ml});html=$('system-grid').innerHTML;
assert.ok(html.includes('Entrance check status unavailable'));
assert.ok(!html.includes('Retry in 5 s'));
""")


if __name__ == '__main__':
    unittest.main()
