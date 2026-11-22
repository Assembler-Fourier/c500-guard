"""Pixel-only settling is secondary System telemetry, not a global stop."""
import unittest
import test_live_dashboard_freshness as freshness


class BackgroundReadinessTests(unittest.TestCase):
    js = freshness.LiveDashboardFreshnessTests.js

    def test_system_camera_card_reports_background_without_blocking_readiness(self):
        page = freshness.page_source()
        body = 'function renderSystem(s){' + page.split('function renderSystem(s){',1)[1].split('async function getJSON',1)[0]
        self.js("""
const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{});return nodes.get(id)};
const mode='live',connectionOk=true,historyPoints=[],finite=Number.isFinite;
const icon=()=>'',esc=value=>String(value),entranceCheckStatus=()=>null,aiComputeLabel=()=>'',aiReading=()=>({value:'Ready'});
let cameraFresh=true;
const deriveStatus=()=>({tone:'healthy',label:'Watching',cameraFresh,aiFresh:true,aligned:true,inputReady:true});
"""+body+"""
const base={last_frame_age:.1,audio:{available:true,muted:false,volume:.5},
background_warmup_remaining_s:1.2,semantic_warmup_remaining_s:0};
renderSystem(base);
assert.match($('system-grid').innerHTML,/Image background/);
assert.match($('system-grid').innerHTML,/Settling · 2 s/);
assert.equal($('system-state').textContent,'Watching');
renderSystem({...base,background_warmup_remaining_s:0});
assert.match($('system-grid').innerHTML,/Image background/);
assert.doesNotMatch($('system-grid').innerHTML,/Settling/);
for(const missing of [undefined,null,NaN,-1]){
 renderSystem({...base,background_warmup_remaining_s:missing});
 assert.doesNotMatch($('system-grid').innerHTML,/Image background/);
}
cameraFresh=false;renderSystem(base);
assert.doesNotMatch($('system-grid').innerHTML,/Image background/);
""")


if __name__=='__main__':
    unittest.main()
