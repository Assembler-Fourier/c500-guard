"""Background re-learning must not announce that the detector is settling."""
import json
import unittest
import test_dashboard_contract as contracts
import dashboard_ui
import test_live_dashboard_freshness as freshness


class DashboardReadinessTests(unittest.TestCase):
    module = contracts.DashboardContractTests.module
    js = freshness.LiveDashboardFreshnessTests.js

    def test_production_readiness_and_pedestrian_attention(self):
        self.js('const ui=(()=>{'+self.module('state.js')+'})();\n'+"""
const base={armed:true,frames:10,last_frame_age:.1,loop_age_s:0,
alignment:{state:'tracking'},input_helper:{connected:true},
audio:{available:true,muted:false,volume:.5},ml:{available:true,state:'ready',age_s:.1},
warmup_remaining_s:2,background_warmup_remaining_s:2,semantic_warmup_remaining_s:0};
assert.equal(ui.deriveStatus(base).title,'Watching');
assert.equal(ui.deriveStatus({...base,semantic_warmup_remaining_s:2}).title,'Settling');
const legacy={...base};delete legacy.semantic_warmup_remaining_s;
assert.equal(ui.deriveStatus(legacy).title,'Settling');
assert.equal(ui.deriveStatus({...base,ack_remaining_s:10}).title,'Acknowledged');
assert.equal(ui.deriveStatus({...base,armed:false}).title,'Alerts paused');
assert.equal(ui.deriveStatus({...base,fault_reason:'Fault'}).title,'System fault');
const alert=ui.deriveStatus({...base,alarm:'main',alarm_evidence:{...{},
source:'ai',label:'person',route_role:'pedestrian',zone_name:'Pedestrian crossing',manual_ack_required:true}});
assert.match(alert.detail,/acknowledge to stop/);
""")

    def test_legacy_status_uses_same_readiness_with_old_server_fallback(self):
        self.js('const window={},document={};\n'+dashboard_ui.CORE_SCRIPT+"""
const ui=window.C500UI;
const base={armed:true,warmup_remaining_s:2,semantic_warmup_remaining_s:0};
assert.doesNotMatch(ui.deriveStatus(base).detail,/settling/i);
assert.match(ui.deriveStatus({...base,semantic_warmup_remaining_s:2}).detail,/2 s/);
delete base.semantic_warmup_remaining_s;
assert.match(ui.deriveStatus(base).detail,/2 s/);
""")


if __name__=='__main__':
    unittest.main()
