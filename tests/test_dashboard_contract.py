"""Frontend configuration and operator-status contracts, with synthetic data only."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import dashboard_ui
from region_geometry import MAX_POLYGON_CORNERS
import test_live_dashboard_freshness as freshness


class DashboardContractTests(unittest.TestCase):
    js = freshness.LiveDashboardFreshnessTests.js

    def module(self, name):
        return dashboard_ui.PAGE.split('__c500Modules["' + name + '"] = (() => {', 1)[1].split('\n})();', 1)[0]

    def test_editor_geometry_uses_the_shared_limit_and_preserves_detailed_outlines(self):
        legacy = ('function validPolygon(points){' + dashboard_ui.EDITOR_SCRIPT.split(
            'function validPolygon(points){', 1)[1].split('function fitPolygon(', 1)[0])
        self.js('const MAX_POLYGON_CORNERS=' + str(MAX_POLYGON_CORNERS) + ';\nconst zones=(()=>{' + self.module('zones.js') + '})();\n' + legacy + """
const polygon=count=>Array.from({length:count},(_,i)=>[.5+.35*Math.cos(2*Math.PI*i/count),.5+.35*Math.sin(2*Math.PI*i/count)]);
for(const count of [21,MAX_POLYGON_CORNERS]){
 const points=polygon(count),saved=JSON.stringify(points);
 assert.equal(zones.validatePolygon(points),null);
 assert.equal(validPolygon(points),true);
 assert.equal(JSON.stringify(points),saved,'Validation must not simplify or change the outline');
}
assert.match(zones.validatePolygon(polygon(MAX_POLYGON_CORNERS+1)),/32/);
assert.equal(validPolygon(polygon(MAX_POLYGON_CORNERS+1)),false);
assert.ok(zones.validatePolygon([[.2,.2],[.8,.8],[.8,.2],[.2,.8]]));
""")
        self.assertIn(f'const MAX_POLYGON_CORNERS = {MAX_POLYGON_CORNERS};', dashboard_ui.PAGE)
        self.assertIn(f'const MAX_POLYGON_CORNERS={MAX_POLYGON_CORNERS};', dashboard_ui.EDITOR_SCRIPT)
        self.assertNotIn('polygon.length>=16', dashboard_ui.EDITOR_SCRIPT)

    def test_ai_delays_remain_warnings_and_do_not_extend_freshness(self):
        self.js('const ui=(()=>{' + self.module('state.js') + '})();\n' + """
const base={armed:true,frames:10,last_frame_age:1.8,loop_age_s:0,alignment:{state:'tracking'},input_helper:{connected:true},audio:{available:true,muted:false,volume:.5},ack_remaining_s:8.2};
for(const age of [.6,1.4,2.9]){
 const status=ui.deriveStatus({...base,ml:{available:true,state:'ready',age_s:age}});
 assert.equal(status.aiFresh,true);assert.equal(status.title,'Acknowledged');
}
for(const state of ['ready','waiting_for_frames']){
 const status=ui.deriveStatus({...base,ml:{available:true,state,age_s:3.1}});
 assert.equal(status.aiFresh,false);assert.equal(status.tone,'warn');
 assert.equal(status.title,'AI result delayed');assert.match(status.detail,/3.1 s/);
 assert.equal(status.quietRemaining,8.2,'A quiet period must not hide stale observations');
}
assert.equal(ui.deriveStatus({...base,ml:{available:true,state:'waiting_for_frames',age_s:null}}).title,'AI waiting');
assert.equal(ui.deriveStatus({...base,ml:{available:false,state:'loading',age_s:null}}).title,'AI loading');
assert.equal(ui.deriveStatus({...base,ml:{available:false,state:'unavailable',age_s:9}}).title,'AI unavailable');
assert.equal(ui.aiReading(null).title,'AI unavailable');
assert.equal(ui.deriveStatus({...base,ml:{available:true,state:'ready',age_s:.6}},false).label,'Reconnecting');
""")

    def test_quiet_hint_only_names_a_recorded_current_acknowledgement(self):
        page = dashboard_ui.PAGE
        helper = 'function acknowledgementHint(' + page.split('function acknowledgementHint(', 1)[1].split('function render(){', 1)[0]
        self.js("const production=true;let mode='live';\n" + helper + """
const status={quietRemaining:8.2,alarmActive:false},ack={via:'input-any-key',at:'2026-11-15T20:02:16',alarm_id:'alarm-1'};
assert.equal(acknowledgementHint({last_ack:ack},status),'Acknowledged by a key at 20:02:16');
assert.equal(acknowledgementHint({last_ack:{...ack,via:'input-mouse'}},status),'Acknowledged by a mouse click at 20:02:16');
assert.equal(acknowledgementHint({last_ack:ack},{...status,quietRemaining:0}),'No active alarm');
assert.equal(acknowledgementHint({last_ack:ack},{...status,alarmActive:true}),'Any key or click · after 1 second');
for(const last_ack of [null,{...ack,via:'barrier-raised'},{...ack,at:'bad'},{...ack,alarm_id:null}]){
 assert.equal(acknowledgementHint({last_ack},status),'Same visit stays acknowledged');
}
mode='quiet';assert.equal(acknowledgementHint({last_ack:ack},status),'Same visit stays acknowledged');
""")


if __name__ == '__main__':
    unittest.main()
