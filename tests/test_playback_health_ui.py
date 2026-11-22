"""Playback health is separate from mixer health; no speaker calls or live server."""
import json
import unittest

import test_dashboard_contract as contracts
import test_dashboard_ui as legacy
import test_served_entrance_checker_ui as served
import dashboard_ui


class PlaybackHealthUITests(unittest.TestCase):
    js = served.ServedEntranceCheckerTests.js
    harness = served.ServedEntranceCheckerTests.harness
    module = contracts.DashboardContractTests.module
    node = legacy.DashboardTests.node
    dashboard_fixture = legacy.DashboardTests.dashboard_fixture

    @classmethod
    def setUpClass(cls):
        cls.page = dashboard_ui.dashboard_page('Isolated playback check')

    def state_harness(self):
        return 'const ui=(()=>{' + self.module('state.js') + '})();\n' + """
const base={armed:true,frames:10,last_frame_age:.1,loop_age_s:0,
alignment:{state:'tracking'},input_helper:{connected:true},
audio:{available:true,muted:false,volume:.8,playback:{error:''}},
ml:{available:true,state:'ready',age_s:.1},semantic_warmup_remaining_s:0};
const failedAudio={...base.audio,playback:{requested:'main',playing:false,
profile:'wake',error:'Playback failed (RuntimeError); retrying',retry_in_s:.5}};
"""

    def test_actual_alarm_card_reports_playback_failure_without_losing_arrival(self):
        self.js(self.state_harness() + """
const s={...base,audio:failedAudio,alarm:'main',alarm_source:'ai',
alarm_evidence:{source:'ai',zone_name:'Entrance waiting',route_role:'entrance'}};
const failed=ui.deriveStatus(s),normal=ui.deriveStatus({...s,audio:base.audio});
assert.equal(failed.alarmActive,true);assert.equal(failed.tone,'alarm');
assert.equal(failed.title,normal.title,'The arrival must remain the primary alarm');
assert.equal(failed.label,normal.label);assert.equal(failed.overline,normal.overline);
assert.match(failed.detail,/Alarm playback failed; retrying/);
assert.ok(failed.detail.endsWith(normal.detail));
assert.doesNotMatch(normal.detail,/playback failed/);
const fault=ui.deriveStatus({...s,alarm:'fault',fault_reason:'Camera lost',reason:'Camera lost'});
assert.equal(fault.title,'System fault');assert.match(fault.detail,/Camera lost/);
assert.match(fault.detail,/Alarm playback failed; retrying/);
""")

    def test_actual_monitor_health_recovers_and_never_flags_intentional_silence(self):
        self.js(self.state_harness() + """
const failed=ui.deriveStatus({...base,audio:failedAudio});
assert.equal(failed.tone,'warn');assert.equal(failed.title,'Alarm playback failed');
for(const playback of [undefined,{},
 {requested:'main',playing:false,profile:'silent',error:''},
 {requested:null,playing:false,error:'   '},
 {requested:'main',playing:true,profile:'wake',error:''}]){
 const status=ui.deriveStatus({...base,audio:{...base.audio,playback}});
 assert.equal(status.title,'Watching');assert.doesNotMatch(status.detail,/playback failed/);
}
const quiet=ui.deriveStatus({...base,ack_remaining_s:10});
assert.equal(quiet.title,'Acknowledged');assert.equal(quiet.quietRemaining,10);
const offline=ui.deriveStatus({...base,audio:failedAudio,alarm:'main'},false);
assert.equal(offline.label,'Reconnecting');assert.doesNotMatch(offline.detail,/playback failed/);
""")

    def test_actual_speaker_badge_shows_failure_and_clears_after_recovery(self):
        self.js(self.harness() + """
const audio={available:true,muted:false,volume:.8,
playback:{requested:'main',playing:false,error:'Playback failed (RuntimeError); retrying'}};
renderSystem({ml,audio});
assert.match($('speaker-state').className,/warn/);
assert.match($('speaker-state').innerHTML,/Alarm playback failed/);
assert.equal($('speaker-state').title,audio.playback.error);
audio.playback={requested:'main',playing:true,error:''};renderSystem({ml,audio});
assert.doesNotMatch($('speaker-state').className,/warn/);
assert.match($('speaker-state').innerHTML,/Speaker 80%/);
assert.equal($('speaker-state').title,'');
audio.playback={requested:'main',playing:false,profile:'silent',error:''};
renderSystem({ml,audio});assert.doesNotMatch($('speaker-state').className,/warn/);
assert.doesNotMatch($('speaker-state').innerHTML,/failed/);
""")

    def test_actual_speaker_error_is_text_only_and_offline_does_not_show_cached_failure(self):
        self.js(self.harness() + """
const error='<img src=x onerror=alert(1)>',audio={available:true,muted:false,
volume:.8,playback:{error}};
renderSystem({ml,audio});
assert.equal($('speaker-state').title,error,'Diagnostic is a DOM text property');
assert.ok(!$('speaker-state').innerHTML.includes(error));
connectionOk=false;renderSystem({ml,audio});
assert.match($('speaker-state').innerHTML,/Speaker unavailable/);
assert.ok(!$('speaker-state').innerHTML.includes('failed'));
assert.equal($('speaker-state').title,'');
""")

    def test_legacy_health_warning_clears_on_recovery_and_ack_silence(self):
        self.node(self.dashboard_fixture() +
                  'vm.runInContext(' + json.dumps(dashboard_ui.CORE_SCRIPT + dashboard_ui.DASHBOARD_SCRIPT) + ',context);\n' + r"""
vm.runInContext(`online=true;latest={armed:true,frames:5,last_frame_age:.1,
alignment:{state:'tracking'},ml:{available:true,state:'ready',age_s:.1},
loop_age_s:.2,input_ack_after_s:1,input_helper:{connected:true},
audio:{available:true,muted:false,volume:.8,playback:{error:'Playback failed <device>; retrying'}}};
render();`,context);
assert.equal(elements.get('health-speaker').textContent,'Playback failed');
assert.equal(elements.get('health-warning').hidden,false);
assert.match(elements.get('health-warning').textContent,/Playback failed <device>/);
vm.runInContext(`latest.audio.playback={requested:null,playing:false,error:''};render();`,context);
assert.equal(elements.get('health-warning').hidden,true);
assert.equal(elements.get('health-badge').textContent,'Healthy');
assert.equal(elements.get('health-speaker').textContent,'80% volume');
""")


if __name__ == '__main__':
    unittest.main()
