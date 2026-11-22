"""Lane panels and inspection crops use fake DOM, clocks and transport only."""
from html.parser import HTMLParser
import unittest

import test_live_dashboard_freshness as freshness
import test_dashboard_view_status as view_status


class LaneDashboardUiTests(unittest.TestCase):
    source = freshness.LiveDashboardFreshnessTests.source
    js = freshness.LiveDashboardFreshnessTests.js

    @classmethod
    def setUpClass(cls):
        cls.page = freshness.page_source()
        cls.helpers = cls.page.split('// BARRIER_STATE_HELPERS_START', 1)[1].split(
            '// BARRIER_STATE_HELPERS_END', 1)[0]

    def routes_harness(self):
        context = 'function barrierContext(' + self.source('function barrierContext(', 'function renderBarrierInspection(')
        routes = 'function hearLane(' + self.source('function hearLane(', 'function highlightRoute(')
        metadata = next(line for line in self.page.splitlines() if line.startswith('const roleMeta='))
        return r"""
let now=1000,production=true,mode='live',connectionOk=true,shutdownRequested=false,lastStateAt=1000,lastFrameAt=1000,displayedFrame='blob:fresh',commandBusy=false,renderedRoutes='';
const performance={now:()=>now};let html='',navigated=null,selected=null,buttons=[],sounds=[];
const nodes=new Map();
const $=id=>{if(!nodes.has(id))nodes.set(id,{textContent:'',className:'',disabled:false});return nodes.get(id)};
nodes.set('route-cards',{set innerHTML(value){html=value;buttons=[];sounds=[]},get innerHTML(){return html}});
const esc=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const icon=name=>`<svg data-icon="${name}"></svg>`,routeRole=zone=>zone.route_role;
const navigate=page=>navigated=page,editor={selectRoute:role=>selected=role},highlightRoute=()=>{};
function button(dataset){return {dataset,listeners:{},addEventListener(name,fn){this.listeners[name]=fn}}}
const document={querySelectorAll(selector){
 if(selector==='.lane-top')return buttons=[...html.matchAll(/data-route="([a-z]+)"/g)].map(m=>button({route:m[1]}));
 if(selector==='[data-lane-sound]')return sounds=[...html.matchAll(/data-lane-sound="([a-z]+)"/g)].map(m=>{const b=button({laneSound:m[1]});nodes.set('hear-'+m[1],b);return b});
 return [];
}};
const calls=[],runCommand=async(...args)=>calls.push(args);
const arm=(id,state)=>({id,state,fresh:true,calibrated:true,samples:{up:1,down:1},reason:'Calibrated arm position confirmed'});
const current={armed:true,zones:[{name:'Entrance stop',route_role:'entrance'},{name:'Exit corner',route_role:'exit'},{name:'People',route_role:'pedestrian'}],barriers:{barriers:[arm('exit','down'),arm('entrance','up')]}};
const healthy={cameraFresh:true,aligned:true,aiFresh:true};
""" + self.helpers + context + metadata + routes

    def test_lane_cards_show_only_current_own_barrier_and_preserve_zone_navigation(self):
        self.js(self.routes_harness() + r"""
renderRoutes(current,healthy);
assert.deepEqual(buttons.map(b=>b.dataset.route),['entrance','exit','pedestrian']);
const cards=[...html.matchAll(/<article.*?<\/article>/gs)].map(m=>m[0]);
assert.match(cards[0],/Entrance/);assert.equal($('lane-entrance-arm').textContent,'UP');
assert.match(cards[1],/Exit/);assert.equal($('lane-exit-arm').textContent,'DOWN');
assert.match(cards[2],/Footpath/);assert.doesNotMatch(cards[2],/lane-pedestrian-arm|data-lane-sound/);
buttons[1].listeners.click();assert.equal(navigated,'setup');assert.equal(selected,'exit');
assert.equal($('hear-entrance').disabled,false);assert.equal($('hear-exit').disabled,false);
""")

    def test_barrier_rows_open_barrier_setup_and_ask_for_it_only_when_unknown(self):
        self.js(self.routes_harness() + r"""
renderRoutes(current,healthy);
assert.equal((html.match(/class="lane-barrier-link"/g)||[]).length,2,'Entrance and exit rows open barrier setup');
assert.equal((html.match(/data-bell="/g)||[]).length,2,'Entrance and exit rows carry the barrier-up reminder switch');
assert.equal($('lane-entrance-setup').hidden,true);assert.equal($('lane-exit-setup').hidden,true);
const unknown=structuredClone(current);unknown.barriers.barriers[1].fresh=false;renderRoutes(unknown,healthy);
assert.equal($('lane-entrance-arm').textContent,'UNKNOWN');assert.equal($('lane-entrance-setup').hidden,false);
assert.equal($('lane-exit-setup').hidden,true);
""")
        self.assertIn("document.querySelectorAll('.lane-barrier-link').forEach(button=>button.addEventListener('click',()=>navigate('setup','barriers')));", self.page)
        self.assertIn('data-go="setup" data-setup="barriers"', self.page)

    def test_stale_and_preview_panels_never_reuse_memory_as_current_position(self):
        self.js(self.routes_harness() + r"""
current.barriers.memory={exit:{state:'up',seen_ago_s:1,for_s:100},entrance:{state:'down',seen_ago_s:1}};
for(const condition of ['state','picture','disconnected','unaligned','raw']){
 now=1000;lastStateAt=lastFrameAt=1000;connectionOk=true;mode='live';renderedRoutes='';
 let status={...healthy},state=structuredClone(current);
 if(condition==='state')now=2800;
 if(condition==='picture')lastFrameAt=-4000;
 if(condition==='disconnected')connectionOk=false;
 if(condition==='unaligned')status.aligned=false;
 if(condition==='raw')state.barriers.barriers.forEach(arm=>arm.fresh=false);
 renderRoutes(state,status);
 assert.equal($('lane-entrance-arm').textContent,'UNKNOWN',condition);
 assert.equal($('lane-exit-arm').textContent,'UNKNOWN',condition);
 if(['state','disconnected'].includes(condition)){
  assert.equal($('hear-entrance').disabled,true,condition);assert.equal($('hear-exit').disabled,true,condition);
  for(const role of ['entrance','exit','pedestrian'])assert.doesNotMatch($(`lane-${role}-indicator`).className,/healthy/,condition);
 }
}
""")

    def test_sound_request_keeps_existing_stale_busy_and_preview_protections(self):
        command = 'async function runCommand(' + self.source('async function runCommand(', 'async function applyZones(')
        hear = 'function hearLane(' + self.source('function hearLane(', 'function renderRoutes(')
        self.js(r"""
let now=1000,production=true,mode='live',commandBusy=false,connectionOk=true,shutdownRequested=false,lastStateAt=1000,lastAckAt=0,liveState={};
const performance={now:()=>now},finite=Number.isFinite,render=()=>{},toast=()=>{};
const calls=[],transport={command:async(...args)=>{calls.push(args);return {ok:true}}};
""" + command + hear + r"""
await hearLane('exit');assert.deepEqual(calls,[['test-tone',{lane:'exit'}]]);
for(const reason of ['stale','busy','offline','shutdown','preview','invalid']){
 now=1000;commandBusy=false;connectionOk=true;shutdownRequested=false;production=true;mode='live';
 if(reason==='stale')now=2800;if(reason==='busy')commandBusy=true;if(reason==='offline')connectionOk=false;
 if(reason==='shutdown')shutdownRequested=true;if(reason==='preview')production=false;
 await hearLane(reason==='invalid'?'pedestrian':'entrance');assert.equal(calls.length,1,reason);
}
""")

    def test_audio_stage_uses_base_profile_for_lane_specific_assets(self):
        helper = 'function audioStageLabel(' + self.source('function audioStageLabel(', 'function render(){')
        self.js(helper + """
assert.equal(audioStageLabel({profile:'entrance_soft',base_profile:'soft'}),'First warning');
assert.equal(audioStageLabel({profile:'exit_notice',base_profile:'notice'}),'Repeating warning');
assert.equal(audioStageLabel({profile:'exit_wake',base_profile:'wake'}),'Escalated alarm');
assert.equal(audioStageLabel({profile:'fault'}),'System warning');
assert.equal(audioStageLabel({profile:'limited'}),'Speaker control limited');
""")

    def test_live_overlay_draws_alert_gate_separately_from_tracking_polygon(self):
        path = 'function pathData(' + self.source('function pathData(', 'var framingGoodSince')
        overlay = 'function renderOverlay(' + self.source('function renderOverlay(', 'function hearLane(')
        self.js("""
let renderedOverlay='',showZones=true;const node={style:{},innerHTML:''},$=()=>node;
const finite=Number.isFinite,esc=String,routeRole=z=>z.route_role;
const roleMeta={entrance:{color:'orange'}};
const gate=[[.665,.480],[.820,.480],[.875,.525],[.875,.585],[.665,.585]];
const zone={name:'Entrance waiting',route_role:'entrance',polygon:[[.6,.4],[.9,.4],[.9,.6],[.6,.6]],confirmation_polygon:gate};
""" + path + overlay + """
renderOverlay({zones:[zone]},{aligned:true});
assert.match(node.innerHTML,/class="confirmation-zone"/);
assert.ok(node.innerHTML.includes(pathData(gate)));
assert.ok(node.innerHTML.includes(pathData(zone.polygon)));
assert.match(node.innerHTML,/Entrance waiting · alert area/);
renderOverlay({zones:[{...zone,confirmation_polygon:null}]},{aligned:true});
assert.doesNotMatch(node.innerHTML,/confirmation-zone/);
""")

    def test_native_editor_alert_gate_stays_locked_when_outline_changes(self):
        draw = 'function drawConfirmations(' + self.source('function drawConfirmations(', '  function draw()')
        self.js("""
const WIDTH=640,HEIGHT=360,validatePolygon=points=>points.length<3?'invalid':null;
const makeSvg=(tag,attributes)=>({tag,attributes,children:[],append(...nodes){this.children.push(...nodes)}});
const confirmations={children:[],replaceChildren(){this.children=[]},append(node){this.children.push(node)}};
const gate=[[.665,.480],[.820,.480],[.875,.525],[.875,.585],[.665,.585]];
let latestState={zones:[{name:'Entrance waiting',polygon:[[.6,.4],[.9,.4],[.9,.6]],confirmation_polygon:gate}]};
let draft=structuredClone(latestState.zones);const before=JSON.stringify(gate);
""" + draw + """
drawConfirmations();assert.equal(confirmations.children.length,1);
const first=confirmations.children[0];assert.equal(first.tag,'polygon');
assert.equal(first.attributes.class,'zone-confirmation');
assert.equal(first.children[0].textContent,'Entrance waiting · saved alert area (locked)');
assert.equal(first.attributes.tabindex,undefined);assert.equal(first.attributes['data-vertex'],undefined);
draft[0].polygon[0]=[.2,.2];drawConfirmations();
assert.equal(confirmations.children[0].attributes.points,first.attributes.points);
assert.equal(JSON.stringify(gate),before);
latestState={zones:[{name:'Entrance waiting'}]};drawConfirmations();assert.equal(confirmations.children.length,0);
for(const zones of [null,{},[null]]){latestState={zones};drawConfirmations();assert.equal(confirmations.children.length,0);}
""")
        self.assertIn('Gold dashes: saved alert area · locked', self.page)
        self.assertIn('.zone-confirmation{fill:none;', self.page)

    def test_diagnostics_distinguish_candidate_from_current_arm_state(self):
        self.js(self.helpers + r"""
const context={live:true,connected:true,cameraFresh:true,aligned:true,pictureFresh:true};
const arm={id:'exit',state:'unknown',calibrated:true,samples:{up:1,down:1},fresh:true,candidate_position:'up',confirmation_s:.25,match:.86,opposite_match:.33};
let row=barrierSnapshot({barriers:[arm]},context)[0];
assert.equal(row.state,'unknown');assert.match(barrierDiagnostic(row),/Checking UP.*0.3 s observed.*Match 0.86 \/ other 0.33/);
for(const patch of [{fresh:false},{candidate_position:'moving',confirmation_s:-1,match:'0.86',opposite_match:Infinity}]){
 row=barrierSnapshot({barriers:[{...arm,...patch}]},context)[0];assert.equal(barrierDiagnostic(row),'');
}
row=barrierSnapshot({barriers:[arm]},{...context,pictureFresh:false})[0];assert.equal(barrierDiagnostic(row),'');
""")

    def test_inspection_crop_uses_configured_own_arm_bounds_and_fresh_displayed_pixels(self):
        renderer = 'function renderBarrierInspection(' + self.source('function renderBarrierInspection(', 'function renderBarriers(')
        self.js(self.helpers + renderer + """
const original={lines:{up:[[.1,.1],[.1,.4]],down:[[.1,.4],[.4,.4]]}};
const box=barrierInspectionBox(original);assert.ok(box);assert.deepEqual(original.lines.up,[[.1,.1],[.1,.4]]);
assert.ok(box[0]<=.1&&box[1]<=.1&&box[0]+box[2]>=.4&&box[1]+box[3]>=.4);
assert.equal(box[2],box[3],'Keep 16:9 source proportions in the 16:9 canvas');
assert.equal(barrierInspectionBox({lines:{up:[[0,0],[1.2,1]],down:[[0,0],[1,1]]}}),null);
const draws=[],canvas={width:320,height:180,hidden:true,getContext:()=>({drawImage:(...args)=>draws.push(args)})},label={textContent:''};
const source={complete:true,naturalWidth:640,naturalHeight:360};
const $=id=>id==='feed'?source:id.endsWith('-label')?label:canvas;
const arm={id:'exit',name:'Exit',inspectionBox:box};
renderBarrierInspection(arm,true);assert.equal(canvas.hidden,false);assert.equal(draws.length,1);assert.equal(draws[0][0],source);
assert.match(label.textContent,/Exit arm · close-up/);assert.equal('src' in source,false);
renderBarrierInspection(arm,false);assert.equal(canvas.hidden,true);assert.equal(draws.length,1);
renderBarrierInspection({...arm,inspectionBox:null},true);assert.equal(canvas.hidden,true);assert.equal(draws.length,1);
source.complete=false;renderBarrierInspection(arm,true);assert.equal(canvas.hidden,true);assert.equal(draws.length,1);
""")

    def test_framing_unknown_and_held_do_not_present_a_zoom_instruction_or_new_fit(self):
        harness = view_status.DashboardViewStatusTests.framing_harness(self)
        self.js(harness + """
renderFraming({alignment:{framing:{state:'unknown',primary:'lost',headline:'ZOOM OUT',text:'Wrong camera',fit_pct:0}}},status);
assert.equal(box.hidden,false);assert.equal(box.children[1].children[0].textContent,'View alignment uncertain');
assert.equal(box.children[1].children.length,2,'No misleading 0% fit meter when the view cannot be measured');
box.children=[];
renderFraming({alignment:{framing:{state:'holding',primary:'held',retained:true,headline:'View held steady',text:'Landmarks briefly unclear',fit_pct:97}}},status);
assert.match(box.className,/held/);assert.doesNotMatch(box.className,/good/);
assert.equal(box.children[1].children.length,2);assert.doesNotMatch(box.children[1].children[1].textContent,/97|framing|%/);
""")

    def test_framing_coach_only_exists_in_camera_setup_and_warnings_do_not_cover_picture(self):
        class Structure(HTMLParser):
            def __init__(self):
                super().__init__(); self.stack=[]; self.ancestors={}
            def handle_starttag(self, tag, attrs):
                attrs=dict(attrs)
                if attrs.get('id'):
                    self.ancestors[attrs['id']]=[item[1] for item in self.stack]
                if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:
                    self.stack.append((tag,attrs.get('id')))
            def handle_endtag(self, tag):
                for index in range(len(self.stack)-1,-1,-1):
                    if self.stack[index][0]==tag:
                        del self.stack[index:];break
        parsed=Structure();parsed.feed(self.page)
        self.assertIn('setup-camera',parsed.ancestors['framing-coach'])
        self.assertNotIn('page-live',parsed.ancestors['framing-coach'])
        self.assertNotIn('camera-stage',parsed.ancestors['camera-warning'])
        for lane in ('entrance','exit'):
            self.assertIn('setup-barriers',parsed.ancestors[f'barrier-{lane}-inspection'])
        self.assertIn('this lane’s arm reaching UP or DOWN',self.page)
        self.assertNotIn('moving either arm (up or down)',self.page)

if __name__=='__main__':
    unittest.main()
