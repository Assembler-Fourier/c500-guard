"""Redesign freshness/lifecycle contracts use only fake DOM and transport."""
import unittest
import test_live_dashboard_freshness as freshness
import test_dashboard_view_status as view_status


class DashboardRedesignTests(unittest.TestCase):
    source = freshness.LiveDashboardFreshnessTests.source
    js = freshness.LiveDashboardFreshnessTests.js

    @classmethod
    def setUpClass(cls):
        cls.page = freshness.page_source()

    def vehicles(self):
        return self.page.split('// VEHICLE_STATUS_START', 1)[1].split('// VEHICLE_STATUS_END', 1)[0].split('\n', 1)[1]

    def test_old_report_with_current_worker_is_not_reported_stale(self):
        self.js(self.vehicles() + """
const current={available:true,age_s:86400,updater_alive:true,updater_state:'ready',sightings_enabled:true,stale:false};
assert.equal(vehicleStatusPresentation(current).tone,'good');
for(const patch of [{sightings_enabled:false},{sightings_error:true},{available:false},{updater_alive:false},{stale:true},{updater_state:'error'}])assert.equal(vehicleStatusPresentation({...current,...patch}).tone,'warn');
assert.match(vehicleStatusPresentation({...current,model_state:'classical'}).detail,/Image matching/);
assert.equal(vehicleStatusPresentation(null).tone,'warn');
""")

    def test_vehicle_refresh_is_visible_bounded_serial_and_does_not_reload_frame(self):
        self.js("""
let active=true,page='vehicles',calls=0;const document={hidden:false},location={origin:'http://local'};
const timers=new Map();let next=0;const setTimeout=(fn,ms)=>{timers.set(++next,{fn,ms});return next},clearTimeout=id=>timers.delete(id);
const nodes=new Map(),messages=[];const $=id=>{if(!nodes.has(id))nodes.set(id,{contentWindow:{postMessage:(...args)=>messages.push(args)}});return nodes.get(id)};
const good={available:true,updater_alive:true,updater_state:'ready',stale:false,report_version:'1'};
let resolve;const transport={read:(path,signal)=>{assert.equal(path,'/api/car-memory/status');calls++;return new Promise(r=>resolve=r)}};
""" + self.vehicles() + """
const first=readVehicleStatus();await readVehicleStatus();assert.equal(calls,1);
assert.equal([...timers.values()].filter(t=>t.ms===4000).length,1);
resolve(good);await first;assert.equal(messages.length,1);assert.equal(messages[0][1],location.origin);
assert.equal([...timers.values()].filter(t=>t.ms===5000).length,1);assert.equal('src' in $('cars-frame'),false);
let request=readVehicleStatus();resolve(good);await request;assert.equal(messages.length,1,'Unchanged reports must not refresh');
request=readVehicleStatus();resolve({...good,report_version:'2'});await request;assert.equal(messages.length,2);
request=readVehicleStatus();stopVehicleStatus();page='live';resolve({...good,report_version:'3'});await request;
assert.equal(messages.length,2,'A response after navigation cannot touch the report');assert.equal(timers.size,0);
await readVehicleStatus();assert.equal(calls,4);page='vehicles';document.hidden=true;await readVehicleStatus();assert.equal(calls,4);
""")

    def test_camera_setup_keeps_fresh_good_guidance_visible_and_hides_stale(self):
        harness = view_status.DashboardViewStatusTests.framing_harness(self)
        self.js("let page='setup',setupTab='camera';\n" + harness + """
const currentSetup={alignment:{framing:{state:'good',primary:'good',headline:'Camera framing good',text:'Current view',fit_pct:98}}};
renderFraming(currentSetup,status);now+=16000;lastStateAt=lastFrameAt=now;renderFraming(currentSetup,status);assert.equal(box.hidden,false);
connectionOk=false;renderFraming(currentSetup,status);assert.equal(box.hidden,true);
""")
        # The live warning bar links straight to Setup > Camera.
        self.assertIn('data-go="setup" data-setup="camera"', self.page)

    def test_setup_sections_are_accessible_tabs(self):
        # The barrier drawer became Setup > Barriers: tabs announce their selection and own their panels.
        for tab in ('camera', 'zones', 'barriers'):
            self.assertRegex(self.page, r'role="tab" data-setup-tab="%s"[^>]*aria-selected="(?:true|false)"' % tab)
            self.assertRegex(self.page, r'<section id="setup-%s" class="setup-panel" role="tabpanel"' % tab)
        choose = 'function chooseSetupTab(' + self.source('function chooseSetupTab(', 'function navigate(')
        self.assertIn("b.setAttribute('aria-selected',String(on))", choose)
        self.assertIn("editor.deactivate()", choose)

    def test_setup_preview_reuses_current_frame_only_while_system_visible(self):
        context='function barrierContext('+self.source('function barrierContext(', 'function renderBarrierInspection(')
        preview='function renderSetupPreview('+self.source('function renderSetupPreview(', '// SETUP_PREVIEW_END')
        self.js("""
let page='setup',setupTab='camera',setupPreviewMode='aligned',mode='live',connectionOk=true,shutdownRequested=false,now=1000,lastStateAt=1000,lastFrameAt=1000,displayedFrame='blob:current';const setupVisible=tab=>page==='setup'&&setupTab===tab;
const performance={now:()=>now},nodes=new Map();
const $=id=>{if(!nodes.has(id))nodes.set(id,{src:null,innerHTML:'',hidden:false,textContent:'',attributes:new Set(['hidden']),toggleAttribute(name,value){if(value)this.attributes.add(name);else this.attributes.delete(name)},getAttribute(name){return this[name]},removeAttribute(name){delete this[name]}});return nodes.get(id)};
$('route-overlay').innerHTML='<path d="saved geometry"/>';$('feed').src='blob:main';
const state={},status={cameraFresh:true,aligned:true};
"""+context+preview+"""
renderSetupPreview(state,status);assert.equal($('setup-feed').src,'blob:current');assert.equal($('setup-feed').hidden,false);
assert.equal($('setup-overlay').innerHTML,$('route-overlay').innerHTML);assert.equal($('feed').src,'blob:main','Main image ownership is untouched');
assert.equal($('setup-overlay').attributes.has('hidden'),false,'SVG visibility must change its attribute, not an HTMLElement-only property');
page='live';renderSetupPreview(state,status);assert.equal($('setup-feed').hidden,true);assert.equal($('setup-feed').src,undefined);assert.equal($('setup-overlay').innerHTML,'');
assert.equal($('setup-overlay').attributes.has('hidden'),true);
displayedFrame='blob:newer';renderSetupPreview(state,status);assert.equal($('setup-feed').src,undefined,'No hidden preview updates');
page='setup';renderSetupPreview(state,{cameraFresh:true,aligned:false});assert.equal($('setup-feed').src,'blob:newer');assert.match($('setup-preview-caption').textContent,/alignment needs attention/);
$('route-overlay').innerHTML='';renderSetupPreview(state,status);assert.equal($('setup-overlay').attributes.has('hidden'),true);assert.match($('setup-preview-caption').textContent,/outlines unavailable/);
for(const bad of ['state','picture','offline']){
 now=1000;lastStateAt=lastFrameAt=1000;connectionOk=true;mode='live';
 if(bad==='state')now=2800;if(bad==='picture')lastFrameAt=-4000;if(bad==='offline')connectionOk=false;
 renderSetupPreview(state,status);assert.equal($('setup-feed').hidden,true,bad);assert.equal($('setup-feed').src,undefined,bad);assert.equal($('setup-overlay').innerHTML,'',bad);
}
""")
        for operation in ('fetch(', 'setInterval(', 'setTimeout(', 'new Image', 'createObjectURL', 'revokeObjectURL'):
            self.assertNotIn(operation,preview)
        frame='async function readFrame('+self.source('async function readFrame(', "document.querySelectorAll('[data-page]')")
        self.assertIn("if(page==='setup'&&setupTab==='camera'){renderSetupPreview(s,status);renderFraming(s,status);}",frame)
        self.assertEqual(frame.count('fetch('),1,'One shared frame request, never another camera reader')

    def test_system_preview_and_guidance_share_picture_expiry(self):
        harness=view_status.DashboardViewStatusTests.framing_harness(self)
        self.js("let page='setup',setupTab='camera';\n"+harness+"""
renderFraming(good,status);assert.equal(box.hidden,false);
now+=5000;lastStateAt=now;renderFraming(good,status);assert.equal(box.hidden,true,'Fresh state cannot renew an expired displayed picture');
lastFrameAt=now;renderFraming(good,status);assert.equal(box.hidden,false);
""")
        self.assertIn('id="setup-feed"',self.page)
        self.assertIn('id="setup-overlay" viewBox="0 0 640 360" preserveAspectRatio="xMidYMid meet"',self.page)


if __name__ == '__main__':
    unittest.main()
