"""Current view and framing advice with fake browser nodes, clocks and state only."""
import unittest

import test_live_dashboard_freshness as freshness


class DashboardViewStatusTests(unittest.TestCase):
    source = freshness.LiveDashboardFreshnessTests.source
    js = freshness.LiveDashboardFreshnessTests.js

    @classmethod
    def setUpClass(cls):
        cls.page = freshness.page_source()

    def framing_harness(self):
        context = "function barrierContext(" + self.source("function barrierContext(", "function renderBarriers(")
        framing = "var framingGoodSince" + self.source("var framingGoodSince", "var renderedObjects=")
        return """
let now=1000,mode='live',connectionOk=true,shutdownRequested=false,lastStateAt=1000,lastFrameAt=1000,displayedFrame='blob:live';
const performance={now:()=>now},Date={now:()=>now};
class Element{constructor(){this.hidden=false;this.textContent='';this.children=[];this.style={};}append(...nodes){this.children.push(...nodes);}}
const box=new Element(),$=()=>box,document={createElement:()=>new Element()};
const good={camera_view:'PTZ 2',alignment:{framing:{primary:'good',headline:'GOOD - leave it here',text:'The zones fit',fit_pct:94}}};
const adjust={camera_view:'PTZ 2',alignment:{framing:{primary:'zoom_in',headline:'ZOOM IN',text:'The picture is too wide',fit_pct:44}}};
const status={cameraFresh:true,aligned:true};
""" + context + framing

    def test_framing_advice_requires_a_current_picture_and_connection(self):
        self.js(self.framing_harness() + """
displayedFrame=null;renderFraming(good,status);assert.equal(box.hidden,true,'No green advice before the picture arrives');
displayedFrame='blob:live';renderFraming(good,status);assert.equal(box.hidden,false);
connectionOk=false;renderFraming(adjust,status);assert.equal(box.hidden,true,'No cached zoom advice while disconnected');
connectionOk=true;now=2800;renderFraming(good,status);assert.equal(box.hidden,true,'A stale state expires before the timer redraw');
lastStateAt=now;lastFrameAt=now-4500;renderFraming(adjust,status);assert.equal(box.hidden,true,'No advice on an expired displayed picture');
lastFrameAt=now;renderFraming(good,{cameraFresh:false,aligned:false});assert.equal(box.hidden,true);
renderFraming(good,{cameraFresh:true,aligned:false});assert.equal(box.hidden,true,'No GOOD banner when alignment is lost');
renderFraming(adjust,{cameraFresh:true,aligned:false});assert.equal(box.hidden,false,'Fresh adjustment advice still works while aligning');
shutdownRequested=true;renderFraming(adjust,status);assert.equal(box.hidden,true);
""")

    def test_good_banner_gets_a_fresh_display_period_after_signal_recovers(self):
        self.js(self.framing_harness() + """
renderFraming(good,status);assert.equal(box.hidden,false);
now=17000;lastStateAt=lastFrameAt=now;renderFraming(good,status);assert.equal(box.hidden,true);
renderFraming({},status);assert.equal(box.hidden,true);
now=18000;lastStateAt=lastFrameAt=now;renderFraming(good,status);
assert.equal(box.hidden,false,'Newly recovered advice must get its full display period');
now=34000;lastStateAt=lastFrameAt=now;renderFraming(good,status);assert.equal(box.hidden,true);
mode='paused';renderFraming(good,status);assert.equal(box.hidden,true);
mode='live';renderFraming(good,status);assert.equal(box.hidden,false);
""")

    def test_camera_label_uses_the_current_view_and_never_an_anpr_constant(self):
        self.assertIn('id="camera-view-name"', self.page)
        self.assertNotIn('<strong>ANPR Entrance</strong>', self.page)
        label = "function cameraViewLabel(" + self.source("function cameraViewLabel(", "function render(){")
        self.js("""
let mode='live',connectionOk=true,shutdownRequested=false,now=1000,lastStateAt=1000;
const performance={now:()=>now};const fresh={cameraFresh:true};
""" + label + """
assert.equal(cameraViewLabel({camera_view:'PTZ 2'},fresh),'PTZ 2');
assert.equal(cameraViewLabel({camera_view:'ANPR Entrance',alignment:{view:'PTZ 2'}},fresh),'ANPR Entrance');
assert.equal(cameraViewLabel({alignment:{view:'PTZ 2'}},fresh),'PTZ 2');
assert.equal(cameraViewLabel({},fresh),'Camera view unconfirmed');
connectionOk=false;assert.equal(cameraViewLabel({camera_view:'PTZ 2'},fresh),'Camera view unconfirmed');
connectionOk=true;now=2800;assert.equal(cameraViewLabel({camera_view:'PTZ 2'},fresh),'Camera view unconfirmed');
lastStateAt=now;assert.equal(cameraViewLabel({camera_view:'PTZ 2'},{cameraFresh:false}),'Camera view unconfirmed');
mode='alarm';assert.equal(cameraViewLabel({camera_view:'PTZ 2'},fresh),'Preview camera');
""")


if __name__ == '__main__':
    unittest.main()
