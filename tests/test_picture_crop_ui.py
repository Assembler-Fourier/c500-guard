"""Live picture trimming (21 Nov 2026): only margins that stayed black are trimmed, never more than a quarter per side,
and the outlines share the trimmed frame. Fake image data only."""
import unittest

import test_live_dashboard_freshness as freshness


class PictureCropTests(unittest.TestCase):
    js = freshness.LiveDashboardFreshnessTests.js

    @classmethod
    def setUpClass(cls):
        cls.page = freshness.page_source()

    def section(self):
        return self.page.split('// PICTURE_CROP_START', 1)[1].split('// PICTURE_CROP_END', 1)[0]

    def harness(self):
        return """
const w=160,h=90;
function picture(left,top,right,bottom,dark=false){
 const lum=new Uint8Array(w*h);
 for(let y=top;y<h-bottom;y++)for(let x=left;x<w-right;x++)lum[y*w+x]=dark?((x*7+y*3)%9===0?60:8):120;
 return lum;
}
const style=new Map(),document={documentElement:{style:{setProperty:(k,v)=>style.set(k,Number(v))}}};
let liveState={},performance={now:()=>0};const $=()=>({naturalWidth:0});
""" + self.section()

    def test_black_keyboard_margins_are_found(self):
        self.js(self.harness() + """
const box=cropBox(picture(2,16,10,1),w,h);
assert.deepEqual(box.map(v=>+v.toFixed(4)),[2/160,16/90,150/160,89/90].map(v=>+v.toFixed(4)));
const full=cropBox(picture(0,0,0,0),w,h);assert.deepEqual(full,[0,0,1,1]);
""")

    def test_trim_is_capped_and_dark_scenery_with_lights_is_kept(self):
        self.js(self.harness() + """
const huge=cropBox(picture(70,40,70,40),w,h);       // mostly black: never more than a quarter per side
assert.ok(huge[0]<=.26&&huge[1]<=.26&&huge[2]>=.74&&huge[3]>=.74);
const night=cropBox(picture(0,0,0,0,true),w,h);     // dark night picture with scattered lights stays whole
assert.deepEqual(night,[0,0,1,1]);
""")

    def test_only_margins_black_in_every_recent_check_are_trimmed(self):
        self.js(self.harness() + """
cropHistory=[[.05,.18,.95,1],[.05,.18,.95,1]];applyCrop();assert.equal(style.get('--crop-w'),640,'Wait for three checks');
cropHistory.push([0,.18,.95,1]);applyCrop();
assert.equal(style.get('--crop-x'),0,'A side that was lit once is kept');
assert.equal(style.get('--crop-y'),64.8);assert.equal(style.get('--crop-w'),608);assert.equal(style.get('--crop-h'),295.2);
assert.equal(style.get('--crop-aspect'),+(608/295.2).toFixed(4));
cropHistory=[[.01,0,1,1],[.01,0,1,1],[.01,0,1,1]];applyCrop();assert.equal(style.get('--crop-w'),640,'Tiny trims are not worth it');
""")

    def test_picture_and_outlines_share_the_trimmed_frame(self):
        frame = self.page.split('<div class="camera-frame">', 1)[1].split('</div>', 1)[0]
        for element in ('id="feed"', 'id="route-overlay"', 'id="barrier-overlay"', 'id="object-overlay"'):
            self.assertIn(element, frame)
        self.assertIn('.camera-stage:fullscreen .camera-frame{left:0;top:0;width:100%;height:100%}', self.page)

    def test_darkening_scene_cannot_keep_zooming_in_after_initial_fit(self):
        self.js(self.harness() + """
cropHistory=Array(3).fill([.01,.1,.94,1]);applyCrop();
const initial=[...style.entries()];
// All six recent samples now contain darker scenery around the edges. This
// used to shrink the view again, even though the physical camera never moved.
cropHistory=Array(6).fill([.07,.2,.87,.9]);applyCrop();
assert.deepEqual([...style.entries()],initial);
""")

    def test_new_visible_content_expands_without_later_cropping_it_away(self):
        self.js(self.harness() + """
cropHistory=Array(3).fill([.05,.18,.95,.98]);applyCrop();
cropHistory=Array(6).fill([0,.1,1,1]);applyCrop();
assert.equal(style.get('--crop-x'),0);assert.equal(style.get('--crop-w'),640);
assert.equal(style.get('--crop-y'),36);assert.equal(style.get('--crop-h'),324);
const expanded=[...style.entries()];
cropHistory=Array(6).fill([.05,.18,.95,.98]);applyCrop();
assert.deepEqual([...style.entries()],expanded);
""")

    def test_camera_or_resolution_change_starts_a_new_fit(self):
        harness = self.harness().replace(
            "let liveState={},performance={now:()=>0};const $=()=>({naturalWidth:0});",
            "let liveState={camera_view:'PTZ 2'},performance={now:()=>10};"
            "let img={naturalWidth:640,naturalHeight:360};const $=()=>img;")
        self.js(harness + """
cropView='PTZ 2:640x360';cropMeasuredAt=10;
cropHistory=Array(3).fill([.05,.1,.95,1]);applyCrop();
assert.equal(style.get('--crop-w'),576);
liveState.camera_view='ANPR Entrance';measurePictureCrop();
assert.equal(style.get('--crop-w'),640);assert.equal(cropBounds,null);
cropHistory=Array(3).fill([0,.05,1,1]);applyCrop();
assert.equal(style.get('--crop-h'),342);
img.naturalWidth=1280;img.naturalHeight=720;measurePictureCrop();
assert.equal(style.get('--crop-h'),360);assert.equal(cropBounds,null);
""")


if __name__ == '__main__':
    unittest.main()
