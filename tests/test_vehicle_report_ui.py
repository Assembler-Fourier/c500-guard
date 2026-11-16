"""Pure report presentation checks; no real sightings or identity data."""
import importlib.util
import json
from pathlib import Path
from html.parser import HTMLParser
import unittest

import test_live_dashboard_freshness as freshness

spec = importlib.util.spec_from_file_location('vehicle_report_template', Path(__file__).parents[1] / 'car-memory/report_template.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class Nodes(HTMLParser):
    def __init__(self):
        super().__init__(); self.cards=[]; self.tags=[]; self.attrs=[]

    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs); self.tags.append(tag); self.attrs.append(attrs)
        if attrs.get('class')=='vehicle-card':self.cards.append(attrs)


class VehicleReportUiTests(unittest.TestCase):
    js = freshness.LiveDashboardFreshnessTests.js

    def row(self, ident, direction='unclear', **values):
        return dict(id=ident, direction=direction, at=1700000000+ident, type='car', colour='grey', night=False, colourless=False, thumb=b'fake jpeg', clip='live:private-path', **values)

    def test_all_unclear_and_paired_sightings_are_visible_once_in_newest_order(self):
        rows=[self.row(i) for i in range(18)] + [self.row(20,'in'),self.row(21,'out')]
        pairs=[{'out':21,'in':20,'band':'strong','score':.91,'margin':.2,'light_differs':False}]
        page=report.render_report(rows,pairs,1700001000,12,'approved')
        parsed=Nodes();parsed.feed(page)
        self.assertEqual([int(card['data-sighting']) for card in parsed.cards],[21,20]+list(range(17,-1,-1)))
        self.assertEqual(sum(card['data-direction']=='unclear' for card in parsed.cards),18)
        self.assertIn('Stronger appearance suggestion',page)
        self.assertNotIn('Same vehicle (strong)',page)
        self.assertNotIn('private-path',page)
        self.assertIn('not an identity probability',page)
        self.assertIn('12 hour retention',page)

    def test_text_is_escaped_and_photo_payload_is_local(self):
        row=self.row(1);row.update(type='<script>alert(1)</script>', learned_colour='" onerror="bad', night=True, colour='grey')
        page=report.render_report([row],[],1700001000,6,'<img src=x onerror=bad>')
        parsed=Nodes();parsed.feed(page)
        self.assertEqual(parsed.tags.count('script'),1,'Only the fixed local application script')
        self.assertNotIn('onerror', ' '.join(' '.join(attrs.keys()) for attrs in parsed.attrs))
        self.assertIn('grey &lt;script&gt;',page)
        self.assertIn('Night · colour unsure',page)
        self.assertIn('6 hour retention',page)
        self.assertNotIn('src="http',page)

    def test_script_parses_and_refresh_requires_parent_same_origin(self):
        self.js('new Function('+json.dumps(report.SCRIPT)+');')
        self.assertIn("event.source!==window.parent||event.origin!==location.origin",report.SCRIPT)
        self.assertIn("$('vehicle-list').contains(document.activeElement)",report.SCRIPT)
        self.assertIn("window.scrollTo(scrollX,scrollY)",report.SCRIPT)
        self.assertNotIn('location.reload',report.SCRIPT)
        self.assertNotIn('setInterval',report.SCRIPT)

    def test_filter_and_direction_grouping_keep_unclear_cards_accessible(self):
        function='function applyFilters(){'+report.SCRIPT.split('function applyFilters(){',1)[1].split("$('search').addEventListener",1)[0]
        self.js("""
const cards=[{direction:'in',search:'grey car entrance'},{direction:'out',search:'white van exit'},{direction:'unclear',search:'grey bus unclear'}].map(dataset=>({dataset,style:{},hidden:false}));
const list={querySelectorAll:selector=>selector==='.vehicle-card'?cards:[],append:()=>{}};
const nodes={'vehicle-list':list,empty:{},'result-count':{}},$=id=>nodes[id];
const document={querySelectorAll:()=>[],createElement:()=>({style:{}})},sessionStorage={setItem:()=>{}};
const prefs={direction:'all',query:'',grouping:'recent'};
"""+function+"""
applyFilters();assert.equal(cards.filter(c=>!c.hidden).length,3);
prefs.direction='unclear';applyFilters();assert.deepEqual(cards.map(c=>c.hidden),[true,true,false]);
prefs.direction='all';prefs.query='grey';applyFilters();assert.deepEqual(cards.map(c=>c.hidden),[false,true,false]);
prefs.query='';prefs.grouping='direction';applyFilters();assert.deepEqual(cards.map(c=>c.style.order),['100000','200001','300002']);
prefs.query='not present';applyFilters();assert.equal(nodes.empty.hidden,false);assert.equal(nodes['result-count'].textContent,'0 of 3 observations');
""")

    def test_report_refresh_defers_focused_results_and_preserves_toolbar_and_scroll(self):
        refresh='let requestedVersion='+report.SCRIPT.split('let requestedVersion=',1)[1].split("$('vehicle-list').addEventListener('focusout'",1)[0]
        self.js("""
const toolbar={value:'grey'},oldCard={dataset:{sighting:'1'},querySelector:()=>null};
let focusedResult=true,fetches=0,filters=0,scrolls=[];
const list={children:[oldCard],querySelectorAll:()=>[],contains:()=>focusedResult,replaceChildren(...nodes){this.children=nodes}};
const summary={replaceChildren(...nodes){this.children=nodes}},metadata={replaceChildren(...nodes){this.children=nodes}};
const nodes={'vehicle-list':list,summary,'report-metadata':metadata,'photo-dialog':{open:false},'refresh-note':{hidden:true}},$=id=>nodes[id];
const document={activeElement:toolbar},window={scrollX:0,scrollY:550,scrollTo:(...args)=>scrolls.push(args)};
const applyFilters=()=>filters++;
const replacement={dataset:{sighting:'2'},querySelector:()=>null};
const next={'vehicle-list':{childNodes:[replacement]},summary:{childNodes:['counts']},'report-metadata':{childNodes:['new time']}};
class DOMParser{parseFromString(){return {getElementById:id=>next[id]}}}
const fetch=async()=>{fetches++;return {ok:true,text:async()=>'<report>'}};
"""+refresh+"""
requestedVersion='new';await refreshReport();assert.equal(list.children[0],oldCard);assert.equal(appliedVersion,'');
focusedResult=false;await refreshReport();assert.equal(list.children[0],replacement);assert.equal(appliedVersion,'new');
assert.equal(document.activeElement,toolbar);assert.equal(toolbar.value,'grey');assert.deepEqual(scrolls,[[0,550]]);assert.equal(filters,1);
await refreshReport();assert.equal(fetches,2,'An already applied version is not downloaded again');
requestedVersion='newer';nodes['photo-dialog'].open=true;await refreshReport();assert.equal(fetches,2,'An open photo remains stable');
""")


if __name__=='__main__':
    unittest.main()
