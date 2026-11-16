"""Settings save isolation: fake transport and DOM, no live configuration writes."""
import unittest
from settings_ui import PAGE
import test_live_dashboard_freshness as freshness


class SettingsInteractionTests(unittest.TestCase):
    js = freshness.LiveDashboardFreshnessTests.js

    def harness(self):
        shown = PAGE.split('function shown(', 1)[1].split('function rowFor(', 1)[0]
        update = PAGE.split('function updateControl(', 1)[1].split('async function load(', 1)[0]
        save = PAGE.split('async function save(', 1)[1].split("$('test').addEventListener", 1)[0]
        return r'''
let loadVersion=0,loads=0,calls=[],messages=[],pending=[];
const load=()=>{loads++},toast=(...args)=>messages.push(args),setTimeout=()=>0;
const post=(path,body)=>{calls.push([path,body]);return new Promise((resolve,reject)=>pending.push({resolve,reject}))};
function row(value){const classes=new Set();const input={value,disabled:false,attributes:{},setAttribute(k,v){this.attributes[k]=v}};const display={textContent:''};return {input,display,classList:{add:k=>classes.add(k),remove:k=>classes.delete(k),contains:k=>classes.has(k)},querySelector:s=>s==='.value'?display:input}}
''' + 'function shown(' + shown + 'function updateControl(' + update + 'async function save(' + save

    def test_save_does_not_replace_other_in_progress_edit(self):
        self.js(self.harness() + r'''
const first={key:'alarms.start_level',kind:'percent',value:.1},second={key:'display.name',kind:'text',value:'Campus'};
const a=row(20),b=row('An unfinished name');const aInput=a.input,bInput=b.input;
const saving=save(first,.2,a);assert.equal(a.input.disabled,true);
pending[0].resolve({ok:true});await saving;
assert.equal(first.value,.2);assert.equal(a.display.textContent,'20%');
assert.equal(b.input.value,'An unfinished name');assert.strictEqual(a.input,aInput);assert.strictEqual(b.input,bInput);
assert.equal(loads,0);assert.equal(a.input.disabled,false);
assert.deepEqual(calls,[['/settings',{changes:{'alarms.start_level':.2}}]]);
''')

    def test_same_row_repeated_save_is_ignored_while_request_pending(self):
        self.js(self.harness() + r'''
const item={key:'barriers.ack_on_lower',kind:'bool',value:false},r=row('');
const one=save(item,true,r);await save(item,false,r);assert.equal(calls.length,1);
pending[0].resolve({ok:true});await one;
assert.equal(item.value,true);assert.equal(r.input.attributes['aria-checked'],'true');
''')

    def test_refused_save_restores_only_its_last_confirmed_value(self):
        self.js(self.harness() + r'''
const item={key:'alarms.start_level',kind:'percent',value:.2},r=row(45);
const saving=save(item,.45,r);pending[0].reject(new Error('Rejected'));await saving;
assert.equal(item.value,.2);assert.equal(r.input.value,20);assert.equal(r.display.textContent,'20%');
assert.deepEqual(messages.at(-1),['Rejected',true]);assert.equal(r.input.disabled,false);assert.equal(loads,0);
''')

    def test_two_different_saves_complete_out_of_order_without_reloading(self):
        self.js(self.harness() + r'''
const a={key:'alarms.start_level',kind:'percent',value:.1},b={key:'alarms.sound_theme',kind:'choice',value:'low'};
const ar=row(20),br=row('pulses');const sa=save(a,.2,ar),sb=save(b,'pulses',br);
pending[1].resolve({ok:true});await sb;pending[0].resolve({ok:true});await sa;
assert.equal(a.value,.2);assert.equal(b.value,'pulses');assert.equal(loads,0);assert.equal(calls.length,2);
''')


    def test_transport_requires_explicit_receipt_and_bounds_wait(self):
        post = PAGE.split('async function post(', 1)[1].split('async function save(', 1)[0]
        self.js(r'''let callback,cleared=0,lastRequest;const setTimeout=(fn,ms)=>{assert.equal(ms,10000);callback=fn;return 1},clearTimeout=()=>cleared++;
let response={ok:true,status:200,json:async()=>({})};const fetch=async(path,options)=>{lastRequest=options;return response};
'''+'async function post('+post+r'''
await assert.rejects(post('/settings',{}),/did not confirm/);assert.equal(cleared,1);assert.ok(lastRequest.signal);
response={ok:true,status:200,json:async()=>({ok:true})};assert.equal((await post('/settings',{})).ok,true);assert.equal(cleared,2);
response={ok:false,status:400,json:async()=>({ok:false,error:'Invalid value'})};await assert.rejects(post('/settings',{}),/Invalid value/);
''')

    def test_response_body_timeout_keeps_uncertain_result_warning(self):
        post = PAGE.split('async function post(', 1)[1].split('async function save(', 1)[0]
        self.js(r'''let cleared=0;const setTimeout=()=>1,clearTimeout=()=>cleared++;
const fetch=async()=>({ok:true,status:200,json:async()=>{const e=new Error('aborted');e.name='AbortError';throw e}});
'''+'async function post('+post+r'''
await assert.rejects(post('/settings',{}),/may have applied; check this setting before retrying/);assert.equal(cleared,1);
''')

    def test_text_edit_matches_backend_whitespace_canonicalisation(self):
        code = PAGE.split("else if(item.kind==='text')input.addEventListener('change',",1)[1].split(');\n else{',1)[0]
        self.js("const input={value:'  Gate   House  '},item={value:'Campus'},row={};let sent;const save=(item,v,row)=>sent=v,toast=()=>{};("+code+")();assert.equal(sent,'Gate House');")

if __name__ == '__main__':
    unittest.main()
