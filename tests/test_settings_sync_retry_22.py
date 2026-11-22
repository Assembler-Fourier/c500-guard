"""Partial saves stay visible and retry the desired value, never its inverse."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]


def literal(name, variable):
    tree=ast.parse((ROOT/'src'/name).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==variable for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(variable)


class SettingsSyncRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node=shutil.which('node')
        if not cls.node:raise unittest.SkipTest('Node.js is needed for command-receipt behavior tests')
        cls.page=literal('dashboard_ui.py','PAGE_TEMPLATE')
        cls.settings=literal('settings_ui.py','PAGE')

    def js(self, code):
        result=subprocess.run([self.node,'-e',"const assert=require('node:assert/strict');\n"+code],
                              capture_output=True,text=True,timeout=12,
                              creationflags=0x08000000 if __import__('os').name=='nt' else 0)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_transport_keeps_partial_receipt_without_claiming_success(self):
        part=self.page.split('// Module: runtime.js\n',1)[1].split('// Module: app.js\n',1)[0]
        self.js("const __c500Modules={};\n"+part+"""
(async()=>{const receipt={ok:false,partial:true,applied:true,error:'Saved here; retry other views',view_sync:{pending:{'barriers.reminder':false}}};
const t=__c500Modules['runtime.js'].createTransport({production:true,fetcher:async()=>({ok:true,json:async()=>receipt})});
await assert.rejects(t.command('settings',{changes:{'barriers.reminder':false}}),e=>e.receipt===receipt&&e.message===receipt.error);
})().catch(e=>{console.error(e);process.exitCode=1});
""")

    def test_dashboard_retry_preserves_applied_false_value_and_clears_on_success(self):
        part=self.page.split('async function runCommand(',1)[1].split('async function applyZones(',1)[0]
        wiring="$('settings-sync-retry').addEventListener"+self.page.split("$('settings-sync-retry').addEventListener",1)[1].split('\n',1)[0]
        self.js("""
let commandBusy=false,connectionOk=true,shutdownRequested=false,lastStateAt=performance.now(),pendingSettingsRetry=null,lastAckAt=0;
const liveState={barriers:{reminder:{enabled:true}}},calls=[],nodes=new Map();
const $=id=>{if(!nodes.has(id))nodes.set(id,{hidden:true,textContent:'',addEventListener(_e,fn){this.click=fn;}});return nodes.get(id);};
const render=()=>{},toast=()=>{},finite=Number.isFinite;
let fail=true;const transport={command:async(action,payload)=>{calls.push(JSON.parse(JSON.stringify(payload)));if(fail){const e=new Error('Partial save');e.receipt={partial:true,applied:true,view_sync:{pending:{'barriers.reminder':false}}};throw e;}return {ok:true};}};
"""+'async function runCommand('+part+'\n'+wiring+"""
(async()=>{await runCommand('settings',{changes:{'barriers.reminder':false}});assert.equal($('settings-sync-warning').hidden,false);
assert.deepEqual(pendingSettingsRetry,{changes:{'barriers.reminder':false}});
liveState.barriers.reminder.enabled=false;fail=false;await $('settings-sync-retry').click();
assert.deepEqual(calls,[{changes:{'barriers.reminder':false}},{changes:{'barriers.reminder':false}}]);
assert.equal(pendingSettingsRetry,null);assert.equal($('settings-sync-warning').hidden,true);
})().catch(e=>{console.error(e);process.exitCode=1});
""")

    def test_standalone_partial_save_keeps_local_value_and_exact_retry(self):
        helper='function showSettingsRetry('+self.settings.split('function showSettingsRetry(',1)[1].split('\n',1)[0]
        methods='async function post('+self.settings.split('async function post(',1)[1].split("$('settings-sync-retry').addEventListener",1)[0]
        wiring="$('settings-sync-retry').addEventListener"+self.settings.split("$('settings-sync-retry').addEventListener",1)[1].split('\n',1)[0]
        self.js("""
let pendingSettingsChanges=null,settingsRequestBusy=false,loadVersion=0,fail=true;const nodes=new Map(),calls=[];
const $=id=>{if(!nodes.has(id))nodes.set(id,{hidden:true,textContent:'',disabled:false,addEventListener(_e,fn){this.click=fn;}});return nodes.get(id);};
const toast=()=>{},load=async()=>{},updateControl=(item,row)=>row.value=item.value;
const fetch=async(path,options)=>{calls.push(JSON.parse(options.body));return {ok:true,json:async()=>fail?{ok:false,applied:true,partial:true,error:'Retry views',view_sync:{pending:{'barriers.reminder':false}}}:{ok:true}};};
"""+helper+'\n'+methods+'\n'+wiring+"""
(async()=>{const item={key:'barriers.reminder',value:true},input={disabled:false},classes=new Set(['saved']);
const row={querySelector:()=>input,classList:{contains:k=>classes.has(k),add:k=>classes.add(k),remove:k=>classes.delete(k)}};
await save(item,false,row);assert.equal(item.value,false);assert.equal(row.value,false);assert.equal(classes.has('saved'),false);
assert.equal($('settings-sync-warning').hidden,false);fail=false;await $('settings-sync-retry').click();
assert.deepEqual(calls,[{changes:{'barriers.reminder':false}},{changes:{'barriers.reminder':false}}]);
assert.equal(pendingSettingsChanges,null);assert.equal($('settings-sync-warning').hidden,true);
})().catch(e=>{console.error(e);process.exitCode=1});
""")

    def test_standalone_settings_requests_cannot_race_retry(self):
        methods='async function post('+self.settings.split('async function post(',1)[1].split('async function save(',1)[0]
        self.js("""
let pendingSettingsChanges={'barriers.reminder':false},settingsRequestBusy=false,resolve,requests=0;
const showSettingsRetry=v=>pendingSettingsChanges=v;
const fetch=()=>{requests++;return new Promise(done=>resolve=()=>done({ok:true,json:async()=>({ok:true,view_sync:{pending:{}}})}));};
"""+methods+"""
(async()=>{const first=post('/settings',{changes:{'barriers.reminder':false}});
await assert.rejects(post('/settings',{changes:{'barriers.reminder':true}}),/Another setting/);
assert.equal(requests,1);resolve();await first;assert.equal(settingsRequestBusy,false);
assert.deepEqual(pendingSettingsChanges,{});
})().catch(e=>{console.error(e);process.exitCode=1});
""")

    def test_dashboard_sync_snapshot_rehydrates_and_rejects_old_poll(self):
        functions='function renderSettingsRetry('+self.page.split('function renderSettingsRetry(',1)[1].split('async function applyZones(',1)[0]
        self.js("""
let pendingSettingsRetry=null;const nodes=new Map();const $=id=>{if(!nodes.has(id))nodes.set(id,{});return nodes.get(id);};
"""+functions+"""
applySettingsSync({revision:20,pending:{'barriers.reminder':false,'barriers.reminder_every_s':10}});
assert.equal($('settings-sync-warning').hidden,false);assert.equal(pendingSettingsRetry.changes['barriers.reminder'],false);
applySettingsSync({revision:19,pending:{}});assert.equal($('settings-sync-warning').hidden,false);
applySettingsSync({revision:21,pending:{}});assert.equal(pendingSettingsRetry,null);assert.equal($('settings-sync-warning').hidden,true);
""")


if __name__=='__main__':unittest.main()
