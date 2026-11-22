"""Read-only Cars health and real command contracts, with scratch files/fake HTTP."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_watch import watch
import test_live_dashboard_freshness as freshness


class CarsStatusContractTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.folder = self.root / 'car-memory'
        self.folder.mkdir()
        patcher = mock.patch.object(watch, 'STATE_DIR', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.now = 100000.
        self.watcher = SimpleNamespace(sightings=SimpleNamespace(enabled=True, last_error=''))

    def report(self, age=5., text='<html>synthetic vehicle report</html>'):
        path = self.folder / 'report.html'
        path.write_text(text, encoding='utf-8')
        os.utime(path, (self.now-age, self.now-age))
        return path

    def heartbeat(self, **changes):
        data = dict(updated_at=self.now-1, state='ready', last_success_at=self.now-30,
                    model_state='classical')
        data.update(changes)
        (self.folder/'worker-status.json').write_text(json.dumps(data), encoding='utf-8')

    def status(self):
        return watch.car_memory_status(self.watcher, now=self.now)

    def test_missing_report_and_worker_are_unavailable_without_writing(self):
        state = self.status()
        self.assertFalse(state['available'])
        self.assertTrue(state['stale'])
        self.assertFalse(state['updater_alive'])
        self.assertEqual([], list(self.folder.iterdir()))

    def test_old_report_with_healthy_idle_worker_is_not_mistaken_for_stopped_updater(self):
        self.report(age=3600.)
        self.heartbeat()
        state = self.status()
        self.assertTrue(state['available'])
        self.assertFalse(state['stale'])
        self.assertEqual(3600., state['age_s'])
        self.assertEqual('classical', state['model_state'])
        self.assertEqual(self.now-30, state['last_success_at'])

    def test_dead_future_invalid_and_error_heartbeats_never_claim_current(self):
        self.report()
        for changes in (dict(updated_at=self.now-90), dict(updated_at=self.now+1),
                        dict(updated_at=None), dict(updated_at=True),
                        dict(updated_at='99999'), dict(updated_at=float('nan')),
                        dict(updated_at=float('inf')), dict(state='error'),
                        dict(state='stopped'), dict(state='other')):
            with self.subTest(changes=changes):
                self.heartbeat(**changes)
                self.assertTrue(self.status()['stale'])
        self.heartbeat(updated_at=self.now-89.9, state='updating')
        self.assertFalse(self.status()['stale'])

    def test_disabled_or_failed_sighting_capture_is_explicit_and_private(self):
        self.report()
        self.heartbeat()
        self.watcher.sightings.enabled = False
        self.assertTrue(self.status()['stale'])
        self.assertFalse(self.status()['sightings_enabled'])
        self.watcher.sightings.enabled = True
        self.watcher.sightings.last_error = 'PRIVATE_PATH and a private vehicle identifier'
        state = self.status()
        self.assertTrue(state['stale'])
        self.assertTrue(state['sightings_error'])
        self.assertNotIn('PRIVATE', json.dumps(state))

    def test_metadata_read_never_reads_report_or_returns_vehicle_content(self):
        self.report(text='<html>PRIVATE_VEHICLE_CONTENT</html>')
        self.heartbeat()
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == 'report.html':
                self.fail('Status must only stat the report, never read it')
            return original_open(path, *args, **kwargs)
        with mock.patch.object(Path, 'open', guarded):
            state = self.status()
        self.assertNotIn('PRIVATE', json.dumps(state))
        self.assertTrue(state['available'])

    def test_malformed_and_oversized_worker_metadata_is_bounded_and_preserves_report(self):
        report = self.report()
        before = report.read_bytes()
        for raw in (b'null', b'[]', b'{broken', b'\xff', b' '*4097):
            with self.subTest(raw=raw[:20]):
                (self.folder/'worker-status.json').write_bytes(raw)
                state = self.status()
                self.assertTrue(state['stale'])
                self.assertEqual('unknown', state['updater_state'])
                self.assertEqual(before, report.read_bytes())

    def test_version_changes_only_when_report_changes_not_with_heartbeat(self):
        report = self.report()
        self.heartbeat()
        first = self.status()['report_version']
        self.heartbeat(updated_at=self.now)
        self.assertEqual(first, self.status()['report_version'])
        report.write_text('<html>another synthetic report</html>', encoding='utf-8')
        os.utime(report, (self.now-1, self.now-1))
        self.assertNotEqual(first, self.status()['report_version'])

    def test_empty_future_or_directory_report_is_unavailable(self):
        self.heartbeat()
        for age, text in ((-1., '<html>future</html>'), (0., '')):
            self.report(age=age, text=text)
            self.assertFalse(self.status()['available'])
        (self.folder/'report.html').unlink()
        (self.folder/'report.html').mkdir()
        self.assertFalse(self.status()['available'])

    def request(self, path, host='127.0.0.1:8787'):
        class FakeServer:
            def __init__(self, address, handler):
                self.RequestHandlerClass = handler
            def serve_forever(self):
                pass
        with mock.patch.object(watch, 'ThreadingHTTPServer', FakeServer), \
                mock.patch.object(watch.threading, 'Thread'):
            server = watch.start_web(self.watcher, 0)
        handler = object.__new__(server.RequestHandlerClass)
        handler.path, handler.headers = path, {'Host': host}
        result = []
        handler._send = lambda code, content_type, data: result.append((code, content_type, data))
        handler.do_GET()
        return result[0]

    def test_route_is_local_and_exact_with_read_only_query_support(self):
        self.report()
        self.heartbeat()
        with mock.patch.object(watch.time, 'time', return_value=self.now):
            code, content_type, body = self.request('/car-memory/status?t=123')
            self.assertEqual((200, 'application/json'), (code, content_type))
            self.assertFalse(json.loads(body)['stale'])
        for path in ('/car-memory/status/extra', '/car-memory/status/../report.html',
                     '/car-memory/%73tatus', '/car-memory/status%3fanything'):
            self.assertEqual(404, self.request(path)[0])
        self.assertEqual(403, self.request('/car-memory/status', host='external.invalid')[0])
        self.assertEqual(200, self.request('/car-memory?t=123')[0])


class UiControlReceiptContractTests(unittest.TestCase):
    js = freshness.LiveDashboardFreshnessTests.js

    def test_live_lane_updates_preserve_focused_controls_and_bind_once(self):
        page = freshness.page_source()
        renderer = ('function renderRoutes(' + page.split('function renderRoutes(', 1)[1]
                    .split('function highlightRoute(', 1)[0])
        self.js(r'''
let production=true,commandBusy=false,renderedRoutes='',mounts=0,connected=true;
const roleMeta={entrance:{name:'Entrance'},exit:{name:'Exit'},pedestrian:{name:'Footpath'}};
const nodes=new Map(),routes=[],selected=[];
const document={activeElement:null,querySelectorAll:q=>q==='.lane-top'?routes:[]};
function node(id,dataset={}){const n={id,dataset,isConnected:true,textContent:'',disabled:false,listeners:{},
 addEventListener(k,fn){(this.listeners[k]??=[]).push(fn)},focus(){document.activeElement=this}};nodes.set(id,n);return n;}
node('hear-entrance');node('hear-exit');     // static Health page controls, never remounted by the lanes
const cards=node('route-cards');Object.defineProperty(cards,'innerHTML',{set(html){
 mounts++;for(const [id,n] of nodes){if(['route-cards','hear-entrance','hear-exit'].includes(id))continue;n.isConnected=false;nodes.delete(id);if(document.activeElement===n)document.activeElement=null;}
 routes.length=0;for(const match of html.matchAll(/id="([^"]+)"/g))node(match[1]);
 for(const role of Object.keys(roleMeta))routes.push(node('route-'+role,{route:role}));
}});
const $=id=>{assert.ok(nodes.has(id),'Mounted node '+id);return nodes.get(id)};
const icon=()=>'',routeRole=z=>z.route_role,highlightRoute=()=>{},navigate=(p,t)=>selected.push(t),editor={selectRoute:r=>selected.push(r)};
const barrierContext=()=>({live:true,connected}),barrierSnapshot=(barriers,context)=>barriers.barriers.map(row=>({...row,state:context.connected?row.state:'unknown',calibrated:true}));
const status={cameraFresh:true,aligned:true,aiFresh:true};
const state={armed:true,zones:[{route_role:'entrance',ai_objects:0}],barriers:{barriers:['entrance','exit'].map(id=>({id,state:'down',fresh:true,calibrated:true}))}};
''' + renderer + r'''
renderRoutes(state,status);const focused=$('hear-entrance'),route=routes[0];focused.focus();
for(let i=1;i<=25;i++){state.zones[0].ai_objects=i;state.barriers.barriers[0].confirmation_s=i/5;renderRoutes(state,status);
 assert.equal($('hear-entrance'),focused);assert.equal(document.activeElement,focused);assert.equal(routes[0],route);}
assert.equal(mounts,1);assert.equal(route.listeners.click.length,1);route.listeners.click[0]();assert.deepEqual(selected,['zones','entrance']);
assert.equal($('lane-entrance-count').textContent,'25 in view');
connected=false;renderRoutes(state,status);assert.equal(focused.disabled,true);assert.equal($('lane-entrance-arm').textContent,'UNKNOWN');
assert.equal($('lane-entrance-status').textContent,'Unavailable');assert.equal(mounts,1);
''')

    def test_ack_and_off_require_receipts_before_changing_ui_state(self):
        page = freshness.page_source()
        transport = ('function createTransport(' + page.split('function createTransport(', 1)[1]
                     .split('return {path,read,command};', 1)[0] + 'return {path,read,command};\n}')
        command = ('async function runCommand(' + page.split('async function runCommand(', 1)[1]
                   .split('async function applyZones(', 1)[0])
        self.js(transport + r'''
let now=1000,commandBusy=false,connectionOk=true,shutdownRequested=false,lastStateAt=1000,lastAckAt=0;
let liveState={alarm:'main',alarm_id:'visit-1'},reply={ok:true,acknowledged:false},messages=[];
const performance={now:()=>now},finite=Number.isFinite,render=()=>{},toast=x=>messages.push(x);
let transport=createTransport({production:true,fetcher:async()=>({ok:true,status:200,json:async()=>reply})});
''' + command + r'''
await runCommand('ack',{alarm_id:'visit-1'});
assert.equal(liveState.alarm,'main');assert.equal(lastAckAt,0);
reply={ok:true,acknowledged:true,ack_remaining_s:10};await runCommand('ack',{alarm_id:'visit-1'});
assert.equal(liveState.alarm,null);assert.equal(lastStateAt,1000);
reply={ok:false,error:'Shutdown request refused'};await runCommand('power/off');
assert.equal(shutdownRequested,false);
reply={ok:true,status:'stopping'};await runCommand('power/off');
assert.equal(shutdownRequested,true);assert.ok(messages.at(-1).includes('requested'));
assert.ok(!messages.at(-1).includes('completed'));
''')


if __name__ == '__main__':
    unittest.main()


class FirstVehicleReportTests(unittest.TestCase):
    def test_waiting_report_reloads_once_only_for_same_origin_parent_version(self):
        from watch import CAR_MEMORY_WAITING
        from test_live_dashboard_freshness import LiveDashboardFreshnessTests
        script=CAR_MEMORY_WAITING.split('<script>',1)[1].split('</script>',1)[0]
        LiveDashboardFreshnessTests.js(self, "let handler,reloads=0;const parent={},location={origin:'http://127.0.0.1:8787',reload:()=>reloads++},window={addEventListener:(kind,fn)=>handler=fn};"+script+"""
const good={source:parent,origin:location.origin,data:{type:'barrierwatcher:report-refresh',reportVersion:'new'}};
handler({...good,origin:'https://elsewhere.example'});handler({...good,source:{}});handler({...good,data:{type:'other'}});assert.equal(reloads,0);
handler(good);handler(good);assert.equal(reloads,1);
""")
