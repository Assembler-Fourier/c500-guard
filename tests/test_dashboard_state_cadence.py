"""Visible state refresh stays serial, with legacy slow retries and stale-response fences."""
import unittest
import test_live_dashboard_freshness as freshness


class DashboardStateCadenceTests(unittest.TestCase):
    js = freshness.LiveDashboardFreshnessTests.js

    def poller(self):
        page = freshness.page_source()
        return 'async function readState(' + page.split('async function readState(', 1)[1].split('async function readEvents(', 1)[0]

    def test_success_is_faster_but_failure_hidden_and_stopping_remain_bounded(self):
        self.js("""
let active=true,stateController=null,liveState={},connectionOk=false,lastStateAt=0,lastAckAt=0;
let historyPoints=[],lastHistoryFrame=null,lastEventVersion=1,lastEventsAt=1000;
let pending=[],renders=0,requests=0,now=1000;
const timers=new Map();let nextId=0;
const document={hidden:false},performance={now:()=>now},finite=Number.isFinite;
const setTimeout=(fn,ms)=>{timers.set(++nextId,{fn,ms});return nextId};
const clearTimeout=id=>timers.delete(id);
const getJSON=(path,signal)=>{assert.equal(path,'/api/state');requests++;return new Promise((resolve,reject)=>pending.push({resolve,reject,signal}))};
const render=()=>renders++,readEvents=()=>assert.fail('Unchanged activity must not refetch');
const maybeReloadForUpdate=()=>{};
function successor(ms){const rows=[...timers];assert.equal(rows.length,1);assert.equal(rows[0][1].ms,ms);timers.delete(rows[0][0]);return rows[0][1].fn;}
""" + self.poller() + """
let request=readState();
assert.equal(requests,1);assert.equal(timers.size,1);assert.equal([...timers.values()][0].ms,4000);
await Promise.resolve();assert.equal(requests,1,'No next request before the previous response settles');
pending.shift().resolve({frames:1,alerts_version:1});await request;
assert.equal(renders,1);assert.equal(connectionOk,true);
let next=successor(250);request=next();assert.equal(requests,2);
pending.shift().reject(new Error('Disconnected'));await request;
assert.equal(connectionOk,false);assert.equal(renders,2);next=successor(600);
document.hidden=true;request=next();pending.shift().resolve({frames:2,alerts_version:1});await request;
assert.equal(connectionOk,true);next=successor(600);
document.hidden=false;request=next();pending.shift().resolve({frames:3,alerts_version:1});await request;
next=successor(250);request=next();const old=stateController;active=false;
pending.shift().resolve({frames:4,alerts_version:1});await request;
assert.equal(liveState.frames,3,'A stopped page discards the response');assert.equal(timers.size,0);
assert.equal(stateController,null);await readState();assert.equal(requests,5);
""")

    def test_timeout_cannot_produce_a_fast_retry_or_keep_controls_current(self):
        self.js("""
let active=true,stateController=null,liveState={frames:1},connectionOk=true,lastStateAt=1000,lastAckAt=1000;
let historyPoints=[],lastHistoryFrame=null,lastEventVersion=1,lastEventsAt=1000;
let now=1000,renders=0;const timers=new Map();let id=0;
const document={hidden:false},performance={now:()=>now},finite=Number.isFinite;
const setTimeout=(fn,ms)=>{timers.set(++id,{fn,ms});return id},clearTimeout=i=>timers.delete(i);
const getJSON=(_path,signal)=>new Promise((_resolve,reject)=>signal.addEventListener('abort',()=>reject(new Error('Timeout'))));
const render=()=>renders++,readEvents=()=>assert.fail('No activity read after timeout'),maybeReloadForUpdate=()=>{};
""" + self.poller() + """
const pending=readState();const deadline=[...timers.values()][0];assert.equal(deadline.ms,4000);
now=5000;deadline.fn();await pending;
assert.equal(connectionOk,false);assert.equal(lastStateAt,1000);assert.equal(lastAckAt,1000);
assert.equal(stateController,null);assert.equal(renders,1);assert.equal(timers.size,1);
assert.equal([...timers.values()][0].ms,600,'A timeout must keep the prior retry cadence');
""")


if __name__ == '__main__':
    unittest.main()
