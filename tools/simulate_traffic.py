"""Accelerated decision-path traffic simulation; never camera/model/audio hardware.

Event-driven synthetic detections exercise the real tracker, visit memory and
alarm owner. Virtual quiet intervals are skipped after sampled fresh clearance.
Virtual hours are NOT hours of image inference or proof of real-camera recall.
Run only against a disposable checkout with tests/ and test fixtures installed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from unittest import mock

W = H = 1000
REAL_CLOCK = time.perf_counter


def simulate(source: Path, episodes=64, seed=20261122, progress=None):
    source = source.resolve()
    if source == Path('C:/c500-guard').resolve():
        raise ValueError('Use a disposable test checkout, not the installed application')
    sys.path[:0] = [str(source/'tests'), str(source/'src')]
    from test_watch import watch, watcher
    from object_tracking import ObjectVisitTracker
    rng = random.Random(seed)
    now = [100.0]
    stats = Counter()
    scenarios = Counter()
    failures = []
    trace_hash = hashlib.sha256()
    started = REAL_CLOCK()
    with mock.patch.object(watch.time, 'monotonic', side_effect=lambda: now[0]), \
         mock.patch.object(watch, 'notify', lambda *_: None):
        obj = watcher()
        # All effects are in memory. Keep actual tracking, visits, alarm/ACK logic.
        obj.log_event = lambda *_a, **_k: None
        obj._queue_snapshot = lambda *_a, **_k: None
        obj.learning = obj.recorder = obj.semantic = obj.barrier_workflow = None
        obj.args.silent = True
        zones = []
        for role, x in (('entrance', .52), ('exit', .04)):
            zones.append(watch.Zone(role, (x,.05,.42,.9), trigger='ai',
                classes=['car','truck','bus','motorcycle'], motion=False, light=False,
                ai_confidence=.5, ai_confirm_observations=2,
                route_group=role, route_role=role, arrival_direction=[0,-1],
                confirmation_polygon=[[x,.2],[x+.42,.2],[x+.42,.60],[x,.60]],
                stationary_confirm_s=3., arrival_min_progress=.012))
        zones.append(watch.Zone('footpath', (.04,.05,.42,.9), trigger='ai',
            classes=['person'], motion=False, light=False, ai_confidence=.5,
            ai_confirm_observations=2, route_group='footpath', route_role='pedestrian'))
        obj.zones = zones
        tracker = ObjectVisitTracker()
        alarms = []
        def row(x,y,label='car',score=.9,width=.18):
            height = .28 if label=='person' else .15
            return {'label':label,'confidence':round(score,4),'raw_confidence':score,
                    'bbox':[(x-width/2)*W,(y-height)*H,(x+width/2)*W,y*H]}
        def step(stamp, objects=(), fresh=True):
            now[0]=stamp
            obj.last_frame_at=stamp
            events, observed=tracker.update(list(objects),zones,W,H,stamp)
            obj.visits.observe(stamp,zones,{'available':fresh,'state':'ready' if fresh else 'retrying',
                'age_s':0. if fresh else 4.,'detections':observed},frame_fresh=fresh,width=W,height=H)
            for kind,reason,event_id in events:
                context=tracker.event_context(event_id,zones)
                details={'source':'ai',**context,'frame_received_at':stamp if fresh else stamp-4.}
                before=obj._alarm_sequence
                if obj.raise_alarm(kind,reason,details=details):tracker.mark_emitted(event_id)
                if obj._alarm_sequence!=before:
                    alarm={'at':stamp,'id':obj.alarm_id,'track':context['track_id'],
                           'label':context['label'],'route':context['route_role']}
                    alarms.append(alarm)
                    stats['alarms']+=1
                    stats['pedestrian_alarms' if context['label']=='person' else 'vehicle_alarms']+=1
                    trace_hash.update(json.dumps(alarm,sort_keys=True).replace(obj._alarm_session,'session').encode())
                    assert obj.buzzer.want in ('main','early')
            stats['observation_packets']+=1
            stats['object_observations']+=len(objects)
            stats['max_visits']=max(stats['max_visits'],len(obj.visits._visits))
            stats['max_tracks']=max(stats['max_tracks'],tracker.track_count)
            assert len(obj.visits._visits)<=obj.visits.max_visits
            return observed
        def ack(stamp):
            now[0]=stamp
            if obj.alarm:
                assert obj.acknowledge(via='simulation')
                assert obj._ack_until==stamp+10.
                assert obj.buzzer.want is None
                stats['acknowledgements']+=1
        def expect(actual, expected, label):
            if actual!=expected:raise AssertionError(f'{label}: expected {expected}, got {actual}')
        for episode in range(episodes):
            base=100.+episode*120.
            kind=episode%8
            before=len(alarms)
            x=.74+rng.uniform(-.012,.012)
            label=rng.choice(['car','truck','bus','motorcycle'])
            score=rng.uniform(.72,.98)
            case=['entrance','exit','pedestrian_latch','following_car','quiet_follower',
                  'short_occlusion','stale_and_empty','two_pedestrians'][kind]
            scenarios[case]+=1
            try:
                if kind in (0,1,5):
                    if kind==1:x=.25+rng.uniform(-.012,.012)
                    stats['vehicles']+=1
                    for dt,y in ((0,.80),(.25,.71),(.5,.62),(.75,.56),(1.,.51)):
                        step(base+dt,[row(x,y,label,score)])
                    expect(len(alarms)-before,1,'moving vehicle')
                    ack(base+2.)
                    if kind==5:
                        for dt in (2.5,3.,3.5):step(base+dt)
                        for dt in (4.,4.25,5.,10.,12.5):step(base+dt,[row(x,.51,label,score)])
                        expect(len(alarms)-before,1,'same vehicle short occlusion')
                    last=base+13.
                elif kind==2:
                    stats['pedestrians']+=1
                    for dt in (0.,.25):step(base+dt,[row(.24,.62,'person',score,.08)])
                    expect(len(alarms)-before,1,'pedestrian arrival')
                    for dt in (1.,2.,3.,5.,8.,12.,20.):step(base+dt)
                    expect(obj.alarm_details.get('manual_ack_required'),True,'pedestrian latch')
                    expect(len(alarms)-before,1,'pedestrian departure keeps one alarm')
                    ack(base+21.)
                    last=base+22.
                elif kind==3:
                    stats['vehicles']+=2
                    for dt in (0.,.25,1.,2.,3.):step(base+dt,[row(x,.55,label,score)])
                    expect(len(alarms)-before,1,'lead confirmed')
                    ack(base+3.25)
                    for n in range(4,12):step(base+n,[row(x,.55,label,score)])
                    step(base+12.,[row(x,.55,label,score),row(x,.85,label,score)])
                    step(base+12.5,[row(x,.55,label,score),row(x,.78,label,score)])
                    for dt,y in ((13.,.71),(13.5,.64),(14.,.57),(14.5,.53)):
                        step(base+dt,[row(x,y,label,score)])
                    expect(len(alarms)-before,2,'separate follower after quiet')
                    assert alarms[-1]['track']!=alarms[before]['track']
                    ack(base+16.)
                    last=base+17.
                elif kind==4:
                    stats['vehicles']+=2
                    # Parallel visible objects must remain distinct throughout quiet.
                    a,b=x-.11,x+.11
                    for dt,y in ((0,.75),(.25,.65),(.5,.56),(.75,.51)):
                        step(base+dt,[row(a,y,label,score,.14)])
                    expect(len(alarms)-before,1,'quiet lead')
                    ack(base+1.)
                    for dt,y in ((2.,.75),(2.25,.65),(2.5,.56),(2.75,.51)):
                        step(base+dt,[row(a,.51,label,score,.14),row(b,y,label,score,.14)])
                    for dt in (3.,5.,8.,10.99):step(base+dt,[row(a,.51,label,score,.14),row(b,.51,label,score,.14)])
                    expect(len(alarms)-before,1,'quiet holds follower')
                    step(base+11.01,[row(a,.51,label,score,.14),row(b,.51,label,score,.14)])
                    expect(len(alarms)-before,2,'pending follower after quiet')
                    ack(base+12.5)
                    last=base+14.
                elif kind==6:
                    stats['negative_episodes']+=1
                    for dt in (0.,.25,.5,1.,2.,3.):
                        step(base+dt,[row(x,.55,'car',.2),row(.98,.8,'person',score,.025)])
                    for dt in (4.,4.25,4.5,5.,6.,7.,8.):step(base+dt,[row(x,.55,label,score)],fresh=False)
                    expect(len(alarms)-before,0,'low confidence/outside/stale cannot alarm')
                    last=base+10.
                else:
                    stats['pedestrians']+=2
                    a,b=.14,.35
                    for dt in (0.,.25):step(base+dt,[row(a,.65,'person',score,.07)])
                    expect(len(alarms)-before,1,'first person')
                    ack(base+1.5)
                    for dt in (2.,2.25,3.,5.,8.,11.49):
                        step(base+dt,[row(a,.65,'person',score,.07),row(b,.65,'person',score,.07)])
                    expect(len(alarms)-before,1,'pedestrian quiet')
                    step(base+11.51,[row(b,.65,'person',score,.07)])
                    expect(len(alarms)-before,2,'second person remains eligible')
                    ack(base+13.)
                    last=base+14.
                for tick in range(17):step(last+tick*.5)
                expect(obj.visits.status(last+8.)['active_visits'],0,'fresh observed clearance')
                expect(obj.alarm,None,'episode acknowledged')
                stats['episodes_passed']+=1
            except Exception as exc:
                failures.append({'episode':episode,'case':case,'virtual_at':now[0],
                                 'error':str(exc),'new_alarms':alarms[before:]})
                # Stop on the first failure so broken state cannot mask later faults.
                break
            if progress and (episode+1)%1000==0:progress(episode+1)
            # Keep evidence bounded while preserving cumulative counters/digest.
            if len(alarms)>100:alarms[:]=alarms[-8:]
        result={'seed':seed,'requested_episodes':episodes,'stats':dict(stats),'scenarios':dict(scenarios),
                'virtual_hours':round(stats['episodes_passed']*120/3600,6),
                'real_seconds':round(REAL_CLOCK()-started,3),'failures':failures,
                'trace_sha256':trace_hash.hexdigest(),
                'scope':'Synthetic detector packets through actual tracker, visits, alarm and ACK; no pixel/model/audio hardware',
                'quiet_intervals':'Event-driven: sampled fresh clearance, then skipped idle intervals; not continuous video inference'}
        obj.loudness.end()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--episodes',type=int,default=64)
    parser.add_argument('--seed',type=int,default=20261122)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=simulate(args.source_root,args.episodes,args.seed,
                    lambda n:print(f'{n} synthetic episodes completed',flush=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return int(bool(result['failures']))


if __name__=='__main__':raise SystemExit(main())
