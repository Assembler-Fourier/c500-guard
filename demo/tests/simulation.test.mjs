import test from 'node:test';
import assert from 'node:assert/strict';
import { SCENARIOS, Simulation } from '../simulation.mjs';

function advance(simulation, seconds) {
  const end = simulation.time + seconds;
  while (simulation.time < end - 1e-8) simulation.step(Math.min(0.05, (end - simulation.time) / simulation.speed));
  return simulation.snapshot();
}

function until(simulation, predicate, limit = 30) {
  const end = simulation.time + limit;
  while (simulation.time < end && !predicate(simulation.snapshot())) simulation.step(0.025);
  assert.ok(predicate(simulation.snapshot()), 'expected state reached within the bounded simulation window');
  return simulation.snapshot();
}

test('all advertised scenarios have deterministic, isolated initial states', () => {
  assert.equal(SCENARIOS.length, 7);
  for (const { id } of SCENARIOS) {
    const first = new Simulation({ scenario: id });
    const second = new Simulation({ scenario: id });
    assert.deepEqual(advance(first, 9), advance(second, 9));
    assert.equal(first.snapshot().scenario, id);
  }
});

test('first contact raises an early alarm before either vehicle reaches its barrier', () => {
  for (const scenario of ['entrance', 'exit']) {
    const sim = new Simulation({ scenario });
    const state = until(sim, ({ alarm }) => Boolean(alarm));
    assert.equal(state.alarm.lane, scenario);
    assert.ok(state.objects[0].progress >= 0.16 - 1e-8);
    assert.ok(state.objects[0].progress < 0.17);
    assert.equal(state.barriers[scenario].up, false);
  }
});

test('an acknowledged visit never repeats, while a distinct follower alerts after ten seconds', () => {
  const sim = new Simulation({ scenario: 'following' });
  const first = until(sim, ({ alarm }) => Boolean(alarm)).alarm.trackId;
  const ackAt = sim.time;
  sim.acknowledge();
  const quietState = advance(sim, 9.9);
  assert.equal(quietState.alarm, null);
  assert.ok(quietState.quietRemaining > 0);
  assert.equal(quietState.stats.alerts, 1);
  const follower = until(sim, ({ alarm }) => Boolean(alarm), 1);
  assert.notEqual(follower.alarm.trackId, first);
  assert.ok(follower.time >= ackAt + 10 - 1e-8);
  assert.equal(follower.stats.alerts, 2);
  assert.equal(follower.objects.find(({ id }) => id === first).acknowledged, true);
  sim.acknowledge();
  assert.equal(advance(sim, 25).stats.alerts, 2);
  assert.equal(sim.snapshot().alarm, null);
});

test('ACK silences a visit but does not physically open its barrier', () => {
  const sim = new Simulation({ scenario: 'entrance' });
  until(sim, ({ alarm }) => Boolean(alarm));
  sim.acknowledge();
  const state = advance(sim, 20);
  assert.equal(state.alarm, null);
  assert.equal(state.stats.alerts, 1);
  assert.equal(state.barriers.entrance.up, false);
  assert.equal(state.objects[0].progress, 0.68);
  assert.equal(state.objects[0].waiting, true);
});

test('serving one lane resolves only the foreground visit without creating global quiet', () => {
  const sim = new Simulation({ scenario: 'following' });
  const first = until(sim, ({ objects }) => objects.length === 2 && objects.every((object) => object.progress >= 0.16)).alarm.trackId;
  const state = sim.serve('exit');
  assert.equal(state.quietRemaining, 0);
  assert.equal(state.barriers.exit.up, true);
  assert.ok(state.alarm);
  assert.notEqual(state.alarm.trackId, first);
  assert.equal(state.stats.alerts, 2);
  assert.equal(state.objects.filter(({ acknowledged }) => acknowledged).length, 1);
  assert.equal(sim.serve('entrance').alarm.trackId, state.alarm.trackId);
  const followerId = state.alarm.trackId;
  advance(sim, 8);
  assert.equal(sim.snapshot().alarm.trackId, followerId);
  assert.equal(sim.snapshot().objects.find(({ id }) => id === followerId).acknowledged, false);
});

test('served vehicle departs and barrier closes five seconds after clearance', () => {
  const sim = new Simulation({ scenario: 'exit' });
  until(sim, ({ objects }) => objects.some(({ waiting }) => waiting));
  assert.equal(sim.snapshot().objects[0].served, false);
  const served = sim.serve('exit').objects[0];
  assert.equal(served.served, true);
  assert.equal(served.acknowledged, true);
  assert.equal(advance(sim, 0.5).objects.find(({ id }) => id === served.id).served, true);
  until(sim, ({ objects }) => objects.length === 0);
  assert.equal(sim.objects.length, 0, 'completed acknowledged visit is released');
  assert.equal(advance(sim, 4.9).barriers.exit.up, true);
  assert.equal(advance(sim, 0.2).barriers.exit.up, false);
  assert.equal(sim.snapshot().barriers.exit.openFor, 0);
});

test('high-refresh rendering does not label a moving vehicle as waiting', () => {
  const sim = new Simulation({ scenario: 'entrance' });
  advance(sim, 2);
  const state = sim.step(1 / 144);
  assert.equal(state.objects[0].waiting, false);
});

test('a gate stays open through both separately served visits and their final clearance', () => {
  const sim = new Simulation({ scenario: 'following' });
  until(sim, ({ objects }) => objects.length === 2 && objects.every(({ progress }) => progress >= 0.16));
  sim.serve('exit');
  sim.serve('exit');
  while (sim.snapshot().objects.length) {
    assert.equal(sim.step(0.025).barriers.exit.up, true);
  }
  assert.equal(advance(sim, 4.9).barriers.exit.up, true);
  assert.equal(advance(sim, 0.2).barriers.exit.up, false);
});

test('pedestrian alarm latches after the person leaves until explicit acknowledgement', () => {
  const sim = new Simulation({ scenario: 'pedestrian' });
  const alarm = until(sim, ({ alarm }) => Boolean(alarm)).alarm;
  const departed = advance(sim, 25);
  assert.equal(departed.objects.length, 0);
  assert.equal(departed.alarm.trackId, alarm.trackId);
  assert.equal(sim.objects.length, 1, 'departed pedestrian is retained until its latched alarm is acknowledged');
  assert.equal(sim.serve('entrance').alarm.trackId, alarm.trackId);
  const acknowledged = sim.acknowledge();
  assert.equal(acknowledged.alarm, null);
  assert.equal(acknowledged.stats.acknowledged, 1);
  assert.equal(advance(sim, 30).stats.alerts, 1);
  assert.equal(sim.objects.length, 0);
});

test('a scripted occlusion retains the exact track and acknowledged visit', () => {
  const sim = new Simulation({ scenario: 'occlusion' });
  const initial = until(sim, ({ alarm }) => Boolean(alarm));
  const id = initial.objects[0].id;
  sim.acknowledge();
  const hidden = until(sim, ({ objects }) => objects.some(({ occluded }) => occluded));
  assert.equal(hidden.objects[0].id, id);
  assert.equal(hidden.objects[0].detected, false);
  assert.equal(hidden.objects[0].acknowledged, true);
  const returned = until(sim, ({ objects }) => objects.some(({ occluded, detected }) => !occluded && detected));
  assert.equal(returned.objects[0].id, id);
  assert.equal(returned.stats.arrivals, 1);
  assert.equal(advance(sim, 20).stats.alerts, 1);
});

test('lighting-only story never invents a vehicle or alarm', () => {
  const sim = new Simulation({ scenario: 'lighting' });
  for (let index = 0; index < 400; index += 1) {
    const state = sim.step(0.1);
    assert.equal(state.alarm, null);
    assert.equal(state.objects.length, 0);
    assert.equal(state.stats.arrivals, 0);
  }
  assert.equal(sim.snapshot().events.filter(({ type }) => type === 'lighting').length, 3);
});

test('pause freezes simulation; speed scales time; large time gaps are bounded', () => {
  const sim = new Simulation({ scenario: 'exit' });
  sim.setPaused(true);
  const frozen = sim.snapshot();
  sim.step(10);
  assert.deepEqual(sim.snapshot(), frozen);
  sim.setPaused(false);
  sim.setSpeed(2);
  assert.ok(Math.abs(sim.step(0.1).time - 0.2) < 1e-8);
  const previous = sim.time;
  sim.step(1000);
  assert.ok(Math.abs(sim.time - previous - 0.5) < 1e-8);
  const time = sim.time;
  sim.step(NaN);
  sim.step(-3);
  assert.equal(sim.time, time);
  assert.equal(sim.setSpeed(Infinity).speed, 2);
  assert.equal(sim.setSpeed(100).speed, 4);
  assert.equal(sim.setSpeed(-2).speed, 0.25);
});

test('reset clears active visits and alarms while preserving display preferences', () => {
  const sim = new Simulation({ scenario: 'exit' });
  until(sim, ({ alarm }) => Boolean(alarm));
  sim.setSpeed(2);
  sim.setLighting('day');
  sim.setPaused(true);
  const state = sim.reset('entrance');
  assert.equal(state.time, 0);
  assert.equal(state.paused, false);
  assert.equal(state.speed, 2);
  assert.equal(state.lighting, 'day');
  assert.equal(state.scenario, 'entrance');
  assert.equal(state.alarm, null);
  assert.equal(state.objects.length, 0);
  assert.deepEqual(state.stats, { arrivals: 0, alerts: 0, acknowledged: 0 });
  assert.equal(sim.setScenario('nonexistent').scenario, 'entrance');
});

test('snapshot consumers cannot mutate simulation state', () => {
  const sim = new Simulation({ scenario: 'exit' });
  const state = until(sim, ({ alarm }) => Boolean(alarm));
  state.alarm.trackId = 'fake';
  state.objects[0].acknowledged = true;
  state.events[0].title = 'fake';
  state.stats.alerts = 1000;
  state.barriers.exit.up = true;
  const actual = sim.snapshot();
  assert.notEqual(actual.alarm.trackId, 'fake');
  assert.equal(actual.objects[0].acknowledged, false);
  assert.notEqual(actual.events[0].title, 'fake');
  assert.equal(actual.stats.alerts, 1);
  assert.equal(actual.barriers.exit.up, false);
});

test('tour scripts are labelled and repeat without blocking on an old visit', () => {
  const sim = new Simulation({ scenario: 'tour' });
  advance(sim, 66);
  const first = sim.snapshot();
  assert.equal(first.stats.arrivals, 6);
  assert.equal(first.stats.alerts, 6);
  assert.equal(first.stats.acknowledged, 6);
  assert.ok(first.events.some(({ title }) => title.startsWith('Demo operator')));
  const repeated = advance(sim, 66);
  assert.equal(repeated.stats.arrivals, 12);
  assert.equal(repeated.stats.alerts, 12);
  assert.equal(repeated.stats.acknowledged, 12);
});

test('a 100-cycle tour bounds retained visits, events and schedule bookkeeping', () => {
  const sim = new Simulation({ scenario: 'tour' });
  sim.setSpeed(4);
  while (sim.time < 6600 - 1e-8) {
    sim.step(Math.min(0.25, (6600 - sim.time) / sim.speed));
    assert.ok(sim.objects.length <= 6, 'past resolved visits must not accumulate');
    assert.ok(sim._scheduled.size <= 13, 'past tour schedules must not accumulate');
    assert.ok(sim.events.length <= 24, 'event feed remains bounded');
  }
  const state = sim.snapshot();
  assert.equal(state.stats.arrivals, 600);
  assert.equal(state.stats.alerts, 600);
  assert.equal(state.stats.acknowledged, 600);
  assert.equal(state.alarm, null);
});
