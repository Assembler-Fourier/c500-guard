/**
 * Browser-only, deterministic demonstration. These objects and observations are
 * scripted; this module does not run AI, access cameras, or control real gates.
 */
export const SCENARIOS = Object.freeze([
  { id: 'tour', title: 'Guided tour', description: 'A scripted operator handles arrivals, a pedestrian and two separate visits.' },
  { id: 'entrance', title: 'Entrance arrival', description: 'Catch an approaching car before it reaches the entrance barrier.' },
  { id: 'exit', title: 'Exit arrival', description: 'An exit vehicle enters at the near corner, with an early warning.' },
  { id: 'following', title: 'Following vehicles', description: 'A separate follower keeps its own identity through the ten-second quiet period.' },
  { id: 'pedestrian', title: 'Pedestrian crossing', description: 'A person crossing toward the footpath needs an explicit acknowledgement.' },
  { id: 'occlusion', title: 'Brief occlusion', description: 'A temporarily hidden vehicle keeps the same track and visit.' },
  { id: 'lighting', title: 'Lighting changes', description: 'Scripted exposure changes demonstrate a scene with no vehicle arrival.' },
]);

const SCENARIO_IDS = new Set(SCENARIOS.map(({ id }) => id));
const VEHICLE_LANES = ['entrance', 'exit'];
const EARLY_CONTACT = 0.16;
const WAIT_POSITION = 0.68;
const QUIET_SECONDS = 10;
const TOUR_LENGTH = 66;
const EPSILON = 1e-8;

const STORIES = {
  entrance: [{ at: 0.7, lane: 'entrance', kind: 'car' }],
  exit: [{ at: 0.7, lane: 'exit', kind: 'truck' }],
  following: [{ at: 0.7, lane: 'exit', kind: 'car' }, { at: 4.2, lane: 'exit', kind: 'truck' }],
  pedestrian: [{ at: 0.7, lane: 'pedestrian', kind: 'person' }],
  occlusion: [{ at: 0.7, lane: 'entrance', kind: 'car', occlusion: true }],
  lighting: [],
  tour: [
    { at: 0.7, lane: 'entrance', kind: 'car' },
    { at: 11, lane: 'exit', kind: 'truck' },
    { at: 22, lane: 'pedestrian', kind: 'person' },
    { at: 34, lane: 'exit', kind: 'car' },
    { at: 37.5, lane: 'exit', kind: 'car' },
    { at: 53, lane: 'entrance', kind: 'car', occlusion: true },
  ],
};

const TOUR_ACTIONS = [
  { at: 8, type: 'serve', lane: 'entrance' },
  { at: 18, type: 'serve', lane: 'exit' },
  { at: 27, type: 'acknowledge' },
  { at: 38, type: 'acknowledge' },
  { at: 42, type: 'serve', lane: 'exit' },
  { at: 51, type: 'serve', lane: 'exit' },
  { at: 61, type: 'serve', lane: 'entrance' },
];

function zoneFor(object) {
  if (object.progress + EPSILON < EARLY_CONTACT) return 'Approaching';
  if (object.lane === 'pedestrian') return object.progress < 0.5 ? 'Pedestrian crossing' : 'Footpath';
  if (object.lane === 'entrance') return object.progress < 0.52 ? 'Top approach' : 'Entrance waiting';
  return object.progress < 0.32 ? 'Exit corner' : object.progress < 0.55 ? 'Exit road' : 'Exit barrier';
}

function labelFor(object) {
  if (object.kind === 'person') return 'Pedestrian crossing';
  return object.lane === 'entrance' ? 'Entrance arrival' : 'Exit arrival';
}

export class Simulation {
  constructor({ scenario = 'tour' } = {}) {
    this.speed = 1;
    this.lighting = 'night';
    this.reset(scenario);
  }

  reset(scenario = this.scenario || 'tour') {
    this.scenario = SCENARIO_IDS.has(scenario) ? scenario : 'tour';
    this.time = 0;
    this.paused = false;
    this.quietUntil = 0;
    this.alarm = null;
    this.objects = [];
    this.events = [];
    this.stats = { arrivals: 0, alerts: 0, acknowledged: 0 };
    this.barriers = {
      entrance: { up: false, openedAt: null, lastPassedAt: null },
      exit: { up: false, openedAt: null, lastPassedAt: null },
    };
    this._nextTrack = 101;
    this._nextEvent = 1;
    this._nextAlarm = 1;
    this._scheduled = new Set();
    this._scheduleCycle = 0;
    this._exposureUntil = 0;
    this._event('Scenario ready', SCENARIOS.find(({ id }) => id === this.scenario).description, null, 'system');
    return this.snapshot();
  }

  setScenario(id) {
    if (!SCENARIO_IDS.has(id)) return this.snapshot();
    return this.reset(id);
  }

  setPaused(value) {
    this.paused = Boolean(value);
    return this.snapshot();
  }

  setSpeed(value) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) this.speed = Math.min(4, Math.max(0.25, numeric));
    return this.snapshot();
  }

  setLighting(value) {
    if (value === 'night' || value === 'day') this.lighting = value;
    return this.snapshot();
  }

  acknowledge() {
    return this._acknowledge(false);
  }

  _acknowledge(demoOperator) {
    // Clicking an inactive control never restarts the global quiet period.
    if (!this.alarm) return this.snapshot();
    const object = this.objects.find(({ id }) => id === this.alarm.trackId);
    if (object && !object.acknowledged) {
      object.acknowledged = true;
      this.stats.acknowledged += 1;
    }
    this.quietUntil = this.time + QUIET_SECONDS;
    this._event(
      demoOperator ? 'Demo operator · acknowledged' : 'Visit acknowledged',
      `${this.alarm.trackId} stays acknowledged; new visits wait for the 10-second quiet period.`,
      this.alarm.lane,
      'acknowledge',
    );
    this.alarm = null;
    return this.snapshot();
  }

  serve(lane) {
    return this._serve(lane, false);
  }

  _serve(lane, demoOperator) {
    if (!VEHICLE_LANES.includes(lane)) return this.snapshot();
    const object = this.objects
      .filter((item) => item.lane === lane && !item.served && item.progress < 1 && item.progress >= EARLY_CONTACT - EPSILON)
      .sort((left, right) => right.progress - left.progress)[0];
    if (!object) return this.snapshot();
    object.served = true;
    object.waiting = false;
    if (!object.acknowledged) {
      object.acknowledged = true;
      this.stats.acknowledged += 1;
    }
    const barrier = this.barriers[lane];
    if (!barrier.up) barrier.openedAt = this.time;
    barrier.up = true;
    barrier.lastPassedAt = null;
    if (this.alarm?.trackId === object.id) this.alarm = null;
    this._event(
      demoOperator ? 'Demo operator · barrier opened' : `${lane === 'entrance' ? 'Entrance' : 'Exit'} barrier opened`,
      `${object.id} is served. Other visits keep their own alarm state.`, lane, 'service',
    );
    this._chooseAlarm();
    return this.snapshot();
  }

  step(dtSeconds) {
    const input = Number(dtSeconds);
    if (this.paused || !Number.isFinite(input) || input <= 0) return this.snapshot();
    // A suspended browser tab must not jump the whole demonstration on return.
    const elapsed = Math.min(input, 0.25) * this.speed;
    // Small deterministic integration steps keep queue spacing and scripted
    // occlusion stable at all supported playback speeds.
    const slices = Math.max(1, Math.ceil(elapsed / 0.025));
    for (let index = 0; index < slices; index += 1) this._tick(elapsed / slices);
    return this.snapshot();
  }

  _tick(dt) {
    this.time += dt;
    this._spawnScheduled();
    this._moveObjects(dt);
    this._chooseAlarm();
    this._runTourActions();
    for (const lane of VEHICLE_LANES) {
      const barrier = this.barriers[lane];
      const stillPassing = this.objects.some((object) => object.lane === lane && object.served && object.progress < 1);
      if (barrier.up && !stillPassing && barrier.lastPassedAt !== null && this.time - barrier.lastPassedAt >= 5 - EPSILON) {
        barrier.up = false;
        barrier.openedAt = null;
        barrier.lastPassedAt = null;
        this._event('Barrier closed', 'The served vehicle has passed; the five-second clearance timer is complete.', lane, 'barrier');
      }
    }
    // Completed acknowledged visits need no further matching in this synthetic
    // demo. Keep departed unacknowledged pedestrians for their latched alarm.
    this.objects = this.objects.filter((object) => object.progress < 1 || !object.acknowledged || this.alarm?.trackId === object.id);
  }

  _once(key, action) {
    if (this._scheduled.has(key)) return;
    this._scheduled.add(key);
    action();
  }

  _spawnScheduled() {
    const cycle = this.scenario === 'tour' ? Math.floor(this.time / TOUR_LENGTH) : 0;
    const localTime = this.time - cycle * TOUR_LENGTH;
    if (cycle !== this._scheduleCycle) {
      this._scheduled.clear();
      this._scheduleCycle = cycle;
    }
    STORIES[this.scenario].forEach((spec, index) => {
      if (localTime + EPSILON < spec.at) return;
      this._once(`spawn:${cycle}:${index}`, () => {
        const object = {
          id: `T-${this._nextTrack++}`, kind: spec.kind, lane: spec.lane,
          progress: 0, detected: false, occluded: false, acknowledged: false,
          waiting: false, served: false, alerted: false, firstContactAt: null,
          occlusion: Boolean(spec.occlusion), wasOccluded: false, passedAt: null,
        };
        this.objects.push(object);
        this.stats.arrivals += 1;
        this._event('New simulated visit', `${object.id} · ${spec.kind} approaching the ${spec.lane === 'pedestrian' ? 'footpath' : spec.lane + ' lane'}.`, spec.lane, 'arrival');
      });
    });
    if (this.scenario === 'lighting') {
      [2, 6, 10].forEach((at, index) => {
        if (this.time + EPSILON < at) return;
        this._once(`lighting:${index}`, () => {
          this._exposureUntil = this.time + 1.5;
          this._event('Exposure change', 'Scripted lighting change; no object visit has been introduced.', null, 'lighting');
        });
      });
    }
  }

  _moveObjects(dt) {
    // Foremost first: a follower never jumps through the visit ahead of it.
    const ordered = [...this.objects].sort((left, right) => right.progress - left.progress);
    for (const object of ordered) {
      if (object.progress >= 1) continue;
      const velocity = object.kind === 'person' ? 0.052 : object.served ? 0.13 : 0.08;
      let next = Math.min(1, object.progress + velocity * dt);
      if (object.kind !== 'person') {
        if (!object.served) next = Math.min(next, WAIT_POSITION);
        const ahead = ordered.filter((item) => item !== object && item.lane === object.lane && item.progress > object.progress && item.progress < 1)
          .sort((left, right) => left.progress - right.progress)[0];
        if (ahead) next = Math.min(next, Math.max(object.progress, ahead.progress - 0.17));
      }
      object.waiting = object.kind !== 'person' && (next - object.progress <= EPSILON || (!object.served && next >= WAIT_POSITION - EPSILON));
      object.progress = next;
      object.occluded = object.occlusion && next >= 0.32 && next < 0.49;
      object.detected = next >= 0.035 && !object.occluded && next < 1;
      if (object.occluded && !object.wasOccluded) this._event('Track retained', `${object.id} is briefly hidden; its visit identity is retained.`, object.lane, 'tracking');
      if (!object.occluded && object.wasOccluded) this._event('Track reacquired', `${object.id} returns with the same identity. No new visit is created.`, object.lane, 'tracking');
      object.wasOccluded = object.occluded;
      if (object.firstContactAt === null && next >= EARLY_CONTACT - EPSILON && object.detected) {
        object.firstContactAt = this.time;
        this._event('Early zone contact', `${object.id} reached ${zoneFor(object).toLowerCase()}.`, object.lane, 'zone');
      }
      if (next >= 1) {
        object.passedAt = this.time;
        if (object.kind !== 'person' && object.served) this.barriers[object.lane].lastPassedAt = this.time;
        this._event('Visit left the view', object.kind === 'person' && !object.acknowledged
          ? `${object.id} has left; its pedestrian alarm still requires acknowledgement.`
          : `${object.id} has completed the simulated route.`, object.lane, 'departure');
      }
    }
  }

  _chooseAlarm() {
    if (this.alarm || this.time < this.quietUntil - EPSILON) return;
    const pending = this.objects
      .filter((object) => object.firstContactAt !== null && !object.acknowledged && !object.alerted)
      .sort((left, right) => left.firstContactAt - right.firstContactAt)[0];
    if (!pending) return;
    pending.alerted = true;
    this.alarm = { id: `A-${this._nextAlarm++}`, lane: pending.lane, label: labelFor(pending), trackId: pending.id, since: this.time };
    this.stats.alerts += 1;
    this._event('Alarm raised', `${pending.id} · ${labelFor(pending)}.`, pending.lane, 'alarm');
  }

  _runTourActions() {
    if (this.scenario !== 'tour') return;
    const cycle = Math.floor(this.time / TOUR_LENGTH);
    const localTime = this.time - cycle * TOUR_LENGTH;
    TOUR_ACTIONS.forEach((action, index) => {
      if (localTime + EPSILON < action.at) return;
      this._once(`operator:${cycle}:${index}`, () => {
        if (action.type === 'serve') this._serve(action.lane, true);
        else this._acknowledge(true);
      });
    });
  }

  _event(title, detail, lane, type) {
    this.events.unshift({ id: `E-${this._nextEvent++}`, time: this.time, title, detail, lane, type });
    this.events.length = Math.min(this.events.length, 24);
  }

  snapshot() {
    const quietRemaining = Math.max(0, this.quietUntil - this.time);
    const visible = this.objects.filter(({ progress }) => progress < 1);
    const pending = this.objects.some((object) => object.firstContactAt !== null && !object.acknowledged && !object.alerted);
    const occluded = visible.some((object) => object.occluded);
    const exposure = this.time < this._exposureUntil;
    const status = this.paused ? 'Paused' : this.alarm ? 'Attention needed' : quietRemaining > EPSILON ? 'Acknowledged · quiet period' : exposure ? 'Lighting change · watching' : 'Watching';
    return {
      time: this.time, scenario: this.scenario, paused: this.paused, speed: this.speed, lighting: this.lighting,
      quietRemaining, alarm: this.alarm ? { ...this.alarm } : null, status,
      events: this.events.map((event) => ({ ...event })),
      objects: visible.map((object) => ({
        id: object.id, kind: object.kind, lane: object.lane, progress: object.progress,
        detected: object.detected, occluded: object.occluded, acknowledged: object.acknowledged, served: object.served,
        waiting: object.waiting, zone: zoneFor(object),
      })),
      barriers: Object.fromEntries(VEHICLE_LANES.map((lane) => [lane, {
        up: this.barriers[lane].up,
        openFor: this.barriers[lane].up ? this.time - this.barriers[lane].openedAt : 0,
      }])),
      stats: { ...this.stats },
      pipeline: {
        capture: this.paused ? 'Paused' : 'Synthetic frames',
        detection: exposure ? 'Exposure change only' : visible.some((object) => object.detected) ? 'Scripted observations' : 'No object observations',
        tracking: occluded ? 'Identity held through occlusion' : visible.length ? `${visible.length} separate visit${visible.length === 1 ? '' : 's'}` : 'Waiting for an arrival',
        zones: visible.some((object) => object.firstContactAt !== null) ? 'Early contact recorded' : 'Watching approach zones',
        alarm: this.alarm ? 'Explicit acknowledgement needed' : quietRemaining > EPSILON ? (pending ? 'Separate visit queued' : 'Ten-second quiet period') : 'Ready for a new visit',
      },
    };
  }
}
