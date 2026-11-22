/** Original, synthetic campus illustration. This renderer never uses camera footage. */
const W = 960;
const H = 620;
const LANE_COLORS = { entrance: '#ffc779', exit: '#76e4ce', pedestrian: '#baabff' };
const ROUTES = {
  entrance: [[0, 982, 92], [.16, 853, 131], [.42, 741, 211], [.68, 644, 352], [.84, 581, 477], [1, 496, 667]],
  exit: [[0, 77, 664], [.16, 240, 558], [.42, 378, 448], [.68, 506, 351], [.84, 568, 196], [1, 744, -38]],
  pedestrian: [[0, 69, 465], [.16, 219, 380], [.42, 370, 296], [.68, 502, 259], [.84, 646, 259], [1, 810, 309]],
};
const TREE_POSITIONS = [[47, 92, 28], [114, 123, 35], [176, 69, 23], [232, 116, 30], [56, 222, 29], [124, 285, 24], [185, 237, 29], [822, 402, 30], [889, 365, 24], [842, 489, 32], [746, 539, 26], [927, 538, 32], [32, 556, 23], [354, 69, 25], [438, 131, 29], [553, 69, 22]];

function clamp(v, min = 0, max = 1) { return Math.max(min, Math.min(max, v)); }
function catmull(a, b, c, d, t) {
  return .5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t * t * t);
}
function routePoint(lane, progress) {
  const points = ROUTES[lane] || ROUTES.exit;
  const p = clamp(progress);
  let i = 0;
  while (i < points.length - 2 && p > points[i + 1][0]) i++;
  const t = (p - points[i][0]) / (points[i + 1][0] - points[i][0]);
  const a = points[Math.max(0, i - 1)], b = points[i], c = points[i + 1], d = points[Math.min(points.length - 1, i + 2)];
  return { x: catmull(a[1], b[1], c[1], d[1], t), y: catmull(a[2], b[2], c[2], d[2], t) };
}
function pose(lane, progress) {
  const p = routePoint(lane, progress), a = routePoint(lane, progress - .002), b = routePoint(lane, progress + .002);
  return { ...p, angle: Math.atan2(b.y - a.y, b.x - a.x) };
}
function rounded(ctx, x, y, w, h, r = 6) {
  ctx.beginPath(); ctx.roundRect(x, y, w, h, r);
}
function line(ctx, points) {
  ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
}
function route(ctx, lane, start = 0, end = 1) {
  ctx.beginPath();
  for (let i = 0; i <= 90; i++) {
    const p = routePoint(lane, start + (end - start) * i / 90);
    if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
  }
}

export class SceneRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    if (!this.ctx) throw new Error('Canvas 2D is unavailable.');
    this.width = 0; this.height = 0; this.pixelRatio = 1;
    this.exposureLevel = 0; this.lastSimulationTime = 0;
    this.resize();
  }

  resize() {
    const bounds = this.canvas.getBoundingClientRect();
    const width = Math.max(1, bounds.width || this.canvas.clientWidth || W);
    const height = Math.max(1, bounds.height || this.canvas.clientHeight || H);
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    if (width === this.width && height === this.height && dpr === this.pixelRatio) return;
    this.width = width; this.height = height; this.pixelRatio = dpr;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
  }

  render(snapshot, { zones = true, stage = 'all' } = {}) {
    this.resize();
    const ctx = this.ctx;
    const night = snapshot.lighting !== 'day';
    this.night = night;
    const scale = Math.min(this.width / W, this.height / H);
    ctx.setTransform(this.pixelRatio, 0, 0, this.pixelRatio, 0, 0);
    ctx.fillStyle = night ? '#080f14' : '#d8e3dc';
    ctx.fillRect(0, 0, this.width, this.height);
    ctx.translate((this.width - W * scale) / 2, (this.height - H * scale) / 2);
    ctx.scale(scale, scale);
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.clip();
    this.background();
    this.roads();
    this.landscape();
    this.building();
    this.crossing();
    this.lamps();
    if (zones) this.zones(stage);
    this.directionArrow('exit', .33);
    this.directionArrow('exit', .88);
    this.directionArrow('entrance', .39);
    this.directionArrow('entrance', .88);
    this.barrier('exit', snapshot.barriers?.exit?.up || false);
    this.barrier('entrance', snapshot.barriers?.entrance?.up || false);
    for (const object of snapshot.objects || []) this.object(object, snapshot);
    this.exposure(snapshot);
    this.mapLabels();
    this.frameCorners();
    ctx.restore();
  }

  exposure(snapshot) {
    const time = Number(snapshot.time) || 0;
    if (time < this.lastSimulationTime) this.exposureLevel = 0;
    const dt = clamp(time - this.lastSimulationTime, 0, .25);
    this.lastSimulationTime = time;
    const target = snapshot.pipeline?.detection === 'Exposure change only' ? .065 : 0;
    const delta = target - this.exposureLevel;
    this.exposureLevel += Math.sign(delta) * Math.min(Math.abs(delta), dt * .13);
    if (this.exposureLevel <= 0) return;
    // One gentle half-second exposure fade, with no flashes or independent timer.
    this.ctx.save();
    this.ctx.fillStyle = `rgba(255, 239, 211, ${this.exposureLevel})`;
    this.ctx.fillRect(0, 0, W, H);
    this.ctx.restore();
  }

  background() {
    const c = this.ctx, n = this.night;
    c.fillStyle = n ? '#0b171b' : '#b5c8b8'; c.fillRect(0, 0, W, H);
    const glow = c.createRadialGradient(545, 305, 60, 545, 305, 620);
    glow.addColorStop(0, n ? '#19312d' : '#d2ddc5'); glow.addColorStop(1, n ? '#091116' : '#9db4a9');
    c.fillStyle = glow; c.fillRect(0, 0, W, H);
    c.strokeStyle = n ? '#24403925' : '#ffffff18'; c.lineWidth = .6;
    for (let x = -H; x < W + H; x += 48) {
      line(c, [[x, 0], [x + H * .5, H]]); c.stroke();
    }
    for (let y = 0; y < H; y += 42) { line(c, [[0, y], [W, y]]); c.stroke(); }
    c.fillStyle = n ? '#698a7213' : '#45634c13';
    for (let y = 12; y < H; y += 17) for (let x = 11; x < W; x += 23) {
      const jitter = ((x * 13 + y * 17) % 13);
      c.fillRect(x + jitter, y, 1.4, 1.4);
    }
  }

  roads() {
    const c = this.ctx, n = this.night;
    c.lineCap = 'round'; c.lineJoin = 'round';
    for (const lane of ['exit', 'entrance']) {
      route(c, lane); c.strokeStyle = n ? '#02090dbb' : '#55716344'; c.lineWidth = 126; c.stroke();
      route(c, lane); c.strokeStyle = n ? '#3c5150' : '#c6cbc1'; c.lineWidth = 112; c.stroke();
      route(c, lane); c.strokeStyle = n ? '#111d25' : '#606c70'; c.lineWidth = 104; c.stroke();
      route(c, lane); c.strokeStyle = n ? '#172731' : '#6b777b'; c.lineWidth = 92; c.stroke();
      c.save(); c.setLineDash([1, 20]); route(c, lane); c.strokeStyle = n ? '#31434a' : '#9aa4a3'; c.lineWidth = 1; c.stroke(); c.restore();
    }
    // The footpath is deliberately separate from the vehicle lanes.
    route(c, 'pedestrian', 0, .6); c.strokeStyle = n ? '#30433f' : '#e0dfcf'; c.lineWidth = 32; c.stroke();
    route(c, 'pedestrian', 0, .6); c.strokeStyle = n ? '#233733' : '#cbcebf'; c.lineWidth = 27; c.stroke();
    c.save(); c.setLineDash([1, 17]); route(c, 'pedestrian', 0, .6); c.strokeStyle = n ? '#61746a55' : '#8d9b8c66'; c.lineWidth = 26; c.stroke(); c.restore();
    // A small splitter island between the two barrier pedestals.
    c.fillStyle = n ? '#758376' : '#e2e5d6';
    rounded(c, 553, 331, 45, 74, 20); c.fill();
    c.fillStyle = n ? '#324e3c' : '#74986b'; rounded(c, 559, 337, 33, 62, 16); c.fill();
  }

  landscape() {
    const c = this.ctx, n = this.night;
    for (const [x, y, r] of TREE_POSITIONS) {
      c.fillStyle = '#00000029'; c.beginPath(); c.ellipse(x + 12, y + 20, r * .92, r * .52, -.3, 0, Math.PI * 2); c.fill();
      c.strokeStyle = n ? '#3c4b35' : '#76694c'; c.lineWidth = 4; line(c, [[x, y + 13], [x, y + 28]]); c.stroke();
      const leaf = c.createRadialGradient(x - r * .32, y - r * .45, 0, x, y, r);
      leaf.addColorStop(0, n ? '#365b46' : '#739b65'); leaf.addColorStop(1, n ? '#122c26' : '#386b48');
      c.fillStyle = leaf;
      c.beginPath();
      for (let i = 0; i <= 12; i++) {
        const a = i * Math.PI / 6, rr = r * (i % 2 ? .88 : 1);
        const xx = x + Math.cos(a) * rr, yy = y + Math.sin(a) * rr * .8;
        if (i === 0) c.moveTo(xx, yy); else c.lineTo(xx, yy);
      }
      c.closePath(); c.fill();
      c.strokeStyle = n ? '#80b39713' : '#daf2b52a'; c.lineWidth = 1; c.stroke();
      c.fillStyle = n ? '#678c5723' : '#b8cf7925'; c.beginPath(); c.ellipse(x - r * .22, y - r * .3, r * .42, r * .24, -.45, 0, Math.PI * 2); c.fill();
    }
  }

  building() {
    const c = this.ctx, n = this.night;
    c.save(); c.translate(300, 234);
    c.fillStyle = '#0000003d'; line(c, [[-36, 15], [91, -8], [120, 69], [4, 104], [-35, 64]]); c.closePath(); c.fill();
    c.fillStyle = n ? '#19333b' : '#829e9d'; line(c, [[-48, -2], [50, -28], [50, 56], [-48, 83]]); c.closePath(); c.fill();
    c.fillStyle = n ? '#0c252e' : '#607e83'; line(c, [[50, -28], [88, -3], [88, 81], [50, 56]]); c.closePath(); c.fill();
    for (let i = 0; i < 4; i++) {
      const xx = -38 + i * 21, yy = 12 - i * 5.6;
      const window = c.createLinearGradient(xx, yy, xx, yy + 46);
      window.addColorStop(0, n ? '#75c6bd' : '#badfdb'); window.addColorStop(1, n ? '#255453' : '#6d9e9e');
      c.fillStyle = window; line(c, [[xx, yy], [xx + 16, yy - 4.5], [xx + 16, yy + 44], [xx, yy + 48.5]]); c.closePath(); c.fill();
      c.strokeStyle = n ? '#09272f' : '#475e68'; c.lineWidth = 2; line(c, [[xx, yy + 21], [xx + 16, yy + 16.5]]); c.stroke();
    }
    c.fillStyle = n ? '#47615b' : '#c1ccbf'; line(c, [[-60, -8], [48, -39], [101, -7], [-10, 24]]); c.closePath(); c.fill();
    c.strokeStyle = n ? '#8ab29d55' : '#e1e6d8'; c.lineWidth = 2; c.stroke();
    c.fillStyle = n ? '#304840' : '#a6b9a6'; line(c, [[-60, -8], [-10, 24], [-10, 32], [-60, 1]]); c.closePath(); c.fill();
    c.fillStyle = n ? '#243c39' : '#90a69b'; line(c, [[-10, 24], [101, -7], [101, 2], [-10, 32]]); c.closePath(); c.fill();
    c.fillStyle = n ? '#a5d8cf' : '#233d3c'; c.font = '600 9px system-ui, sans-serif'; c.fillText('GUARD HOUSE', -39, 74);
    c.restore();
  }

  crossing() {
    const c = this.ctx;
    c.save();
    for (let t = .59; t < .94; t += .024) {
      const p = pose('pedestrian', t);
      c.save(); c.translate(p.x, p.y); c.rotate(p.angle);
      c.fillStyle = this.night ? '#a2b9b036' : '#d3ddd3aa'; c.fillRect(-4, -13, 8, 26); c.restore();
    }
    c.restore();
  }

  lamps() {
    const c = this.ctx;
    for (const [x, y] of [[449, 230], [732, 353], [269, 487], [568, 520], [795, 102]]) {
      if (this.night) {
        const g = c.createRadialGradient(x, y, 0, x, y, 87);
        g.addColorStop(0, '#ffeac315'); g.addColorStop(1, '#ffeac300');
        c.fillStyle = g; c.beginPath(); c.arc(x, y, 87, 0, Math.PI * 2); c.fill();
      }
      c.strokeStyle = this.night ? '#708482' : '#64746e'; c.lineWidth = 2.5;
      line(c, [[x, y + 15], [x, y - 15], [x + 10, y - 21]]); c.stroke();
      if (this.night) { c.shadowColor = '#ffe9bd'; c.shadowBlur = 12; }
      c.fillStyle = this.night ? '#ffecc4' : '#eef3e4'; rounded(c, x + 5, y - 24, 13, 4, 2); c.fill(); c.shadowBlur = 0;
    }
  }

  zones(stage) {
    const c = this.ctx;
    for (const lane of ['exit', 'entrance', 'pedestrian']) {
      c.save();
      route(c, lane, .16, lane === 'pedestrian' ? .96 : .73);
      c.lineWidth = lane === 'pedestrian' ? 34 : 70;
      c.strokeStyle = `${LANE_COLORS[lane]}${stage === 'zones' ? '23' : '11'}`; c.stroke();
      c.setLineDash([6, 7]);
      route(c, lane, .16, lane === 'pedestrian' ? .96 : .73);
      c.strokeStyle = `${LANE_COLORS[lane]}38`; c.lineWidth = 1.2; c.stroke();
      const p = pose(lane, .16);
      c.translate(p.x, p.y); c.rotate(p.angle);
      c.setLineDash([]); c.strokeStyle = `${LANE_COLORS[lane]}99`; c.lineWidth = 1.5;
      const width = lane === 'pedestrian' ? 19 : 37;
      line(c, [[0, -width], [0, width]]); c.stroke();
      c.restore();
    }
  }

  directionArrow(lane, progress) {
    const c = this.ctx, p = pose(lane, progress);
    c.save(); c.translate(p.x, p.y); c.rotate(p.angle);
    c.strokeStyle = this.night ? '#91aaa044' : '#d7e1d4a0'; c.lineWidth = 2; c.lineCap = 'square';
    line(c, [[-10, 0], [11, 0], [4, -6], [11, 0], [4, 6]]); c.stroke(); c.restore();
  }

  barrier(lane, up) {
    const c = this.ctx, p = pose(lane, .68), color = LANE_COLORS[lane];
    c.save(); c.translate(p.x, p.y); c.rotate(p.angle);
    c.fillStyle = '#00000044'; rounded(c, -9, -48, 20, 16, 3); c.fill();
    c.fillStyle = this.night ? '#446269' : '#8aa7a2'; rounded(c, -10, -51, 16, 14, 3); c.fill();
    c.fillStyle = color; rounded(c, -6, -49, 8, 3, 1); c.fill();
    c.strokeStyle = '#00000036'; c.lineWidth = 7;
    if (up) line(c, [[-2, -42], [-27, -73]]); else line(c, [[-2, -42], [-2, 42]]);
    c.stroke();
    c.strokeStyle = '#e0e7d9'; c.lineWidth = 5;
    if (up) line(c, [[-5, -45], [-30, -76]]); else line(c, [[-5, -45], [-5, 39]]);
    c.stroke(); c.setLineDash([5, 8]); c.strokeStyle = color; c.lineWidth = 5; c.stroke(); c.setLineDash([]);
    c.fillStyle = color; c.beginPath(); c.arc(-5, -45, 4, 0, Math.PI * 2); c.fill();
    c.restore();
  }

  object(object, snapshot) {
    const c = this.ctx, p = pose(object.lane, Number(object.progress) || 0);
    const color = LANE_COLORS[object.lane] || LANE_COLORS.exit;
    const person = object.kind === 'person';
    const truck = object.kind === 'truck';
    const length = person ? 16 : truck ? 66 : 44, width = person ? 15 : truck ? 25 : 22;
    // Progress is the approaching nose, so a waiting vehicle stops before the arm.
    if (!person) {
      p.x -= Math.cos(p.angle) * (length / 2 + 9);
      p.y -= Math.sin(p.angle) * (length / 2 + 9);
    }
    if (object.detected || object.occluded) {
      c.save(); c.setLineDash(object.occluded ? [3, 5] : []);
      route(c, object.lane, Math.max(0, object.progress - .13), object.progress);
      c.strokeStyle = `${color}55`; c.lineWidth = 2; c.stroke(); c.restore();
    }
    c.save(); c.translate(p.x, p.y); c.rotate(p.angle);
    c.globalAlpha = object.occluded ? .38 : 1;
    if (person) this.person(snapshot.time, object.id); else this.vehicle(truck, object, length, width);
    c.restore();
    if (object.detected || object.occluded) {
      const bx = Math.abs(Math.cos(p.angle)) * length / 2 + Math.abs(Math.sin(p.angle)) * width / 2 + 8;
      const by = Math.abs(Math.sin(p.angle)) * length / 2 + Math.abs(Math.cos(p.angle)) * width / 2 + 8;
      c.save(); c.strokeStyle = color; c.lineWidth = 1.3;
      if (object.occluded) { c.setLineDash([4, 5]); c.globalAlpha = .65; }
      const corner = 8;
      for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
        line(c, [[p.x + sx * (bx - corner), p.y + sy * by], [p.x + sx * bx, p.y + sy * by], [p.x + sx * bx, p.y + sy * (by - corner)]]); c.stroke();
      }
      c.setLineDash([]);
      const active = snapshot.alarm?.trackId === object.id;
      const detail = object.occluded ? ' · HELD' : object.acknowledged ? ' · ACK' : active ? ' · ALERT' : '';
      const label = `${object.id}${detail}`;
      c.font = '600 10px ui-monospace, SFMono-Regular, Consolas, monospace';
      const tw = c.measureText(label).width + 12;
      const lx = clamp(p.x - bx, 5, W - tw - 5), ly = clamp(p.y - by - 23, 5, H - 23);
      c.fillStyle = active ? '#2c2320ed' : '#0b1721ed'; rounded(c, lx, ly, tw, 18, 4); c.fill();
      c.fillStyle = color; c.fillText(label, lx + 6, ly + 12.5);
      c.restore();
    }
  }

  vehicle(truck, object, length, width) {
    const c = this.ctx, n = this.night, l = length / 2;
    if (n && !object.occluded) {
      for (const side of [-1, 1]) {
        const beam = c.createLinearGradient(l, 0, l + 105, 0);
        beam.addColorStop(0, '#fff2c829'); beam.addColorStop(1, '#fff2c800');
        c.fillStyle = beam; line(c, [[l - 2, side * 7], [l + 115, side * 7 - 29], [l + 115, side * 7 + 29]]); c.closePath(); c.fill();
      }
    }
    c.fillStyle = '#0000005a'; rounded(c, -l + 4, -width / 2 + 4, length, width, 6); c.fill();
    c.fillStyle = '#061017';
    for (const x of [-l + 8, l - 10]) for (const y of [-width / 2 - 2, width / 2 - 1]) { rounded(c, x - 4, y, 8, 4, 1.5); c.fill(); }
    const palettes = object.lane === 'entrance' ? ['#e4e9df', '#9fb9c1', '#d9b17c'] : ['#83cabb', '#e0e7df', '#9caee1'];
    const last = Number(String(object.id).replace(/\D/g, '')) || 0;
    const paint = palettes[last % palettes.length];
    const body = c.createLinearGradient(0, -width / 2, 0, width / 2);
    body.addColorStop(0, paint); body.addColorStop(1, n ? '#567a83' : '#78989b');
    c.fillStyle = body; rounded(c, -l, -width / 2, length, width, truck ? 3 : 7); c.fill();
    c.strokeStyle = '#edfff04d'; c.lineWidth = 1; c.stroke();
    if (truck) {
      c.fillStyle = '#b7cbc2'; rounded(c, -l + 2, -width / 2 + 1, length - 18, width - 2, 2); c.fill();
      c.strokeStyle = '#718e8d55'; c.lineWidth = 1;
      for (let x = -l + 8; x < l - 17; x += 7) { line(c, [[x, -width / 2 + 2], [x, width / 2 - 2]]); c.stroke(); }
      c.fillStyle = '#234653'; rounded(c, l - 12, -width / 2 + 3, 6, width - 6, 2); c.fill();
    } else {
      c.fillStyle = n ? '#203f50' : '#345a6a';
      line(c, [[7, -width / 2 + 2], [12, -width / 2 + 5], [12, width / 2 - 5], [7, width / 2 - 2]]); c.closePath(); c.fill();
      line(c, [[-9, -width / 2 + 3], [-14, -width / 2 + 6], [-14, width / 2 - 6], [-9, width / 2 - 3]]); c.closePath(); c.fill();
      c.fillStyle = paint; rounded(c, -8, -width / 2 + 3, 14, width - 6, 4); c.fill();
      c.strokeStyle = '#ffffff33'; c.lineWidth = 1; line(c, [[-6, -width / 2 + 4], [4, -width / 2 + 4]]); c.stroke();
    }
    c.fillStyle = n ? '#fff6d6' : '#f2f6d8';
    rounded(c, l - 3, -width / 2 + 2, 3, 5, 1); c.fill(); rounded(c, l - 3, width / 2 - 7, 3, 5, 1); c.fill();
    c.fillStyle = '#ff6f61';
    rounded(c, -l, -width / 2 + 2, 3, 5, 1); c.fill(); rounded(c, -l, width / 2 - 7, 3, 5, 1); c.fill();
  }

  person(time, id) {
    const c = this.ctx;
    const stride = Math.sin((Number(time) || 0) * 7 + String(id).length) * 4;
    c.fillStyle = '#00000040'; c.beginPath(); c.ellipse(3, 4, 10, 6, 0, 0, Math.PI * 2); c.fill();
    c.strokeStyle = '#596a8f'; c.lineWidth = 3; c.lineCap = 'round';
    line(c, [[-2, -3], [-7 + stride, -4], [-2, -3], [-7 - stride, 4]]); c.stroke();
    c.fillStyle = '#b7a4f4'; c.beginPath(); c.ellipse(0, 0, 6, 7, 0, 0, Math.PI * 2); c.fill();
    c.strokeStyle = '#baabff'; c.lineWidth = 2.5;
    line(c, [[2, -5], [stride, -9], [2, -5]]); c.stroke(); line(c, [[2, 5], [-stride, 9]]); c.stroke();
    c.fillStyle = '#edcaa1'; c.beginPath(); c.arc(3, 0, 4, 0, Math.PI * 2); c.fill();
  }

  mapLabels() {
    const c = this.ctx;
    const labels = [[725, 70, 'ENTRANCE APPROACH', 'entrance'], [64, 583, 'EXIT APPROACH', 'exit'], [77, 342, 'FOOTPATH', 'pedestrian']];
    for (const [x, y, text, lane] of labels) {
      c.font = '600 10px ui-monospace, SFMono-Regular, Consolas, monospace';
      const w = c.measureText(text).width + 29;
      c.fillStyle = this.night ? '#09151cdb' : '#e0ece0e8'; rounded(c, x, y, w, 24, 5); c.fill();
      c.fillStyle = this.night ? LANE_COLORS[lane] : '#365954'; c.fillRect(x + 8, y + 10, 4, 4); c.fillText(text, x + 19, y + 15.5);
    }
    c.fillStyle = this.night ? '#688078' : '#496b62'; c.font = '10px ui-monospace, SFMono-Regular, Consolas, monospace'; c.fillText('SYNTHETIC CAMPUS / NOT A CAMERA FEED', 22, 27);
    c.save(); c.translate(894, 577); c.strokeStyle = this.night ? '#59776c' : '#517467'; c.lineWidth = 1;
    line(c, [[0, 10], [0, -14], [-4, -6], [0, -14], [4, -6]]); c.stroke(); c.fillStyle = this.night ? '#7f978b' : '#405f51'; c.font = '9px ui-monospace, monospace'; c.fillText('N', -3, -21); c.restore();
  }

  frameCorners() {
    const c = this.ctx;
    c.strokeStyle = this.night ? '#93bdad22' : '#59746944'; c.lineWidth = 1;
    for (const [x, y, dx, dy] of [[12, 12, 1, 1], [948, 12, -1, 1], [12, 608, 1, -1], [948, 608, -1, -1]]) {
      line(c, [[x + dx * 15, y], [x, y], [x, y + dy * 15]]); c.stroke();
    }
  }

  destroy() { /* No timers, listeners, footage, or resources are retained. */ }
}
