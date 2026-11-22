__c500Modules["zones.js"] = (() => {
'use strict';
const SVG_NS = 'http://www.w3.org/2000/svg';
const WIDTH = 640;
const HEIGHT = 360;
const EPSILON = 1e-8;
const MAX_POLYGON_CORNERS = __MAX_POLYGON_CORNERS__;
const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
const copy = value => JSON.parse(JSON.stringify(value));
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function orientation(a, b, c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function onSegment(a, b, point) {
  return Math.abs(orientation(a, b, point)) <= EPSILON &&
    point[0] >= Math.min(a[0], b[0]) - EPSILON && point[0] <= Math.max(a[0], b[0]) + EPSILON &&
    point[1] >= Math.min(a[1], b[1]) - EPSILON && point[1] <= Math.max(a[1], b[1]) + EPSILON;
}

function intersects(a, b, c, d) {
  const abC = orientation(a, b, c);
  const abD = orientation(a, b, d);
  const cdA = orientation(c, d, a);
  const cdB = orientation(c, d, b);
  if (((abC > EPSILON && abD < -EPSILON) || (abC < -EPSILON && abD > EPSILON)) &&
      ((cdA > EPSILON && cdB < -EPSILON) || (cdA < -EPSILON && cdB > EPSILON))) return true;
  return onSegment(a, b, c) || onSegment(a, b, d) || onSegment(c, d, a) || onSegment(c, d, b);
}

// Exported for the isolated preview's geometry checks. No detector settings are changed here.
function validatePolygon(polygon) {
  if (!Array.isArray(polygon) || polygon.length < 3) return 'A region needs at least three points.';
  if (polygon.length > MAX_POLYGON_CORNERS) return `A region supports at most ${MAX_POLYGON_CORNERS} points.`;
  if (polygon.some(point => !Array.isArray(point) || point.length !== 2 ||
      point.some(value => !Number.isFinite(value) || value < 0 || value > 1))) return 'Keep every point inside the image.';
  let twiceArea = 0;
  for (let i = 0; i < polygon.length; i++) {
    const current = polygon[i];
    const next = polygon[(i + 1) % polygon.length];
    if (Math.hypot(current[0] - next[0], current[1] - next[1]) < 0.00001) return 'Two points overlap.';
    twiceArea += current[0] * next[1] - next[0] * current[1];
    for (let j = i + 1; j < polygon.length; j++) {
      if (j === i + 1 || (i === 0 && j === polygon.length - 1)) continue;
      if (intersects(current, next, polygon[j], polygon[(j + 1) % polygon.length])) return 'Region edges cross. Move a point or undo.';
    }
  }
  if (Math.abs(twiceArea) < 2 / (WIDTH * HEIGHT)) return 'The region is too small.';
  return null;
}

const icon = (name) => {
  const paths = {
    undo: '<path d="M9 4 4 9l5 5"/><path d="M4 9h9a7 7 0 0 1 0 14" transform="translate(0 -3)"/>',
    reset: '<path d="M3 10a9 9 0 1 1 2 8M3 4v6h6"/>',
    export: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    minus: '<path d="M5 12h14"/>',
    shape: '<path d="m5 5 14-2 2 14-15 4Z"/><circle cx="5" cy="5" r="2"/><circle cx="19" cy="3" r="2"/><circle cx="21" cy="17" r="2"/><circle cx="6" cy="21" r="2"/>'
  };
  return `<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${paths[name] || paths.shape}</svg>`;
};

function zoneRole(zone) {
  const role = String(zone.route_role || '').toLowerCase();
  if (role === 'pedestrian' || /pedestrian|footpath/i.test(zone.name)) return 'pedestrian';
  if (role === 'entrance' || /entrance|top road/i.test(zone.name)) return 'entrance';
  return 'exit';
}

const geometryOf = zones => (zones || []).map(({ name, polygon, exclude_polygons }) => ({
  name, polygon: copy(polygon), ...(exclude_polygons ? { exclude_polygons: copy(exclude_polygons) } : {})
}));
const geometryKey = zones => JSON.stringify((zones || []).map(({ name, polygon, exclude_polygons }) => ({
  name, polygon, exclude_polygons: exclude_polygons || []
})));

function initZones({ imageUrl = '/api/scene.jpg', getState = () => null, onToast = () => {}, onApply = null, production = false } = {}) {
  const host = document.querySelector('#zone-workspace');
  if (!host) throw new Error('The zone editor needs #zone-workspace.');
  host.innerHTML = `
    <div class="zone-layout">
      <aside class="zone-sidebar" aria-label="Detection regions">
        <div class="zone-sidebar-heading"><span>REGIONS</span><span class="zone-count">—</span></div>
        <div class="zone-list" role="group" aria-label="Choose a region"></div>
        <div class="zone-sidebar-foot"><span class="zone-draft-dot"></span><span class="zone-draft-status">Live layout</span></div>
      </aside>
      <div class="zone-editor">
        <div class="zone-editor-heading">
          <div><span class="zone-eyebrow">REGION EDITOR</span><h2 class="zone-current-title">Waiting for camera</h2></div>
          <span class="zone-preview-badge">Preview · edits stay here</span>
        </div>
        <div class="zone-source-update" hidden><span>Source changed</span><button type="button" class="zone-tool" data-zone-action="reload">Load update</button></div>
        <div class="zone-canvas-wrap">
          <img id="zone-image" alt="Aligned campus camera view for region editing" draggable="false">
          <svg class="zone-canvas" viewBox="0 0 640 360" aria-label="Region polygons. Select a point and use arrow keys to move it." role="group">
            <defs class="zone-definitions"></defs><g class="zone-polygons"></g><g class="zone-exclusions"></g><g class="zone-confirmations"></g><g class="zone-handles"></g>
          </svg>
          <div class="zone-canvas-empty">Connecting to the scene</div>
          <div class="zone-canvas-caption"><span class="zone-point-count">—</span><span class="zone-zoom-note">640 × 360</span></div>
        </div>
        <p class="zone-confirmation-legend"><span></span>Gold dashes: saved alert area · locked. Outer outline: tracking area.</p>
        <div class="zone-edit-footer">
          <div class="zone-vertex-tools">
            <button class="zone-tool" data-zone-action="add" title="Insert a point on the longest edge">${icon('plus')}<span>Add point</span></button>
            <button class="zone-tool" data-zone-action="remove" title="Remove the selected point">${icon('minus')}<span>Remove point</span></button>
            <span class="zone-keyboard-hint">Drag points · arrow keys to refine</span>
          </div>
          <div class="zone-history-tools">
            <button class="zone-tool" data-zone-action="undo" title="Undo the last edit">${icon('undo')}<span>Undo</span></button>
            <button class="zone-tool" data-zone-action="reset" title="Restore the loaded source layout">${icon('reset')}<span>Reset</span></button>
            <button class="zone-export" data-zone-action="export">${icon('export')}<span>Export layout</span></button>
            ${production ? '<button type="button" class="zone-apply" data-zone-action="apply">Apply layout</button>' : ''}
          </div>
        </div>
        <p class="zone-validation" role="status" aria-live="polite"></p>
        ${production ? '<p class="zone-apply-hint" role="status" aria-live="polite"></p>' : ''}
      </div>
    </div>`;

  const query = selector => host.querySelector(selector);
  const list = query('.zone-list');
  const svg = query('.zone-canvas');
  const polygons = query('.zone-polygons');
  const definitions = query('.zone-definitions');
  const exclusions = query('.zone-exclusions');
  const confirmations = query('.zone-confirmations');
  const handles = query('.zone-handles');
  const picture = query('#zone-image');
  const canvasEmpty = query('.zone-canvas-empty');
  let initial = null;
  let draft = [];
  let active = 0;
  let selected = -1;
  let history = [];
  let drag = null;
  let imageLoaded = false;
  let lastImageUrl = imageUrl;
  let latestSource = null;
  let latestState = null;
  let lastStateReceived = 0;
  let sourceCapturedAt = null;
  let busy = false;
  let awaitingVerification = false;
  let sourceAvailable = false;
  const importedExclusions = new Set();

  const current = () => draft[active];
  const remember = snapshot => {
    history.push(copy(snapshot));
    if (history.length > 80) history.shift();
  };
  const dirty = () => initial && geometryKey(initial) !== geometryKey(draft);
  const sourceChanged = () => initial && latestSource && geometryKey(initial) !== geometryKey(latestSource);
  const allValid = () => draft.length > 0 && new Set(draft.map(zone => zone.name)).size === draft.length &&
    draft.every(zone => !validatePolygon(zone.polygon) && !(zone.exclude_polygons || []).some(validatePolygon));
  const sourceNamesMatch = () => initial && equal(initial.map(zone => zone.name), draft.map(zone => zone.name));
  const freshState = () => latestState && latestState._uiConnected !== false && Number.isFinite(latestState.frames) && latestState.frames > 0 &&
    Number.isFinite(latestState.last_frame_age) && latestState.last_frame_age >= 0 && latestState.last_frame_age <= 5 &&
    Date.now() - lastStateReceived < 5000;
  const canApply = () => production && typeof onApply === 'function' && initial && dirty() && allValid() &&
    sourceNamesMatch() && sourceAvailable && freshState() && latestState.armed === false && !sourceChanged() && !busy && !drag && !awaitingVerification;

  function saveEdit(mutator) {
    if (!initial || drag || busy) return;
    const before = copy(draft);
    mutator();
    if (!equal(before, draft)) remember(before);
    draw();
  }

  function status() {
    const changed = Boolean(dirty());
    const outdated = Boolean(sourceChanged());
    query('.zone-draft-status').textContent = outdated ? (changed ? 'Draft from previous layout' : 'Source changed') : changed ? 'Local draft' : 'Live layout';
    query('.zone-preview-badge').textContent = outdated ? 'Source changed' : production ? (changed ? 'Unsaved layout' : 'Live layout') : (changed ? 'Local draft · preview' : 'Preview · edits stay here');
    host.classList.toggle('zone-is-dirty', changed);
    host.classList.toggle('zone-source-outdated', outdated);
    host.classList.toggle('zone-is-busy', busy);
    host.setAttribute('aria-busy', String(busy));
    query('[data-zone-action="undo"]').disabled = !history.length || busy;
    query('[data-zone-action="reset"]').disabled = !changed || busy;
    query('[data-zone-action="add"]').disabled = !current() || current().polygon.length >= MAX_POLYGON_CORNERS || busy;
    query('[data-zone-action="remove"]').disabled = !current() || selected < 0 || current().polygon.length <= 3 || busy;
    query('[data-zone-action="export"]').disabled = !draft.length || busy;
    query('.zone-source-update').hidden = !sourceChanged();
    query('[data-zone-action="reload"]').disabled = busy || Boolean(drag);
    query('.zone-point-count').textContent = current() ? `${current().polygon.length} points${current().exclude_polygons?.length ? ' · exclusion locked' : ''}` : 'No regions';
    const error = current() ? validatePolygon(current().polygon) : null;
    query('.zone-validation').textContent = error || '';
    host.classList.toggle('zone-invalid', Boolean(error));
    if (production) {
      query('[data-zone-action="apply"]').disabled = !canApply();
      query('[data-zone-action="apply"]').textContent = busy ? 'Applying…' : 'Apply layout';
      query('.zone-apply-hint').textContent = busy ? 'Saving layout' : awaitingVerification ? 'Waiting to verify the saved layout' :
        sourceChanged() ? 'Load the latest layout before applying' : !freshState() || !sourceAvailable ? 'Waiting for a fresh camera reading' :
        latestState?.armed !== false ? 'Pause alerts to apply' : !sourceNamesMatch() ? 'Region names changed · reset this draft' : '';
    }
    for (const handle of handles.children) {
      const index = Number(handle.dataset.vertex);
      const point = current().polygon[index];
      handle.classList.toggle('is-selected', index === selected);
      handle.setAttribute('aria-pressed', String(index === selected));
      handle.setAttribute('aria-label', `Point ${index + 1}, ${Math.round(point[0] * WIDTH)} by ${Math.round(point[1] * HEIGHT)} pixels. Arrow keys move; Shift moves ten pixels.`);
    }
  }

  function selectZone(index, focus = false) {
    if (drag || busy || index < 0 || index >= draft.length) return;
    active = index;
    selected = -1;
    draw();
    if (focus) list.children[index]?.focus();
  }

  function makeSvg(tag, attributes) {
    const element = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
    return element;
  }

  function updateGeometry() {
    for (const polygon of polygons.children) {
      const index = Number(polygon.dataset.zone);
      const loops = [draft[index].polygon, ...(draft[index].exclude_polygons || [])];
      polygon.setAttribute('d', loops.map(loop => 'M' + loop.map(([x, y]) => `${x * WIDTH},${y * HEIGHT}`).join(' L') + ' Z').join(' '));
      definitions.querySelector(`#zone-clip-${index} polygon`).setAttribute('points', draft[index].polygon.map(([x, y]) => `${x * WIDTH},${y * HEIGHT}`).join(' '));
    }
    for (const handle of handles.children) {
      const [x, y] = current().polygon[Number(handle.dataset.vertex)];
      handle.setAttribute('transform', `translate(${x * WIDTH} ${y * HEIGHT})`);
    }
    status();
  }

  function drawConfirmations() {
    confirmations.replaceChildren();
    for (const zone of Array.isArray(latestState?.zones) ? latestState.zones : []) {
      const points = zone?.confirmation_polygon;
      if (!Array.isArray(points) || validatePolygon(points)) continue;
      const outline = makeSvg('polygon', { class: 'zone-confirmation', points: points.map(([x, y]) => `${x * WIDTH},${y * HEIGHT}`).join(' '), 'vector-effect': 'non-scaling-stroke' });
      const title = makeSvg('title', {});
      title.textContent = `${zone.name} · saved alert area (locked)`;
      outline.append(title);confirmations.append(outline);
    }
  }

  function draw() {
    list.replaceChildren();
    polygons.replaceChildren();
    definitions.replaceChildren();
    exclusions.replaceChildren();
    handles.replaceChildren();
    drawConfirmations();
    query('.zone-count').textContent = String(draft.length || '—');
    query('.zone-current-title').textContent = current()?.name || 'Waiting for camera';
    draft.forEach((zone, index) => {
      const role = zoneRole(zone);
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `zone-select zone-role-${role}${index === active ? ' is-active' : ''}`;
      button.setAttribute('aria-pressed', String(index === active));
      const marker = document.createElement('span');
      marker.className = 'zone-select-number';
      marker.textContent = String(index + 1).padStart(2, '0');
      const label = document.createElement('span');
      label.className = 'zone-select-label';
      label.textContent = zone.name;
      const arrow = document.createElement('span');
      arrow.className = 'zone-select-arrow';
      arrow.setAttribute('aria-hidden', 'true');
      arrow.textContent = '↗';
      button.append(marker, label, arrow);
      button.addEventListener('click', () => selectZone(index, true));
      button.addEventListener('keydown', event => {
        const next = { ArrowDown: (index + 1) % draft.length, ArrowUp: (index + draft.length - 1) % draft.length, Home: 0, End: draft.length - 1 }[event.key];
        if (next !== undefined) { event.preventDefault(); selectZone(next, true); }
      });
      list.append(button);
    });
    // Draw the selected region last so its edges stay legible where routes overlap.
    const order = draft.map((_, index) => index).filter(index => index !== active);
    if (draft.length) order.push(active);
    for (const index of order) {
      const clip = makeSvg('clipPath', { id: `zone-clip-${index}` });
      clip.append(makeSvg('polygon', {}));
      definitions.append(clip);
      const polygon = makeSvg('path', { 'data-zone': index, class: `zone-polygon zone-role-${zoneRole(draft[index])}${index === active ? ' is-active' : ''}`, 'vector-effect': 'non-scaling-stroke', 'fill-rule': 'evenodd', 'clip-path': `url(#zone-clip-${index})` });
      polygon.addEventListener('click', () => selectZone(index));
      const title = makeSvg('title', {});
      title.textContent = draft[index].name;
      polygon.append(title);
      polygons.append(polygon);
    }
    for (const hole of current()?.exclude_polygons || []) {
      const exclusion = makeSvg('polygon', { class: 'zone-exclusion', points: hole.map(([x, y]) => `${x * WIDTH},${y * HEIGHT}`).join(' '), 'vector-effect': 'non-scaling-stroke' });
      const title = makeSvg('title', {});
      title.textContent = 'Ignored area · locked in this preview';
      exclusion.append(title);
      exclusions.append(exclusion);
    }
    current()?.polygon.forEach((_, index) => {
      const handle = makeSvg('g', { 'data-vertex': index, tabindex: 0, role: 'button', class: `zone-handle zone-role-${zoneRole(current())}` });
      handle.append(makeSvg('circle', { r: 10, class: 'zone-handle-hit' }));
      handle.append(makeSvg('circle', { r: 3.3, class: 'zone-handle-dot', 'vector-effect': 'non-scaling-stroke' }));
      handle.addEventListener('focus', () => { selected = index; status(); });
      handle.addEventListener('pointerdown', event => {
        if (event.button !== 0 || drag || busy) return;
        event.preventDefault();
        selected = index;
        handle.focus({ preventScroll: true });
        drag = { pointerId: event.pointerId, before: copy(draft), index };
        svg.setPointerCapture(event.pointerId);
        svg.classList.add('is-dragging');
        status();
      });
      handle.addEventListener('keydown', event => {
        if (event.key === 'Delete' || event.key === 'Backspace') {
          event.preventDefault();
          event.stopPropagation();
          removePoint();
          return;
        }
        const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        if (!directions[event.key] || drag || busy) return;
        event.preventDefault();
        event.stopPropagation();
        const before = copy(draft);
        const [dx, dy] = directions[event.key];
        const step = event.shiftKey ? 10 : 1;
        const point = current().polygon[index];
        point[0] = clamp(point[0] + dx * step / WIDTH);
        point[1] = clamp(point[1] + dy * step / HEIGHT);
        if (!equal(before, draft)) remember(before);
        selected = index;
        updateGeometry();
      });
      handles.append(handle);
    });
    updateGeometry();
  }

  function finishDrag(cancel = false) {
    if (!drag) return;
    const completed = drag;
    drag = null;
    if (cancel) draft = completed.before;
    else if (!equal(completed.before, draft)) remember(completed.before);
    if (svg.hasPointerCapture(completed.pointerId)) svg.releasePointerCapture(completed.pointerId);
    svg.classList.remove('is-dragging');
    updateGeometry();
  }

  svg.addEventListener('pointermove', event => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const matrix = svg.getScreenCTM();
    if (!matrix) return;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    current().polygon[drag.index] = [clamp(point.x / WIDTH), clamp(point.y / HEIGHT)];
    updateGeometry();
  });
  svg.addEventListener('pointerup', event => { if (event.pointerId === drag?.pointerId) finishDrag(); });
  svg.addEventListener('pointercancel', () => finishDrag(true));
  svg.addEventListener('lostpointercapture', () => finishDrag());
  host.addEventListener('keydown', event => {
    if (event.key === 'Escape' && drag) { event.preventDefault(); finishDrag(true); }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z' && !event.shiftKey) {
      event.preventDefault();
      undo();
    }
  });

  function undo() {
    if (busy) return;
    if (drag) { finishDrag(true); return; }
    if (!history.length) return;
    draft = history.pop();
    // Exclusions are read-only. Undo restores outline edits, never an older mask.
    for (const zone of draft) {
      const authoritative = initial?.find(candidate => candidate.name === zone.name);
      if (!authoritative) continue;
      if (Object.hasOwn(authoritative, 'exclude_polygons')) zone.exclude_polygons = copy(authoritative.exclude_polygons);
      else delete zone.exclude_polygons;
    }
    active = Math.min(active, draft.length - 1);
    selected = Math.min(selected, (current()?.polygon.length || 0) - 1);
    draw();
    onToast('Last edit undone');
  }

  function removePoint() {
    if (!current() || selected < 0 || current().polygon.length <= 3 || drag || busy) return;
    saveEdit(() => { current().polygon.splice(selected, 1); selected = Math.min(selected, current().polygon.length - 1); });
    handles.children[selected]?.focus();
  }

  function exportLayout() {
    if (drag || !draft.length) return;
    for (let index = 0; index < draft.length; index++) {
      const error = validatePolygon(draft[index].polygon);
      const exclusionError = (draft[index].exclude_polygons || []).map(validatePolygon).find(Boolean);
      if (error || exclusionError) {
        selectZone(index);
        onToast(`${draft[index].name}: ${error || 'Invalid exclusion polygon.'}`);
        return;
      }
    }
    const payload = {
      schema: 'c500-zone-layout-v1',
      source_captured_at: sourceCapturedAt,
      source_changed: Boolean(sourceChanged()),
      note: 'Local preview layout. Normalized coordinates on the aligned 640 × 360 scene. Does not change the running system.',
      zones: draft.map(({ name, polygon, exclude_polygons }) => ({ name, polygon: polygon.map(point => point.map(value => Number(value.toFixed(7)))), ...(exclude_polygons ? { exclude_polygons: copy(exclude_polygons) } : {}) }))
    };
    const blob = new Blob([JSON.stringify(payload, null, 2) + '\n'], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'c500-zone-layout-draft.json';
    link.hidden = true;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    onToast('Layout exported · running system unchanged');
  }

  host.addEventListener('click', event => {
    const button = event.target.closest('[data-zone-action]');
    if (!button || button.disabled) return;
    const action = button.dataset.zoneAction;
    if (action === 'undo') undo();
    if (action === 'reload') loadUpdate();
    if (action === 'apply') void applyLayout();
    if (action === 'remove') removePoint();
    if (action === 'reset' && initial) {
      saveEdit(() => { draft = copy(initial); selected = -1; });
      onToast('Loaded source layout restored');
    }
    if (action === 'export') exportLayout();
    if (action === 'add' && current() && current().polygon.length < MAX_POLYGON_CORNERS) {
      saveEdit(() => {
        const polygon = current().polygon;
        let longest = 0;
        let edge = 0;
        polygon.forEach((point, index) => {
          const next = polygon[(index + 1) % polygon.length];
          const length = Math.hypot((point[0] - next[0]) * WIDTH, (point[1] - next[1]) * HEIGHT);
          if (length > longest) { longest = length; edge = index; }
        });
        const a = polygon[edge];
        const b = polygon[(edge + 1) % polygon.length];
        polygon.splice(edge + 1, 0, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
        selected = edge + 1;
      });
      handles.children[selected]?.focus();
    }
  });

  picture.addEventListener('load', () => { imageLoaded = true; canvasEmpty.hidden = true; });
  picture.addEventListener('error', () => { if (!imageLoaded) canvasEmpty.textContent = 'Camera view unavailable'; });

  function updateImage(src) {
    if (typeof src !== 'string' || !src) return;
    lastImageUrl = src;
    if (picture.getAttribute('src') !== src) picture.src = src;
  }

  function normalizeSource(value) {
    const zones = Array.isArray(value) ? value : value?.zones;
    if (!Array.isArray(zones) || !zones.length || new Set(zones.map(zone => zone?.name)).size !== zones.length) return null;
    if (zones.some(zone => !zone || typeof zone.name !== 'string' || !zone.name.trim() || validatePolygon(zone.polygon) ||
      (Object.hasOwn(zone, 'exclude_polygons') && (!Array.isArray(zone.exclude_polygons) || zone.exclude_polygons.some(validatePolygon))))) return null;
    return zones.map(zone => {
      const local = { name: zone.name, polygon: copy(zone.polygon), route_role: zone.route_role || '' };
      if (Array.isArray(zone.exclude_polygons)) {
        local.exclude_polygons = copy(zone.exclude_polygons);
      } else {
        // State may omit config-only masks. Keep a known mask only for identical source geometry.
        const previous = (latestSource || initial || []).find(candidate => candidate.name === zone.name);
        if (previous?.exclude_polygons && equal(previous.polygon, local.polygon)) local.exclude_polygons = copy(previous.exclude_polygons);
      }
      return local;
    });
  }

  function rebase(source, keepUndo = false) {
    const activeName = current()?.name;
    if (keepUndo && draft.length) remember(draft);
    draft = copy(source);
    initial = copy(draft);
    latestSource = copy(source);
    importedExclusions.clear();
    for (const zone of source) if (Object.hasOwn(zone, 'exclude_polygons')) importedExclusions.add(zone.name);
    active = Math.max(0, draft.findIndex(zone => zone.name === activeName));
    selected = -1;
    sourceCapturedAt = new Date().toISOString();
    awaitingVerification = false;
    sourceAvailable = true;
    draw();
  }

  function loadUpdate() {
    if (!latestSource || drag || busy) return;
    rebase(latestSource, true);
    onToast('Latest layout loaded · previous draft is in Undo');
  }

  async function applyLayout() {
    render(getState());
    if (!canApply()) { status(); return; }
    const submitted = geometryOf(draft);
    const baseline = geometryOf(initial);
    busy = true;
    status();
    try {
      const result = await onApply(copy(submitted), copy(baseline));
      if (result?.ok === false) throw new Error(result.error || 'The layout could not be saved.');
      const authoritative = normalizeSource(result?.state || result);
      if (!authoritative) {
        awaitingVerification = true;
        onToast('Save returned · waiting for the live layout to verify it');
      } else {
        if (result?.state && typeof result.state.armed === 'boolean') {
          latestState = result.state;
          lastStateReceived = Date.now();
        }
        rebase(authoritative, true);
        onToast('Layout applied');
      }
    } catch (error) {
      onToast(error instanceof Error ? error.message : 'The layout could not be saved.');
    } finally {
      busy = false;
      status();
    }
  }

  function render(state) {
    latestState = state;
    drawConfirmations();
    lastStateReceived = Date.now();
    const source = normalizeSource(state);
    sourceAvailable = Boolean(source);
    if (!source) { status(); return; }
    if (!initial) {
      if (!Number.isFinite(state.frames) || state.frames <= 0 || !Number.isFinite(state.last_frame_age) || state.last_frame_age > 10) { status(); return; }
      rebase(source);
      return;
    }
    latestSource = source;
    if (awaitingVerification && geometryKey(source) === geometryKey(draft) && !busy) {
      rebase(source, true);
      onToast('Saved layout verified');
      return;
    }
    if (drag || busy) { status(); return; }
    // The first state may arrive before config-only exclusions. An untouched outline
    // can adopt those masks without becoming a stale draft or disturbing selection.
    const primaryGeometry = zones => JSON.stringify(zones.map(({ name, polygon }) => ({ name, polygon })));
    if (!dirty() && sourceChanged() && primaryGeometry(initial) === primaryGeometry(source)) {
      for (const snapshot of [initial, draft, ...history]) {
        for (const zone of snapshot) {
          const authoritative = source.find(candidate => candidate.name === zone.name);
          if (!authoritative) continue;
          if (Object.hasOwn(authoritative, 'exclude_polygons')) zone.exclude_polygons = copy(authoritative.exclude_polygons);
          else delete zone.exclude_polygons;
        }
      }
      importedExclusions.clear();
      for (const zone of source) if (Object.hasOwn(zone, 'exclude_polygons')) importedExclusions.add(zone.name);
      sourceCapturedAt = new Date().toISOString();
      draw();
      return;
    }
    let updatedMask = false;
    for (const zone of source) {
      if (!Object.hasOwn(zone, 'exclude_polygons') || importedExclusions.has(zone.name)) continue;
      const original = initial.find(item => item.name === zone.name);
      const local = draft.find(item => item.name === zone.name);
      if (!original || !local || !equal(original.polygon, zone.polygon)) continue;
      original.exclude_polygons = copy(zone.exclude_polygons);
      local.exclude_polygons = copy(zone.exclude_polygons);
      for (const snapshot of history) {
        const item = snapshot.find(candidate => candidate.name === zone.name);
        if (item) item.exclude_polygons = copy(zone.exclude_polygons);
      }
      importedExclusions.add(zone.name);
      updatedMask = true;
    }
    if (updatedMask) draw();
    else status();
  }

  function selectRoute(role) {
    const aliases = { entry: 'entrance', pedestrians: 'pedestrian', footpath: 'pedestrian' };
    const wanted = aliases[role] || role;
    const index = draft.findIndex(zone => zoneRole(zone) === wanted);
    if (index >= 0) selectZone(index);
    return index >= 0;
  }

  function activate() {
    render(getState());
    const url = new URL(lastImageUrl, window.location.href);
    url.searchParams.set('zone_preview', String(Date.now()));
    updateImage(url.href);
  }

  function deactivate() { finishDrag(true); }

  draw();
  render(getState());
  return { render, activate, deactivate, updateImage, selectRoute, getDraft: () => geometryOf(draft), getSourceGeometry: () => geometryOf(initial) };
}


return Object.freeze({"validatePolygon": validatePolygon, "initZones": initZones});
})();

