import { Simulation, SCENARIOS } from './simulation.mjs?v=__DEMO_VERSION__';
import { SceneRenderer } from './scene.mjs?v=__DEMO_VERSION__';

const $ = (id) => document.getElementById(id);
const simulation = new Simulation();
const renderer = new SceneRenderer($('scene'));
const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
if (motionPreference.matches) simulation.setPaused(true);
let zones = true;
let lastFrame = 0;
let lastUI = 0;
let frameRequest = 0;
let eventKey = '';
let trackKey = null;
let announcementKey = '';
let lastSound = -Infinity;
let soundEnabled = false;
let audioContext;
const voices = new Set();

const scenarioIcons = { tour:'◈', entrance:'↘', exit:'↗', following:'⇉', pedestrian:'↝', occlusion:'◌', lighting:'☼' };
const PIPELINE = [
  { title:'Capture', subtitle:'The latest frame', tag:'FRESHNESS FIRST', heading:'A current picture.', copy:'The installed application reads camera frames and checks their freshness. A stale or unavailable feed is a system condition to surface, not an empty road to assume. Capture and incident encoding use OpenCV and FFmpeg.', tags:['Python','OpenCV','FFmpeg','Frame freshness'] },
  { title:'Detect', subtitle:'Objects + image cues', tag:'COMPLEMENTARY EVIDENCE', heading:'More than one kind of signal.', copy:'YOLOX-Tiny provides object observations through ONNX Runtime or optional OpenVINO acceleration. Configured light and motion cues can add early warnings. An optional YOLOX-S verifier provides another entrance check; this page uses scripted objects instead of running those models.', tags:['YOLOX-Tiny · ONNX','ONNX Runtime · CPU','Optional OpenVINO','Optional YOLOX-S'] },
  { title:'Track', subtitle:'A visit over time', tag:'CONTINUITY MATTERS', heading:'Keep the visit in view.', copy:'Custom geometry and velocity association connect detections. Visit memory carries an acknowledgement across repeated observations without letting a separately tracked follower inherit it. Brief occlusion can be bridged, but real tracking can still lose or confuse objects.', tags:['Geometry + velocity','Visit memory','Separate follower identity','Occlusion handling'] },
  { title:'Routes', subtitle:'Where attention matters', tag:'PLACE + PURPOSE', heading:'Different routes. Different context.', copy:'Configured polygons distinguish the entrance, exit and pedestrian path. Separately calibrated barrier evidence describes the observed arm state. Alignment checks help detect a changed camera view; the browser illustration uses an entirely fictional layout.', tags:['Route polygons','Image alignment','Barrier evidence','Local calibration'] },
  { title:'Alert', subtitle:'The operator acts', tag:'HUMAN IN CONTROL', heading:'A reason to look up.', copy:'Lane-specific tones and escalation request attention. Acknowledgement remembers the visit, with the configured ten-second quiet period. Pedestrian alerts require acknowledgement. The local dashboard and optional Windows WebView2 app display the evidence; neither operates a physical barrier.', tags:['Lane tones','Acknowledgement','Ten-second quiet','Windows WebView2'] },
];

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}
function clock(seconds) {
  return `${Math.floor(seconds / 60).toString().padStart(2,'0')}:${Math.floor(seconds % 60).toString().padStart(2,'0')}`;
}
function controlLabel(id, key, markup) {
  // Preserve button children between frames: replacing a pressed child before
  // pointer-up can swallow a click even though the button itself survives.
  const button = $(id);
  if (button.dataset.displayState === key) return;
  button.dataset.displayState = key;
  button.innerHTML = markup;
}
function stopVoices() {
  for (const voice of voices) { try { voice.stop(); } catch { /* Already ended. */ } }
  voices.clear();
}
function resetSoundClock() { stopVoices(); lastSound = -Infinity; }
function updateSoundButton(message) {
  $('sound').setAttribute('aria-pressed', String(soundEnabled));
  $('sound').lastElementChild.textContent = message || (soundEnabled ? 'Sound on' : 'Sound off');
}
function tone(lane) {
  if (!soundEnabled || !audioContext || audioContext.state !== 'running' || document.hidden || simulation.paused) return;
  const frequencies = lane === 'entrance' ? [520,690] : lane === 'exit' ? [690,520] : [620,620,790];
  frequencies.forEach((frequency, index) => {
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    const at = audioContext.currentTime + index * .18;
    oscillator.type = 'sine'; oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(0, at);
    gain.gain.linearRampToValueAtTime(.055, at + .02);
    gain.gain.exponentialRampToValueAtTime(.001, at + .14);
    oscillator.connect(gain); gain.connect(audioContext.destination);
    oscillator.onended = () => { voices.delete(oscillator); oscillator.disconnect(); gain.disconnect(); };
    voices.add(oscillator); oscillator.start(at); oscillator.stop(at + .16);
  });
}

function setScenario(id) {
  const preservePause = simulation.paused;
  simulation.setScenario(id);
  simulation.setPaused(preservePause);
  resetSoundClock(); eventKey = ''; trackKey = null; announcementKey = '';
  updateUI(simulation.snapshot());
  renderer.render(simulation.snapshot(), { zones });
}
for (const scenario of SCENARIOS) {
  const button = element('button','scenario-button');
  button.dataset.scenario = scenario.id;
  button.setAttribute('aria-pressed', String(scenario.id === 'tour'));
  button.append(element('span','',scenarioIcons[scenario.id]), element('span','',scenario.id === 'entrance' ? 'Entrance' : scenario.id === 'exit' ? 'Exit' : scenario.id === 'following' ? 'Following cars' : scenario.id === 'pedestrian' ? 'Pedestrian' : scenario.id === 'occlusion' ? 'Occlusion' : scenario.id === 'lighting' ? 'Light change' : scenario.title));
  button.title = scenario.description;
  button.addEventListener('click', () => setScenario(scenario.id));
  $('scenario-controls').append(button);
}
function selectPipeline(index) {
  const step = PIPELINE[index];
  $('tech-number').textContent = String(index + 1).padStart(2,'0');
  $('tech-tag').textContent = step.tag; $('tech-title').textContent = step.heading;
  $('tech-copy').textContent = step.copy;
  $('tech-tags').replaceChildren(...step.tags.map(tag => element('span','',tag)));
  for (const button of $('pipeline').children) button.setAttribute('aria-pressed', String(Number(button.dataset.step) === index));
}
PIPELINE.forEach((step,index) => {
  const button = element('button','pipeline-button'); button.dataset.step = index;
  button.append(element('span','',String(index+1).padStart(2,'0')), element('strong','',step.title), element('small','',step.subtitle));
  button.addEventListener('click',()=>selectPipeline(index));
  $('pipeline').append(button);
});
selectPipeline(0);

function updateUI(state) {
  const scenario = SCENARIOS.find(({id})=>id === state.scenario);
  $('scenario-note').textContent = scenario.description + (state.scenario === 'tour' ? ' The demo operator handles the barriers automatically; choose another scenario to take control.' : ' Acknowledge the alert, then use Open & serve for a vehicle.');
  if (state.scenario === 'lighting') $('scenario-note').textContent = scenario.description + ' These are visual effects, not the output of a real false-positive classifier.';
  for (const button of $('scenario-controls').children) button.setAttribute('aria-pressed',String(button.dataset.scenario === state.scenario));
  $('sim-clock').textContent = clock(state.time);
  $('scene-badge').lastElementChild.textContent = state.paused ? 'Simulation paused' : scenario.title;
  const visible = state.objects.filter(object=>object.progress < 1);
  $('scene-objects').textContent = `${visible.length} TRACK${visible.length === 1?'':'S'}`;
  $('track-count').textContent = `${visible.length} TRACK${visible.length === 1?'':'S'}`;
  controlLabel('pause', String(state.paused), state.paused ? '▶ <span>Play</span>' : 'Ⅱ <span>Pause</span>');
  $('pause').setAttribute('aria-label',state.paused?'Play simulation':'Pause simulation');
  $('speed').textContent = `${state.speed}×`;
  $('speed').setAttribute('aria-label',`Playback speed: ${state.speed} times`);
  controlLabel('lighting', state.lighting, state.lighting === 'night' ? '☾ <span>Night</span>' : '☼ <span>Day</span>');
  $('lighting').setAttribute('aria-label',state.lighting === 'night'?'Switch to day view':'Switch to night view');
  $('lighting').title = $('lighting').getAttribute('aria-label');
  const quiet = state.quietRemaining > .01;
  $('status-card').dataset.state = state.alarm ? 'alert' : quiet ? 'quiet' : 'watching';
  $('status-pill').textContent = state.paused ? 'Paused' : state.alarm ? 'Attention' : quiet ? 'Acknowledged' : 'Watching';
  $('status-symbol').textContent = state.alarm ? '!' : quiet ? '✓' : '⌁';
  $('alarm-lane').textContent = state.alarm ? `${state.alarm.lane === 'pedestrian'?'FOOTPATH':state.alarm.lane.toUpperCase()} · ${state.alarm.trackId}` : 'ALL ROUTES';
  $('status-title').textContent = state.alarm ? state.alarm.label + '.' : quiet ? 'Visit acknowledged.' : state.paused ? 'Ready when you are.' : 'Watching the gate.';
  $('status-detail').textContent = state.alarm ? state.alarm.lane === 'pedestrian' ? 'Pedestrian attention stays active until you acknowledge.' : 'An early route contact. Acknowledge, or serve this lane.' : quiet ? 'This visit stays quiet. A separate new arrival keeps its own place in line.' : state.paused ? 'Press Play to start the synthetic scenario.' : 'Every new visit gets its own attention.';
  $('acknowledge').disabled = !state.alarm;
  $('quiet-label').textContent = quiet ? 'New-visit quiet period' : state.alarm ? 'Attention until acknowledged' : 'No active alert';
  $('quiet-value').textContent = quiet ? `${state.quietRemaining.toFixed(1)}s` : '';
  $('quiet-progress').style.width = `${Math.max(0,Math.min(100,state.quietRemaining*10))}%`;
  for (const lane of ['entrance','exit']) {
    $(''+lane+'-state').textContent = state.barriers[lane].up ? `Up · ${Math.floor(state.barriers[lane].openFor)}s` : 'Down';
    $('serve-'+lane).disabled = !state.objects.some(object=>object.lane === lane && object.progress>=.16 && object.progress<1 && !object.served);
  }
  $('arrival-count').textContent = state.stats.arrivals;
  $('alert-count').textContent = state.stats.alerts;
  $('ack-count').textContent = state.stats.acknowledged;
  const newEventKey = state.events.slice(0,3).map(event=>event.id).join('|');
  if (newEventKey !== eventKey) {
    eventKey = newEventKey;
    $('event-feed').replaceChildren(...state.events.slice(0,3).map(event=> {
      const row = element('li'); const content = element('div');
      content.append(element('strong','',event.title),element('small','',event.detail));
      row.append(element('time','',clock(event.time)),content); return row;
    }));
  }
  const newTrackKey = visible.map(object=>`${object.id}:${object.occluded}:${object.zone}:${object.acknowledged}:${object.waiting}`).join('|');
  if (newTrackKey !== trackKey || !$('track-list').children.length) {
    trackKey = newTrackKey;
    const tracks = visible.slice(0,3).map(object=> {
      const row = element('div','track-item'); row.dataset.lane = object.lane;
      const description = element('div'); description.append(element('strong','',`${object.id} · ${object.kind.toUpperCase()}`),element('small','',object.zone));
      row.append(element('span','track-icon',object.kind === 'person'?'♙':object.lane === 'entrance'?'↘':'↗'), description, element('span','',object.occluded?'HELD':object.acknowledged?'RESOLVED':object.waiting?'WAITING':'TRACKED'));
      return row;
    });
    if (!tracks.length) { const empty=element('div','empty-state');empty.append(element('span','','⌖'),element('p','','The next visit starts a new track.'));tracks.push(empty); }
    $('track-list').replaceChildren(...tracks);
  }
  const announce = state.alarm ? state.alarm.id : quiet ? 'quiet' : 'watching';
  if (announce !== announcementKey) {
    announcementKey = announce;
    $('announcer').textContent = state.alarm ? `${state.alarm.label}. Track ${state.alarm.trackId}. Acknowledge available.` : quiet ? 'Visit acknowledged. Ten-second quiet period.' : 'Watching for the next arrival.';
    if (!state.alarm) resetSoundClock();
  }
  if (state.alarm && soundEnabled && state.time-lastSound>=3) { tone(state.alarm.lane); lastSound=state.time; }
}

$('acknowledge').addEventListener('click',()=>{simulation.acknowledge();resetSoundClock();updateUI(simulation.snapshot());});
for (const lane of ['entrance','exit']) $('serve-'+lane).addEventListener('click',()=>{simulation.serve(lane);resetSoundClock();updateUI(simulation.snapshot());});
$('restart').addEventListener('click',()=>setScenario(simulation.scenario));
$('pause').addEventListener('click',()=>{simulation.setPaused(!simulation.paused);resetSoundClock();updateUI(simulation.snapshot());});
$('speed').addEventListener('click',()=>{simulation.setSpeed(simulation.speed === 1?2:simulation.speed===2? .5:1);updateUI(simulation.snapshot());});
$('lighting').addEventListener('click',()=>{simulation.setLighting(simulation.lighting==='night'?'day':'night');updateUI(simulation.snapshot());});
$('zones').addEventListener('click',()=>{zones=!zones;$('zones').setAttribute('aria-pressed',String(zones));});
$('expand').addEventListener('click',async()=>{try {if (document.fullscreenElement) await document.exitFullscreen();else await $('scene-wrap').requestFullscreen();} catch {$('announcer').textContent='Fullscreen is not available in this browser.';}});
$('sound').addEventListener('click',async()=>{
  if (soundEnabled) {soundEnabled=false;stopVoices();updateSoundButton();return;}
  try {
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (!Audio) throw new Error('Audio unavailable');
    audioContext ||= new Audio(); await audioContext.resume();
    soundEnabled=audioContext.state==='running';lastSound=-Infinity;updateSoundButton();
    if (!soundEnabled) $('announcer').textContent='Audio is blocked by the browser.';
  } catch {soundEnabled=false;updateSoundButton('Sound unavailable');$('announcer').textContent='Audio is not available. Visual alerts still work.';}
});
document.addEventListener('keydown',(event)=>{
  if (event.key.toLowerCase() !== 'a' || event.ctrlKey || event.altKey || event.metaKey || event.shiftKey || event.repeat) return;
  if (event.target.closest('input,textarea,select,[contenteditable="true"]')) return;
  if (simulation.alarm) {event.preventDefault();$('acknowledge').click();}
});
motionPreference.addEventListener('change',(event)=>{if(event.matches){simulation.setPaused(true);resetSoundClock();updateUI(simulation.snapshot());}});
document.addEventListener('visibilitychange',()=>{
  if(document.hidden){simulation.setPaused(true);resetSoundClock();cancelAnimationFrame(frameRequest);lastFrame=0;updateUI(simulation.snapshot());}
  else {lastFrame=0;frameRequest=requestAnimationFrame(frame);}
});
window.addEventListener('pagehide',()=>{simulation.setPaused(true);soundEnabled=false;updateSoundButton();stopVoices();cancelAnimationFrame(frameRequest);if(audioContext?.state==='running')audioContext.suspend();});
window.addEventListener('pageshow',(event)=>{if(event.persisted){lastFrame=0;cancelAnimationFrame(frameRequest);frameRequest=requestAnimationFrame(frame);}});
function frame(now) {
  const state = simulation.step(lastFrame ? (now-lastFrame)/1000 : 0); lastFrame=now;
  renderer.render(state,{zones});
  if(now-lastUI>=100){updateUI(state);lastUI=now;}
  frameRequest=requestAnimationFrame(frame);
}
updateUI(simulation.snapshot());frameRequest=requestAnimationFrame(frame);
