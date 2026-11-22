__c500Modules["state.js"] = (() => {
'use strict';
// Presentation only. This module never reads or changes the watcher.
const finite = value => typeof value === 'number' && Number.isFinite(value);
const clean = value => typeof value === 'string' ? value.trim() : '';
const compact = (value, limit = 110) => {
  const text = clean(value).replace(/\s+/g, ' ');
  return text.length > limit ? text.slice(0, limit - 1).trimEnd() + '…' : text;
};
const ALIGNED = new Set(['tracking', 'aligned', 'ready', 'holding', 'disabled']);

function shortZoneName(value) {
  const name = clean(value);
  const key = name.toLowerCase().replace(/[\s_-]+/g, ' ').trim();
  return ({
    'exit corner immediate': 'Exit corner',
    'exit corner': 'Exit corner',
    'exit approach': 'Exit road',
    'top road arrival': 'Top approach',
    'top approach': 'Top approach',
    'entrance waiting': 'Entrance',
    'pedestrian footpath': 'Footpath',
    'pedestrian path': 'Footpath',
    'exit barrier': 'Exit barrier',
  })[key] || compact(name, 42);
}

function sourceLabel(source) {
  return ({ ai: 'AI detection', light: 'Light change', motion: 'Image change',
    fault: 'System', system: 'System' })[clean(source).toLowerCase()] || 'System';
}

function routeRole(value) {
  const item = value && typeof value === 'object' ? value : { name: value };
  const evidence = item.evidence || item.alarm_evidence || {};
  const role = clean(item.route_role || evidence.route_role).toLowerCase();
  if (['exit', 'entrance', 'pedestrian'].includes(role)) return role;
  const name = clean(item.name || item.zone_name || evidence.zone_name || item.reason).toLowerCase();
  if (/pedestrian|footpath|footway/.test(name)) return 'pedestrian';
  if (/\bexit\b/.test(name)) return 'exit';
  if (/entrance|top[ -]?(road|approach)/.test(name)) return 'entrance';
  return '';
}

function aiReading(ml = {}) {
  ml=ml&&typeof ml==='object'?ml:{};
  if (ml.state === 'loading') return {title:'AI loading',value:'Loading',detail:'The local object detector is starting.'};
  if (ml.available === true && ['ready','waiting_for_frames'].includes(ml.state)) {
    if (finite(ml.age_s) && ml.age_s >= 3) return {title:'AI result delayed',value:'Delayed',detail:`Last result ${ml.age_s.toFixed(1)} s ago. Waiting for fresh observations.`};
    if (ml.state === 'waiting_for_frames') return {title:'AI waiting',value:'Waiting',detail:'Waiting for fresh object observations.'};
  }
  return {title:'AI unavailable',value:'Unavailable',detail:'Object observations are not current.'};
}

function deriveStatus(value, connectionOk = true) {
  const s = value && typeof value === 'object' ? value : {};
  const connected = connectionOk === true;
  const armed = s.armed === true;
  const cameraFresh = connected && finite(s.frames) && s.frames > 0 &&
    finite(s.last_frame_age) && s.last_frame_age >= 0 && s.last_frame_age <= 3;
  const aligned = cameraFresh && ALIGNED.has(s.alignment?.state);
  const loopFresh = connected && finite(s.loop_age_s) && s.loop_age_s >= 0 && s.loop_age_s < 5;
  const ml = s.ml || {};
  const aiFresh = cameraFresh && aligned && ml.available === true && ml.state === 'ready' &&
    finite(ml.age_s) && ml.age_s >= 0 && ml.age_s < 3;
  const inputReady = connected && s.input_helper?.connected === true;
  const quietRemaining = connected && armed && finite(s.ack_remaining_s) ? Math.max(0, s.ack_remaining_s) : 0;
  const alarmActive = connected && armed && ['main', 'early', 'fault'].includes(s.alarm);
  const faultRecovered = alarmActive && s.alarm === 'fault' && typeof s.fault_reason === 'string' && !s.fault_reason.trim();
  const playbackFailed = connected && typeof s.audio?.playback?.error === 'string' && !!s.audio.playback.error.trim();
  const base = { cameraFresh, aligned, aiFresh, inputReady, quietRemaining, alarmActive, faultRecovered };
  const result = (tone, label, title, detail, overline, icon) =>
    ({ ...base, tone, label, title, detail: alarmActive && playbackFailed ? 'Alarm playback failed; retrying. '+detail : detail, overline, icon });

  if (s._preview?.off === true) {
    return { ...result('neutral', 'Offline', 'Full OFF', 'Full shutdown preview.', 'PROGRAM POWER', 'power'),
      cameraFresh: false, aligned: false, aiFresh: false, inputReady: false,
      quietRemaining: 0, alarmActive: false };
  }
  // A cached alarm cannot override loss of the live API connection.
  if (!connected) return result('warn', 'Reconnecting', 'Reconnecting to the watcher',
    'No answer from the watcher yet. After a restart it is back within about 20 s by itself.', 'CONNECTION', 'alert');

  if (alarmActive && ['main', 'early'].includes(s.alarm)) {
    const evidence = s.alarm_evidence || {};
    const rawZone = clean(evidence.zone_name) || clean(s.reason).split(':')[0];
    const zone = shortZoneName(rawZone);
    const source = sourceLabel(s.alarm_source || evidence.source);
    const role = routeRole({ evidence, reason: s.reason });
    const overline = [role === 'pedestrian' ? 'FOOTPATH' : role.toUpperCase(), source.toUpperCase()].filter(Boolean).join(' / ');
    const detail = evidence.manual_ack_required === true ? 'Pedestrian alert — acknowledge to stop.' :
      source === 'AI detection' ? 'Object detected in this route.' :
      source === 'Light change' ? 'Brightness changed in this route.' :
      source === 'Image change' ? 'A local image change needs attention.' : 'Activity needs attention.';
    return result(s.alarm === 'main' ? 'alarm' : 'warn', s.alarm === 'main' ? 'Alarm' : 'Early warning',
      zone || 'Activity detected', detail, overline || 'ACTIVITY', 'alert');
  }
  if (faultRecovered) return result('neutral', 'Recovered', 'System recovered',
    'The previous interruption resolved. Acknowledge to clear.', 'RESOLVED INTERRUPTION', 'check');
  if (alarmActive && s.alarm === 'fault') return result('warn', 'Fault', 'System fault',
    (compact(s.reason || s.fault_reason) || 'A system check needs attention.')+(Number.isFinite(s.blind_for_s)&&s.blind_for_s>=60?' · Not watching for '+Math.round(s.blind_for_s/60)+' min':''), 'SYSTEM', 'alert');
  if (!cameraFresh) return result('warn', 'Unavailable', 'Camera unavailable',
    'Waiting for a fresh camera image.', 'CAMERA', 'alert');
  if (!loopFresh) return result('warn', 'Delayed', finite(s.loop_age_s) ? 'System delayed' : 'System unavailable',
    finite(s.loop_age_s) ? 'Detection has stopped responding.' : 'Detector health is unknown.', 'DETECTOR', 'alert');
  if (!aligned) return result('warn', 'Needs attention', 'Alignment lost',
    'Scene detection is paused.', 'CAMERA ALIGNMENT', 'alert');
  if (!inputReady) return result('warn', 'Needs attention', 'Input unavailable',
    'Key and click acknowledgement are unavailable.', 'CONTROLS', 'alert');
  if (!aiFresh) {const ai=aiReading(ml);return result('warn', 'Needs attention', ai.title, ai.detail, 'DETECTION', 'alert');}
  if (playbackFailed) return result('warn', 'Check sound', 'Alarm playback failed',
    'Playback is retrying. Check the speaker and use Test alarm.', 'ALARM AUDIO', 'sound');
  if (s.audio?.available!==true || s.audio?.muted!==false || !finite(s.audio?.volume) || s.audio.volume<=0) return result('warn','Check sound',s.audio?.muted?'Speaker muted':'Speaker unavailable',
    'Check the speaker and use Test alarm.', 'ALARM AUDIO', 'sound');
  if (clean(s.fault_reason)) return result('warn', 'Needs attention', 'System fault',
    compact(s.fault_reason)+(Number.isFinite(s.blind_for_s)&&s.blind_for_s>=60?' · Not watching for '+Math.round(s.blind_for_s/60)+' min':''), 'SYSTEM', 'alert');
  if (s.armed === false) return result('neutral', 'Paused', 'Alerts paused',
    'The program is still running.', 'PROGRAM RUNNING', 'pause');
  if (!armed) return result('warn', 'Unknown', 'Alert status unknown',
    'Waiting for the watcher state.', 'CONTROLS', 'alert');
  const readinessWarmup=finite(s.semantic_warmup_remaining_s)?s.semantic_warmup_remaining_s:s.warmup_remaining_s;
  if (finite(readinessWarmup) && readinessWarmup > 0) return result('warn', 'Settling', 'Settling',
    `${Math.ceil(readinessWarmup)} s remaining.`, 'CAMERA VIEW', 'shield');
  if (quietRemaining > 0) return result('healthy', 'Acknowledged', 'Acknowledged',
    `${Math.ceil(quietRemaining)} s quiet.`, 'DETECTION CONTINUES', 'check');
  return result('healthy', 'Watching', 'Watching', 'Ready for the next arrival.', 'BARRIER GATE', 'shield');
}


return Object.freeze({"shortZoneName": shortZoneName, "sourceLabel": sourceLabel, "routeRole": routeRole, "deriveStatus": deriveStatus, "aiReading": aiReading});
})();

