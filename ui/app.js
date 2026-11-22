__c500Modules["app.js"] = (() => {
'use strict';
// BarrierWatcher interface (21 Nov 2026 revamp). Reads /state, /alerts and the zone layout; every command needs a
// receipt from the watcher before the interface changes. Live, Events and Vehicles are daily pages; Setup, Settings
// and Health hold the occasional tools.
const {initZones} = __c500Modules["zones.js"];
const {deriveStatus, routeRole, sourceLabel, shortZoneName, aiReading} = __c500Modules["state.js"];
const {createTransport, mergeLiveZones, mergeGeometryDraft} = __c500Modules["runtime.js"];

const production=document.body.dataset.runtime==='production';
const transport=createTransport({production,parseConfig:text=>{const doc=new DOMParser().parseFromString(text,'text/html');const node=doc.querySelector('script#initial-config[type="application/json"]');if(!node)throw new Error('Layout unavailable');return JSON.parse(node.textContent);}});
const $=id=>document.getElementById(id);
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const icon=name=>`<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const PAGES=['live','events','vehicles','setup','settings','health'],SETUP_TABS=['camera','zones','barriers'];
const LEGACY_PAGES={monitor:'live',activity:'events',cars:'vehicles',zones:'setup',system:'health'};
let liveState={},configuration=null,connectionOk=false;
let page='live',setupTab='camera',showZones=true,events=[],activityFilter='all',lastEventVersion=null,lastEventsAt=0;
let displayedFrame=null,lastFrameAt=0,frameRunning=false,active=true,renderedOverlay='',renderedEvents='',renderedActivity='',renderedRoutes='';
let historyPoints=[],lastHistoryFrame=null,toastTimer=null,stateController=null,frameController=null;
let commandBusy=false,shutdownRequested=false,configurationBusy=false,configurationTimer=null,lastStateAt=0,lastAckAt=0;
const roleMeta={entrance:{name:'Entrance',color:'var(--entry)',icon:'enter'},exit:{name:'Exit',color:'var(--exit)',icon:'exit'},pedestrian:{name:'Footpath',color:'var(--path)',icon:'walk'}};
const editor=initZones({imageUrl:transport.path('/api/scene.jpg'),getState:()=>({...mergedLive(),_uiConnected:connectionOk}),onToast:toast,production,onApply:applyZones});
const setupVisible=tab=>page==='setup'&&setupTab===tab;

// BARRIER_STATE_HELPERS_START
function savedCount(value){return value===true||(typeof value==='number'&&Number.isFinite(value)&&value>0);}
function barrierInspectionBox(raw){
 const lines=['up','down'].map(position=>raw?.lines?.[position]);
 if(lines.some(line=>!Array.isArray(line)||line.length!==2||line.some(point=>!Array.isArray(point)||point.length!==2||point.some(n=>typeof n!=='number'||!Number.isFinite(n)||n<0||n>1))))return null;
 const points=lines.flat(),xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
 const left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);
 const size=Math.min(1,Math.max(.24,right-left+.12,bottom-top+.12));
 return [Math.max(0,Math.min(1-size,(left+right-size)/2)),Math.max(0,Math.min(1-size,(top+bottom-size)/2)),size,size];
}
function barrierSnapshot(value, context={}){
 const items=Array.isArray(value?.barriers)?value.barriers:[];
 const usable=context.live===true&&context.connected===true&&context.cameraFresh===true&&context.aligned===true&&context.pictureFresh===true;
 return ['exit','entrance'].map(id=>{
  const matches=items.filter(item=>item&&item.id===id),raw=matches.length===1?matches[0]:{};
  const samples={up:savedCount(raw.samples?.up),down:savedCount(raw.samples?.down)};
  const calibrated=raw.calibrated===true&&samples.up&&samples.down;
  const candidate=typeof raw.state==='string'?raw.state.toLowerCase():'unknown';
  const known=usable&&raw.fresh===true&&calibrated&&['up','down'].includes(candidate);
  const state=known?candidate:'unknown',live=usable&&calibrated&&Array.isArray(raw.live_line)?raw.live_line:null,line=live||raw.active_line;
  const validLine=(known||!!live)&&Array.isArray(line)&&line.length===2&&line.every(point=>Array.isArray(point)&&point.length===2&&point.every(n=>typeof n==='number'&&Number.isFinite(n)&&n>=0&&n<=1))&&Math.hypot(line[1][0]-line[0][0],line[1][1]-line[0][1])>.001;
  const confidence=known&&typeof raw.confidence==='number'&&Number.isFinite(raw.confidence)&&raw.confidence>=0&&raw.confidence<=1?raw.confidence:null;
  const upFor=state==='up'&&typeof raw.up_for_s==='number'&&Number.isFinite(raw.up_for_s)&&raw.up_for_s>=0?raw.up_for_s:null;
  const reason=!calibrated?'Save UP and DOWN views':!usable?'Fresh aligned picture required':raw.fresh!==true?'Waiting for a fresh arm reading':typeof raw.reason==='string'&&raw.reason.trim()?raw.reason.trim().slice(0,240):known?'Observed from saved views':'Arm position is uncertain';
  const diagnosticFresh=usable&&calibrated&&raw.fresh===true;
  const candidatePosition=diagnosticFresh&&['up','down'].includes(raw.candidate_position)?raw.candidate_position:null;
  const measured=value=>diagnosticFresh&&typeof value==='number'&&Number.isFinite(value)?value:null;
  const confirmation=measured(raw.confirmation_s),confirmationS=confirmation!==null&&confirmation>=0?confirmation:null;
  return {id,name:id==='exit'?'Exit':'Entrance',state,calibrated,samples,confidence,upFor,reason,candidatePosition,confirmationS,match:measured(raw.match),oppositeMatch:measured(raw.opposite_match),inspectionBox:usable?barrierInspectionBox(raw):null,line:validLine?line.map(point=>point.slice()):null};
 });
}
function barrierDiagnostic(arm){
 const parts=[];
 if(arm.state==='unknown'&&arm.candidatePosition)parts.push(`Checking ${arm.candidatePosition.toUpperCase()}`);
 if(arm.candidatePosition&&arm.confirmationS!==null)parts.push(`${arm.confirmationS.toFixed(1)} s observed`);
 if(arm.match!==null&&arm.oppositeMatch!==null)parts.push(`Match ${arm.match.toFixed(2)} / other ${arm.oppositeMatch.toFixed(2)}`);
 return parts.join(' · ');
}
function barrierRequest(action,id,position){
 if(action==='capture-road')return {action};
 if(!['exit','entrance'].includes(id))throw new Error('Choose an existing barrier.');
 if(action==='reset')return {action,id};
 if(action!=='capture'||!['up','down'].includes(position))throw new Error('Choose an UP or DOWN view.');
 return {action,id,position};
}
function barrierLinesMarkup(barriers){
 return barriers.filter(arm=>arm.line).map(arm=>{
  const [[x1,y1],[x2,y2]]=arm.line,color=arm.id==='exit'?'var(--exit)':'var(--entry)';
  return `<g class="observed-arm" style="--arm-color:${color}"><title>${arm.name} barrier ${arm.state.toUpperCase()}</title><path d="M${(x1*640).toFixed(2)},${(y1*360).toFixed(2)} L${(x2*640).toFixed(2)},${(y2*360).toFixed(2)}"/><circle cx="${(x1*640).toFixed(2)}" cy="${(y1*360).toFixed(2)}" r="3"/><circle cx="${(x2*640).toFixed(2)}" cy="${(y2*360).toFixed(2)}" r="3"/></g>`;
 }).join('');
}
// BARRIER_STATE_HELPERS_END

// BARRIER_SETUP_START
// The Setup › Barriers tab (was a drawer). Captures never move a physical barrier.
let barrierBusy=false,barrierSignature='';
function barrierContext(status){
 return {live:true,connected:connectionOk&&!shutdownRequested&&performance.now()-lastStateAt<1800,cameraFresh:status.cameraFresh,aligned:status.aligned,pictureFresh:Boolean(displayedFrame)&&performance.now()-lastFrameAt<4500};
}
function renderBarrierInspection(arm,picture){
 const canvas=$(`barrier-${arm.id}-inspection`),label=$(`barrier-${arm.id}-inspection-label`),source=$('feed'),box=arm.inspectionBox;
 const available=picture&&box&&source.complete&&source.naturalWidth>0&&source.naturalHeight>0;
 canvas.hidden=true;label.textContent='Close-up unavailable · fresh aligned picture required';
 if(!available)return;
 try{
  const [x,y,width,height]=box,context=canvas.getContext('2d');
  context.drawImage(source,x*source.naturalWidth,y*source.naturalHeight,width*source.naturalWidth,height*source.naturalHeight,0,0,canvas.width,canvas.height);
  canvas.hidden=false;label.textContent=`${arm.name} arm · close-up ${(1/width).toFixed(1)}×`;
 }catch(_){canvas.hidden=true;label.textContent='Close-up unavailable';}
}
function renderBarriers(s,status){
 const context=barrierContext(status),arms=barrierSnapshot(s.barriers,context),canCapture=production&&Object.values(context).every(Boolean)&&!barrierBusy;
 const signature=JSON.stringify(arms);
 if(signature!==barrierSignature){
  barrierSignature=signature;
  const lines=barrierLinesMarkup(arms);$('barrier-overlay').innerHTML=lines;$('barrier-capture-overlay').innerHTML=lines;
 }
 const visible=setupVisible('barriers');
 for(const arm of arms){
  renderBarrierInspection(arm,visible&&Object.values(context).every(Boolean));
  const state=$(`barrier-${arm.id}-state`);state.textContent=!arm.calibrated?'NOT SET UP':arm.state.toUpperCase();state.className=`arm-pill ${arm.state}`;
  $(`barrier-${arm.id}-reason`).textContent=[arm.reason,barrierDiagnostic(arm)].filter(Boolean).join(' · ');
  for(const position of ['up','down']){
   const sample=$(`barrier-${arm.id}-${position}-sample`);sample.textContent=arm.samples[position]?'Saved':'Not saved';sample.classList.toggle('saved',arm.samples[position]);
   $(`capture-${arm.id}-${position}`).disabled=!canCapture;
  }
  $(`reset-${arm.id}`).disabled=!production||!connectionOk||shutdownRequested||barrierBusy||!Object.values(arm.samples).some(Boolean);
 }
 $('capture-road').disabled=!canCapture||!arms.every(arm=>arm.calibrated);
 if(visible){
  const picture=context.live&&context.connected&&context.cameraFresh&&context.pictureFresh;
  $('barrier-capture-image').hidden=!picture;$('barrier-picture-empty').hidden=picture;
  $('barrier-picture-state').textContent=picture?(context.aligned?'LIVE VIEW':'CHECK ALIGNMENT'):'PICTURE UNAVAILABLE';
  $('barrier-picture-state').className=`badge ${picture&&context.aligned?'':'warn'}`;
  if(picture&&$('barrier-capture-image').getAttribute('src')!==displayedFrame)$('barrier-capture-image').src=displayedFrame;
 }
 $('barrier-capture-status').textContent=barrierBusy?'Saving the observed view…':!production?'Preview only':!canCapture?'A fresh aligned camera picture is required.':'Move the physical arm, then capture its current position.';
}
async function saveBarrierView(action,id,position){
 if(barrierBusy)return;
 let payload;try{payload=barrierRequest(action,id,position);}catch(error){$('barrier-feedback').textContent=error.message;return;}
 const status=deriveStatus(currentState(),connectionOk),context=barrierContext(status);
 if(!production||!connectionOk||shutdownRequested||(action==='capture'||action==='capture-road')&&!Object.values(context).every(Boolean)){
  $('barrier-feedback').textContent='A fresh aligned live picture is required before saving.';return;
 }
 barrierBusy=true;$('barrier-feedback').textContent='';render();
 try{
  await transport.command('barrier-calibration',payload);
  $('barrier-feedback').textContent=action==='capture-road'?'Empty road saved.':action==='reset'?`${id==='exit'?'Exit':'Entrance'} examples reset.`:`${id==='exit'?'Exit':'Entrance'} ${position.toUpperCase()} view saved.`;
 }catch(error){$('barrier-feedback').textContent=error?.message||'Capture was not confirmed. Check the live state before retrying.';}
 finally{barrierBusy=false;render();}
}
document.querySelectorAll('[data-barrier-action]').forEach(button=>button.addEventListener('click',()=>saveBarrierView(button.dataset.barrierAction,button.dataset.barrier,button.dataset.barrierPosition)));
// BARRIER_SETUP_END

function toast(text){clearTimeout(toastTimer);$('toast').textContent=text;$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,3300);}
function mergedLive(){return mergeLiveZones(liveState,configuration);}
function currentState(){const s=mergedLive();if(production&&finite(s.ack_remaining_s))s.ack_remaining_s=Math.max(0,s.ack_remaining_s-(performance.now()-lastAckAt)/1000);return s;}
async function runCommand(action,payload={}){
 if(commandBusy||!connectionOk||shutdownRequested||performance.now()-lastStateAt>=1800)return;
 commandBusy=true;render();
 try{
  const result=await transport.command(action,payload);
  if(action==='power/off'){shutdownRequested=true;toast('Shutdown requested. Open the __SYSTEM_NAME__ app to start it again.');}
  else if(action==='ack'){
   if(liveState.alarm_id===payload.alarm_id)Object.assign(liveState,{alarm:null,alarm_id:null,last_ack:null,ack_remaining_s:finite(result.ack_remaining_s)&&result.ack_remaining_s>=0?result.ack_remaining_s:0});
   lastAckAt=performance.now();toast('Acknowledged · this visit stays quiet');
  }else if(action==='arm'||action==='disarm'){liveState.armed=result.armed;toast(result.control_warning||(result.armed?'Alerts resumed':'Alerts paused · the program keeps running'));}
  else if(action==='test-tone')toast('Alarm test started');
  else if(action==='settings'&&'barriers.reminder' in (payload.changes||{}))toast(payload.changes['barriers.reminder']?'Barrier-up reminder on':'Barrier-up reminder off');
 }catch(error){toast(error.message);}
 finally{commandBusy=false;render();}
}
async function applyZones(draft,baseline){
 if(!production)throw new Error('Preview is read only.');
 const state=await transport.read('/api/state',AbortSignal.timeout(4500));
 if(state.armed!==false)throw new Error('Pause alerts before applying the layout.');
 const config=await transport.read('/api/config',AbortSignal.timeout(4500));
 const payload=mergeGeometryDraft(config,draft,baseline);
 await transport.command('zones',payload);
 configuration=await transport.read('/api/config',AbortSignal.timeout(4500));
 liveState=await transport.read('/api/state',AbortSignal.timeout(4500));lastStateAt=performance.now();connectionOk=true;
 render();return configuration;
}

// VEHICLE_STATUS_START — read-only, visible-page polling; the embedded report keeps its own controls.
let vehicleStatusTimer=null,vehicleStatusController=null,vehicleReportVersion='';
function vehicleStatusPresentation(s){
 if(!s||typeof s.available!=='boolean')return {tone:'warn',title:'History status unavailable',detail:'Saved history may still be readable; current collection is unverified.'};
 if(s.sightings_enabled===false)return {tone:'warn',title:'Vehicle collection is off',detail:'Showing saved history. New sightings are not being collected.'};
 if(s.sightings_error)return {tone:'warn',title:'Vehicle collection needs attention',detail:'Showing saved history. New sightings may be missing.'};
 if(!s.available)return {tone:'warn',title:'Vehicle history unavailable',detail:'No generated history report is available yet.'};
 if(s.stale||!s.updater_alive||!['ready','updating'].includes(s.updater_state))return {tone:'warn',title:'History updater is not current',detail:'Showing the last saved report. New visits may not appear.'};
 const detail=s.model_state==='classical'?'Image matching mode · possible matches need review.':'Saved observations · possible matches need review.';
 return {tone:'good',title:s.updater_state==='updating'?'Updating vehicle history':'Vehicle collection is current',detail};
}
function stopVehicleStatus(){clearTimeout(vehicleStatusTimer);vehicleStatusTimer=null;vehicleStatusController?.abort();vehicleStatusController=null;}
function vehiclePageVisible(){return active&&page==='vehicles'&&!document.hidden;}
async function readVehicleStatus(){
 if(!vehiclePageVisible()||vehicleStatusController)return;
 clearTimeout(vehicleStatusTimer);vehicleStatusTimer=null;
 const controller=new AbortController();vehicleStatusController=controller;
 const timeout=setTimeout(()=>controller.abort(),4000);
 try{
  const state=await transport.read('/api/car-memory/status',controller.signal);
  if(controller.signal.aborted||vehicleStatusController!==controller||!vehiclePageVisible())return;
  const status=vehicleStatusPresentation(state);
  $('vehicle-status').className=`notice ${status.tone}`;
  $('vehicle-status-title').textContent=status.title;$('vehicle-status-detail').textContent=status.detail;
  const version=typeof state.report_version==='string'?state.report_version:'';
  if(version&&version!==vehicleReportVersion){
   $('cars-frame').contentWindow?.postMessage({type:'barrierwatcher:report-refresh',reportVersion:version},location.origin);
   vehicleReportVersion=version;
  }
 }catch{
  if(vehicleStatusController===controller&&vehiclePageVisible()){
   const status=vehicleStatusPresentation(null);$('vehicle-status').className='notice warn';
   $('vehicle-status-title').textContent=status.title;$('vehicle-status-detail').textContent=status.detail;
  }
 }finally{
  clearTimeout(timeout);
  if(vehicleStatusController===controller){vehicleStatusController=null;if(vehiclePageVisible())vehicleStatusTimer=setTimeout(readVehicleStatus,5000);}
 }
}
// VEHICLE_STATUS_END

function openEmbedded(next){
 const frame=next==='vehicles'?$('cars-frame'):next==='settings'?$('settings-frame'):null;
 if(!frame)return;
 // Explicit navigation opens the latest vehicle report; Settings keeps its place.
 if(next==='vehicles'||!frame.getAttribute('src'))frame.setAttribute('src',frame.dataset.src+(next==='vehicles'?(frame.dataset.src.includes('?')?'&':'?')+'t='+Date.now():''));
}
function chooseSetupTab(tab){
 if(!SETUP_TABS.includes(tab))tab='camera';
 if(setupTab==='zones'&&tab!=='zones')editor.deactivate();
 setupTab=tab;
 document.querySelectorAll('[data-setup-tab]').forEach(b=>{const on=b.dataset.setupTab===tab;b.classList.toggle('selected',on);b.setAttribute('aria-selected',String(on));});
 for(const name of SETUP_TABS)$(`setup-${name}`).hidden=name!==tab;
 if(page==='setup'&&tab==='zones'){editor.activate();if(displayedFrame)editor.updateImage(displayedFrame);}
 if(!setupVisible('camera'))stopSetupRawPreview();else if(setupPreviewMode==='raw')readSetupRawPreview();
 const s=currentState(),status=deriveStatus(s,connectionOk);renderBarriers(s,status);renderFraming(s,status);renderSetupPreview(s,status);
 if(page==='setup')history.replaceState(null,'',`#setup/${tab}`);
}
function navigate(next,tab){
 next=LEGACY_PAGES[next]||next;
 if(!PAGES.includes(next))next='live';
 openEmbedded(next);
 if(page==='setup'&&next!=='setup'&&setupTab==='zones')editor.deactivate();
 if(next!=='vehicles')stopVehicleStatus();
 page=next;
 document.querySelectorAll('.page').forEach(el=>{el.hidden=el.id!==`page-${next}`;});
 document.querySelectorAll('[data-page]').forEach(el=>{const on=el.dataset.page===next;el.classList.toggle('active',on);if(on)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
 history.replaceState(null,'',`#${next}`);
 if(next==='setup')chooseSetupTab(tab||setupTab);else stopSetupRawPreview();
 if(next==='events')renderActivity();
 if(next==='health')renderSystem(currentState());
 if(next==='vehicles')readVehicleStatus();
 document.querySelector('.pages').scrollTo({top:0,behavior:'instant'});
}
function updateClock(){const d=new Date();$('clock').textContent=d.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit'});$('date').textContent=d.toLocaleDateString('en-GB',{weekday:'short',day:'2-digit',month:'short'}).toUpperCase();}

function cameraViewLabel(s,status){
 if(!connectionOk||shutdownRequested||performance.now()-lastStateAt>=1800||!status.cameraFresh)return 'Camera view unconfirmed';
 const name=s.camera_view||s.alignment?.view;
 return typeof name==='string'&&name.trim()?name.trim():'Camera view unconfirmed';
}
function acknowledgementHint(s,status){
 if(status.faultRecovered)return 'Clear the resolved interruption';
 if(status.alarmActive)return production?'Any key or click · after 1 second':'Any key or click · preview only';
 if(status.quietRemaining<=0)return 'No active alarm';
 const ack=s.last_ack,via=ack&&({'input-any-key':'a key','input-mouse':'a mouse click',web:'the app button',terminal:'the terminal',api:'a control request',ctl:'a control request',hotkey:'the hotkey'})[ack.via];
 const clock=typeof ack?.at==='string'&&/^\d{4}-\d{2}-\d{2}T([01]\d|2[0-3]):[0-5]\d:[0-5]\d$/.test(ack.at)?ack.at.slice(11):null;
 return via&&clock&&typeof ack.alarm_id==='string'&&ack.alarm_id?`Acknowledged by ${via} at ${clock}`:'Same visit stays acknowledged';
}
function audioStageLabel(plan){
 return ({soft:'First warning',notice:'Repeating warning',wake:'Escalated alarm',urgent:'Urgent alarm',limited:'Speaker control limited',fault:'System warning',barrier_open:'Open arm reminder',silent:'Sound paused'})[plan.base_profile||plan.profile]||'Alarm';
}
function soundReady(s){const audio=s.audio||{};const error=typeof audio.playback?.error==='string'?audio.playback.error.trim():'';return connectionOk&&!error&&audio.available===true&&audio.muted===false&&finite(audio.volume)&&audio.volume>0;}
function render(){
 const s=currentState(),status=deriveStatus(s,connectionOk);
 if(shutdownRequested){Object.assign(status,{tone:'neutral',label:connectionOk?'Stopping':'Off',title:connectionOk?'Shutting down':'Program off',detail:connectionOk?'Waiting for the program to stop.':'Open the __SYSTEM_NAME__ app to start it again.',overline:'PROGRAM POWER',icon:'power',alarmActive:false,quietRemaining:0});}
 const tone=status.tone==='healthy'?'':status.tone;
 document.body.classList.toggle('tone-alarm',status.alarmActive&&status.tone==='alarm');
 document.body.classList.toggle('tone-warn',status.tone==='warn');
 $('global-state').className=`state-pill ${tone}`;$('global-state').innerHTML=`<span class="status-dot"></span><span>${esc(status.label)}</span>`;
 // Status card: calm while watching, the alarm itself when there is one.
 $('attention-card').className=`hero ${tone}`;
 $('attention-kicker').textContent=status.faultRecovered?'RECOVERED':status.alarmActive?'ATTENTION':status.quietRemaining>0?'ACKNOWLEDGED':status.tone==='warn'?'CHECK':'STATUS';
 $('signal-icon').innerHTML=`<use href="#i-${esc(status.icon||'shield')}"/>`;
 $('attention-overline').textContent=status.overline||'BARRIER GATE';
 $('attention-title').textContent=status.title;$('attention-detail').textContent=status.detail;
 $('acknowledge').hidden=!status.alarmActive;$('acknowledge').disabled=!status.alarmActive||commandBusy;
 const paused=connectionOk&&s.armed===false&&!shutdownRequested;
 $('hero-resume').hidden=!paused||status.alarmActive;$('hero-resume').disabled=commandBusy||!connectionOk;
 $('ack-hint').textContent=acknowledgementHint(s,status);$('ack-hint').hidden=!status.alarmActive&&!(status.quietRemaining>0);
 const quiet=status.quietRemaining||0;$('quiet-track').hidden=quiet<=0;
 $('quiet-seconds').textContent=`${quiet.toFixed(1)}s`;$('quiet-fill').style.width=`${Math.min(100,quiet/(finite(s.ack_quiet_s)&&s.ack_quiet_s>0?s.ack_quiet_s:5)*100)}%`;
 const audioPlan=s.loudness||{};$('alarm-audio-meta').hidden=!status.alarmActive||status.faultRecovered;
 $('alarm-audio-meta').textContent=audioPlan.audible===false?'Sound paused · still watching the route':audioStageLabel(audioPlan)+(finite(audioPlan.level_pct)&&audioPlan.audible!==false?` · ${audioPlan.level_pct}% volume`:'');
 // One pause control, always in the top bar.
 const pause=$('pause-alerts');pause.disabled=production&&(commandBusy||!connectionOk||shutdownRequested);
 pause.classList.toggle('is-paused',paused);pause.innerHTML=icon(s.armed===false?'play':'pause')+`<span>${s.armed===false?'Resume alerts':'Pause alerts'}</span>`;
 for(const id of ['power-off','power-off-health','test-alarm'])$(id).disabled=production&&(commandBusy||!connectionOk||shutdownRequested);
 // Camera
 const frameReady=displayedFrame&&status.cameraFresh&&performance.now()-lastFrameAt<4500;
 $('camera-empty').hidden=Boolean(frameReady);
 $('empty-title').textContent=shutdownRequested?'Program off':!status.cameraFresh?'Camera unavailable':'Connecting camera';
 $('empty-subtitle').textContent=!connectionOk?'Waiting for the watcher':'Waiting for a fresh picture';
 $('camera-stage').classList.toggle('stale',!frameReady);
 $('camera-badge').textContent=!status.cameraFresh?'NO SIGNAL':!frameReady?'NO PICTURE':'LIVE';$('camera-badge').className=`badge ${frameReady?'':'warn'}`;
 $('camera-view-name').textContent=cameraViewLabel(s,status);
 const warning=!status.aligned&&status.cameraFresh?(s.alignment?.state==='keyboard_menu'?'The keyboard menu is open over the camera picture':'Zone alignment needs attention'):!status.aiFresh&&status.cameraFresh?aiReading(s.ml).title:null;
 $('camera-warning').hidden=!warning;if(warning)$('camera-warning').querySelector('span').textContent=warning;
 $('frame-health').textContent=finite(s.last_frame_age)&&connectionOk?`Picture ${s.last_frame_age.toFixed(1)} s old`:'No picture';
 $('alignment-mini').textContent=status.aligned?'Zones aligned':s.alignment?.state==='keyboard_menu'?'Keyboard menu open':'Zones not aligned';
 $('dot-camera').className=status.cameraFresh?'good':'warn';$('dot-ai').className=status.aiFresh?'good':'warn';$('dot-input').className=status.inputReady?'good':'warn';$('dot-sound').className=soundReady(s)?'good':'warn';
 renderOverlay(s,status);renderRoutes(s,status);renderBarriers(s,status);renderObjects(s,status);renderFraming(s,status);renderSetupPreview(s,status);
 if(page==='health')renderSystem(s);
 editor.render({...mergedLive(),_uiConnected:connectionOk});
}
function pathData(points){if(!Array.isArray(points)||points.length<3||points.some(p=>!Array.isArray(p)||p.length!==2||p.some(n=>!finite(n)||n<0||n>1)))return '';return points.map((p,i)=>`${i?'L':'M'}${(p[0]*640).toFixed(2)},${(p[1]*360).toFixed(2)}`).join(' ')+'Z';}

// CAMERA_SETUP_HELPERS_START
function cameraSetupPresentation(s,context,checks={}){
 const data=s?.camera_setup,valid=data?.version===1;
 const fresh=context.live===true&&context.connected===true&&context.cameraFresh===true&&context.pictureFresh===true;
 const view=valid?(typeof data.view==='string'?data.view:data.view?.name||s.camera_view||'View unconfirmed'):'View unconfirmed';
 const reference=valid&&typeof data.reference_id==='string'?data.reference_id:'';
 const identity=reference?JSON.stringify([view,reference]):'';
 const state=valid&&['waiting','adjust','stabilizing','ready'].includes(data.state)?data.state:'waiting';
 const score=fresh&&valid&&state!=='waiting'&&finite(data.score_pct)&&data.score_pct>=0&&data.score_pct<=100?Math.round(data.score_pct):null;
 const required=valid&&finite(data.required_stable_s)&&data.required_stable_s>0?data.required_stable_s:5;
 const stable=fresh&&valid&&finite(data.stable_s)?Math.max(0,Math.min(required,data.stable_s)):0;
 const reviewed=['view','exit','entrance','pedestrian'].every(key=>checks[key]===true);
 const ready=fresh&&context.aligned===true&&valid&&state==='ready'&&data.ready_to_save===true&&stable>=required&&Boolean(reference);
 const paused=s?.armed===false,pending=valid&&data.calibration?.pending===true;
 const blockers=!fresh?['Waiting for a fresh camera picture and live readings.']:!valid?['Setup measurements are not available yet.']:(Array.isArray(data.blockers)?data.blockers.filter(x=>typeof x==='string').slice(0,8):[]);
 const guidance=fresh&&valid&&data.guidance&&typeof data.guidance.headline==='string'?data.guidance:null;
 return {data:valid?data:null,fresh,view,identity,state,score,stable,required,reviewed,ready,paused,pending,blockers,guidance,
  canFinish:ready&&reviewed&&!pending,canCapture:ready&&reviewed&&paused&&!pending,
  heading:!fresh?'Camera check unavailable':!valid?'Waiting for setup readings':state==='ready'?'View is stable':state==='stabilizing'?'Hold this view steady':state==='adjust'?'Framing needs attention':'Checking this view',
  tone:ready?'good':state==='adjust'&&fresh?'warn':'waiting'};
}
// CAMERA_SETUP_HELPERS_END
// CAMERA_SETUP_FLOW_START
let setupChecks={view:false,exit:false,entrance:false,pedestrian:false},setupIdentity='',setupCompletedAt='',setupReviewPersistent=false,setupCaptureRequest=null;
let setupPreviewMode='aligned',setupRawFrame=null,setupRawAt=0,setupRawController=null,setupRawTimer=null,setupRawGeneration=0;
const SETUP_STORAGE='barrierwatcher.cameraSetup.review.v1';
function syncSetupReview(identity){
 if(!identity||identity===setupIdentity)return;
 setupIdentity=identity;setupChecks={view:false,exit:false,entrance:false,pedestrian:false};setupCompletedAt='';setupReviewPersistent=false;
 try{const saved=JSON.parse(localStorage.getItem(SETUP_STORAGE)||'null');if(saved?.identity===identity&&typeof saved.completedAt==='string'&&saved.completedAt){setupCompletedAt=saved.completedAt;setupReviewPersistent=true;setupChecks={view:true,exit:true,entrance:true,pedestrian:true};}}catch(_){/* Storage is optional; readiness always comes from current measurements. */}
 for(const key of Object.keys(setupChecks))$(`setup-check-${key}`).checked=setupChecks[key];
 $('setup-empty-confirm').checked=false;
}
function renderCameraSetup(s,status){
 const context=barrierContext(status);let view=cameraSetupPresentation(s,context,setupChecks);
 if(view.fresh&&view.identity){syncSetupReview(view.identity);view=cameraSetupPresentation(s,context,setupChecks);}
 const data=view.data,calibration=data?.calibration||{},busy=commandBusy||Boolean(setupCaptureRequest);
 $('setup-current-view').textContent=view.fresh?`View: ${view.view}`:'View unconfirmed';
 $('setup-state-title').textContent=view.heading;
 $('setup-state-mark').textContent=view.ready?'✓':view.tone==='warn'?'!':'…';
 $('setup-state-mark').className=`setup-state-mark ${view.tone}`;
 $('setup-state-card').className=`card setup-state ${view.tone}`;
 $('setup-score').textContent=view.score===null?'—':view.score+'%';
 $('setup-stability').max=view.required;$('setup-stability').value=view.stable;
 $('setup-stability').setAttribute('aria-valuetext',`${view.stable.toFixed(1)} of ${view.required} steady seconds`);
 $('setup-stability-label').textContent=`${view.stable.toFixed(1)} / ${view.required} steady seconds`;
 const coverage=data?.coverage;
 $('setup-coverage').textContent=view.fresh&&finite(coverage?.required_visible)&&finite(coverage?.total_required)?`${coverage.required_visible} / ${coverage.total_required} areas visible`:'Coverage unavailable';
 $('setup-landmarks').textContent=view.fresh&&finite(data?.alignment?.inliers)?`${data.alignment.inliers} landmarks matched`:'Landmarks unavailable';
 $('setup-reference').textContent=data?.reference?.saved===true?'Saved reference loaded':'No saved reference';
 $('setup-blockers').replaceChildren(...view.blockers.map(text=>{const item=document.createElement('li');item.textContent=text;return item;}));
 $('setup-blockers').hidden=!view.blockers.length;
 $('setup-ready-copy').textContent=view.ready?'All measured framing checks pass. Confirm what you see below.':view.guidance?.detail||'Wait for current measurements. Only adjust the camera when a direction is measured.';
 for(const key of Object.keys(setupChecks))$(`setup-check-${key}`).disabled=!view.fresh||!view.identity||busy;
 $('setup-finish').disabled=!production||!view.canFinish||busy;
 $('setup-finish').textContent=setupCompletedAt&&view.ready?'Setup reviewed':'Finish setup';
 $('setup-capture').disabled=!production||!view.canCapture||busy||!$('setup-empty-confirm').checked;
 $('setup-empty-confirm').disabled=!production||!view.canCapture||busy;
 $('setup-pause').disabled=!production||!view.fresh||busy;
 $('setup-pause').hidden=view.paused;
 $('setup-capture-note').textContent=!view.paused?'Pause alerts before replacing the reference.':!view.reviewed?'Confirm the view and all three paths first.':!view.ready?'Wait until framing is stable and every area is visible.':'Capture replaces the camera reference. Barrier UP/DOWN examples stay separate.';
 $('setup-review-state').textContent=setupCompletedAt?(setupReviewPersistent?'Reviewed on this PC. Measurements are still checked live.':'Reviewed for this session.'):'Review once for this view. A new reference or layout needs a new review.';
 if(setupCaptureRequest&&view.fresh&&lastStateAt>setupCaptureRequest.submittedAt){
  if(view.view!==setupCaptureRequest.view){$('setup-action-status').textContent='The camera view changed. Check the active view before capturing again.';setupCaptureRequest=null;}
  else if(finite(calibration.revision)&&calibration.revision>setupCaptureRequest.revision&&!calibration.pending){$('setup-action-status').textContent='New reference saved. Check the outlines, then finish setup.';setupCaptureRequest=null;}
  else if(calibration.error&&!calibration.pending){$('setup-action-status').textContent='Reference was not saved: '+String(calibration.error);setupCaptureRequest=null;}
  else if(!calibration.pending&&performance.now()-setupCaptureRequest.submittedAt>45000){$('setup-action-status').textContent='Save completion was not confirmed. Check the current reference before retrying.';setupCaptureRequest=null;}
  else $('setup-action-status').textContent=calibration.pending?'Saving the reference — waiting for confirmation.':'Reference requested — waiting for the completed save.';
 }
 if(!setupVisible('camera'))stopSetupRawPreview();else ensureSetupRawPreview();
}
function finishCameraSetup(){
 const s=currentState(),view=cameraSetupPresentation(s,barrierContext(deriveStatus(s,connectionOk)),setupChecks);
 if(!production||commandBusy||setupCaptureRequest||!view.canFinish||view.identity!==setupIdentity)return;
 setupCompletedAt=new Date().toISOString();
 try{localStorage.setItem(SETUP_STORAGE,JSON.stringify({identity:view.identity,completedAt:setupCompletedAt}));setupReviewPersistent=true;$('setup-action-status').textContent='Setup review saved. The existing reference is unchanged.';}
 catch(_){setupReviewPersistent=false;$('setup-action-status').textContent='Setup reviewed for this session. Browser storage is unavailable.';}
 render();
}
async function captureSetupReference(){
 const s=currentState(),view=cameraSetupPresentation(s,barrierContext(deriveStatus(s,connectionOk)),setupChecks);
 if(!production||commandBusy||setupCaptureRequest||!view.canCapture||view.identity!==setupIdentity||!$('setup-empty-confirm').checked)return;
 const request={view:view.view,revision:finite(view.data.calibration?.revision)?view.data.calibration.revision:0,submittedAt:performance.now()};
 setupCaptureRequest=request;commandBusy=true;$('setup-action-status').textContent='Requesting a new camera reference…';render();
 try{await transport.command('alignment/calibrate',{guided:true,reference_id:view.data.reference_id});if(setupCaptureRequest===request)$('setup-action-status').textContent='Reference requested — waiting for the completed save.';}
 catch(error){if(setupCaptureRequest===request){setupCaptureRequest=null;$('setup-action-status').textContent=error.message+' Check reference status before retrying.';}}
 finally{commandBusy=false;render();}
}
function stopSetupRawPreview(){
 setupRawGeneration++;clearTimeout(setupRawTimer);setupRawTimer=null;
 if(setupRawController)setupRawController.abort();setupRawController=null;
 if(setupRawFrame)URL.revokeObjectURL(setupRawFrame);setupRawFrame=null;setupRawAt=0;
}
function ensureSetupRawPreview(){
 if(active&&!document.hidden&&setupVisible('camera')&&setupPreviewMode==='raw'&&connectionOk&&!shutdownRequested&&!setupRawController&&!setupRawTimer)readSetupRawPreview();
}
async function readSetupRawPreview(){
 clearTimeout(setupRawTimer);setupRawTimer=null;
 if(!active||document.hidden||!setupVisible('camera')||setupPreviewMode!=='raw'||setupRawController||!connectionOk||shutdownRequested)return;
 const generation=setupRawGeneration,controller=new AbortController();setupRawController=controller;
 const timeout=setTimeout(()=>controller.abort(),3000);let candidate=null;
 try{
  const response=await fetch(transport.path(`/api/raw.jpg?t=${Date.now()}`),{cache:'no-store',signal:controller.signal});if(!response.ok)throw new Error('Raw picture unavailable');
  candidate=URL.createObjectURL(await response.blob());const decoded=new Image();decoded.src=candidate;
  await new Promise((resolve,reject)=>{const abort=()=>finish(new Error('Raw picture expired'));const finish=error=>{controller.signal.removeEventListener('abort',abort);error?reject(error):resolve();};if(controller.signal.aborted)return abort();controller.signal.addEventListener('abort',abort,{once:true});decoded.decode().then(()=>finish(),finish);});
  if(generation!==setupRawGeneration||!setupVisible('camera')||setupPreviewMode!=='raw'||document.hidden)return;
  const previous=setupRawFrame;setupRawFrame=candidate;candidate=null;setupRawAt=performance.now();if(previous)URL.revokeObjectURL(previous);
  const s=currentState();renderSetupPreview(s,deriveStatus(s,connectionOk));
 }catch(_){if(generation===setupRawGeneration){const s=currentState();renderSetupPreview(s,deriveStatus(s,connectionOk));}}
 finally{clearTimeout(timeout);if(candidate)URL.revokeObjectURL(candidate);if(setupRawController===controller){setupRawController=null;if(active&&setupVisible('camera')&&setupPreviewMode==='raw'&&!document.hidden)setupRawTimer=setTimeout(readSetupRawPreview,500);}}
}
function chooseSetupPreview(kind){
 if(!['aligned','raw'].includes(kind))return;
 stopSetupRawPreview();setupPreviewMode=kind;
 for(const value of ['aligned','raw']){const button=$(`setup-preview-${value}`);button.setAttribute('aria-pressed',String(value===kind));button.classList.toggle('selected',value===kind);}
 if(kind==='raw')readSetupRawPreview();const s=currentState();renderSetupPreview(s,deriveStatus(s,connectionOk));
}
// CAMERA_SETUP_FLOW_END

// SETUP_PREVIEW_START — consumes the existing decoded frame; never opens a camera reader.
function renderSetupPreview(s,status){
 const image=$('setup-feed'),overlay=$('setup-overlay'),empty=$('setup-preview-empty'),caption=$('setup-preview-caption');
 const context=barrierContext(status),visible=setupVisible('camera');
 const raw=typeof setupPreviewMode!=='undefined'&&setupPreviewMode==='raw';
 const picture=raw?(setupRawFrame&&performance.now()-setupRawAt<2000?setupRawFrame:null):displayedFrame;
 const fresh=visible&&context.live&&context.connected&&context.cameraFresh&&(raw?Boolean(picture):context.pictureFresh);
 image.hidden=!fresh;overlay.toggleAttribute('hidden',true);empty.hidden=fresh;
 if(!fresh){
  image.removeAttribute('src');overlay.innerHTML='';
  caption.textContent=visible?(raw?'Original camera input unavailable':'Current camera preview unavailable'):'Camera setup is closed';
  return;
 }
 if(image.getAttribute('src')!==picture)image.src=picture;
 // Raw input has different coordinates: never put canonical zone outlines on it.
 const outlines=raw||!context.aligned?'':$('route-overlay').innerHTML;
 if(overlay.innerHTML!==outlines)overlay.innerHTML=outlines;
 overlay.toggleAttribute('hidden',!outlines.trim());
 caption.textContent=raw?'Original camera input — no alignment or zone overlay':!context.aligned?'Current picture — alignment needs attention; outlines hidden':outlines.trim()?'Aligned view with the saved zone outlines':'Aligned view — outlines unavailable';
}
// SETUP_PREVIEW_END
var framingGoodSince=null,framingSignature='';
function renderFraming(s,status){
 if(typeof renderCameraSetup==='function')renderCameraSetup(s,status);
 const box=$('framing-coach');if(!box)return;
 const setup=s?.camera_setup,guide=setup?.version===1?setup.guidance:null;
 const measured=guide?{primary:guide.action==='wait'?'lost':guide.action,state:setup.state==='waiting'?'unknown':setup.state,headline:guide.headline,text:guide.detail,fit_pct:guide.fit_pct}:null;
 const f=measured||(s&&s.alignment&&s.alignment.framing),context=barrierContext(status);
 if(!f||!f.headline||!context.live||!context.connected||!context.cameraFresh||!context.pictureFresh||(f.primary==='good'&&!context.aligned)){box.hidden=true;framingSignature='';framingGoodSince=null;return;}
 const good=f.primary==='good',uncertain=f.state==='unknown'||f.primary==='lost',retained=f.retained===true||f.state==='holding'||f.primary==='held';
 if(good){framingGoodSince=framingGoodSince??performance.now();}else framingGoodSince=null;
 const arrows={zoom_in:'⊕',zoom_out:'⊖',turn_left:'◀',turn_right:'▶',tilt_up:'▲',tilt_down:'▼',good:'✓',lost:'⊖'};
 const pct=Math.max(0,Math.min(100,Number(f.fit_pct)||0));
 const signature=[f.state,f.primary,f.headline,f.text,pct].join('|');
 box.hidden=false;
 if(signature===framingSignature)return;
 framingSignature=signature;
 box.className='framing-coach'+(retained?' held':good?' good':uncertain?' lost':'');
 box.textContent='';
 const arrow=document.createElement('span');arrow.className='fc-arrow';arrow.textContent=retained?'—':uncertain?'?':arrows[f.primary]||'⊕';
 const body=document.createElement('div');
 const head=document.createElement('div');head.className='fc-head';head.textContent=uncertain?'View alignment uncertain':f.headline;
 const text=document.createElement('div');text.className='fc-text';text.textContent=uncertain?'Waiting for clear landmarks. Check the camera position if this persists.':(f.text||'');
 const meter=document.createElement('div');meter.className='fc-meter';const bar=document.createElement('i');bar.style.width=pct+'%';meter.append(bar);
 body.append(head,text);if(!uncertain&&!retained)body.append(meter);box.append(arrow,body);
}
var renderedObjects='';
function overlayX(v){return (Math.max(0,Math.min(1,Number(v)||0))*640).toFixed(1);}
function overlayY(v){return (Math.max(0,Math.min(1,Number(v)||0))*360).toFixed(1);}
function renderObjects(s,status){
 const svg=$('object-overlay');
 const show=status.cameraFresh&&status.aligned;
 const objects=show&&status.aiFresh&&Array.isArray(s.objects)?s.objects.slice(0,12):[];
 const changes=show&&Array.isArray(s.change_boxes)?s.change_boxes.slice(0,8):[];
 const signature=JSON.stringify([objects,changes]);if(signature===renderedObjects)return;renderedObjects=signature;
 const X=overlayX,Y=overlayY,parts=[];
 for(const c of changes){
  if(!Array.isArray(c.box)||c.box.length!==4)continue;
  const x1=X(c.box[0]),y1=Y(c.box[1]),w=Math.max(2,X(c.box[2])-x1),h=Math.max(2,Y(c.box[3])-y1);
  const word=c.kind==='light'?'LIGHTS':'CHANGE',ty=y1>14?y1-4:Number(y1)+h+10;
  parts.push(`<g class="chg${c.alarm?' alarm':''}"><rect x="${x1}" y="${y1}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="3"/><text x="${x1}" y="${Number(ty).toFixed(1)}">${word} · ${esc(shortZoneName(c.zone||''))}</text></g>`);
 }
 for(const o of objects){
  if(!Array.isArray(o.box)||o.box.length!==4)continue;
  const x1=X(o.box[0]),y1=Y(o.box[1]),w=Math.max(2,X(o.box[2])-x1),h=Math.max(2,Y(o.box[3])-y1);
  const kind=o.label==='person'?'person':'vehicle',state=['alert','zone','seen'].includes(o.state)?o.state:'seen';
  const trail=(Array.isArray(o.trail)?o.trail:[]).filter(p=>Array.isArray(p)&&p.length===2).map(p=>`${X(p[0])},${Y(p[1])}`).join(' ');
  const name=String(o.label||'object').toUpperCase()+' '+Math.round((Number(o.confidence)||0)*100)+'%';
  const where=o.zone?' · '+shortZoneName(o.zone):'';
  const ty=y1>14?y1-4:Number(y1)+h+10;
  parts.push(`<g class="obj ${state} ${kind}">${trail.includes(' ')?`<polyline points="${trail}"/>`:''}<rect x="${x1}" y="${y1}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="2"/><text x="${x1}" y="${Number(ty).toFixed(1)}">${esc(name+where)}</text></g>`);
 }
 svg.innerHTML=parts.join('');
}
var reloadingForUpdate=false;
function maybeReloadForUpdate(s){
 const mine=(document.querySelector('meta[name="bw-version"]')||{}).content||'';
 if(reloadingForUpdate||!mine||!s||typeof s.ui_version!=='string'||!s.ui_version||s.ui_version===mine)return;
 if(s.alarm||page==='setup'||page==='settings'||document.querySelector('dialog[open]'))return;
 reloadingForUpdate=true;location.reload();
}
function renderOverlay(s,status){
 const signature=JSON.stringify([s.zones,status.aligned,showZones]);
 $('route-overlay').style.opacity=showZones?'1':'0';
 if(signature===renderedOverlay)return;renderedOverlay=signature;
 const paths=(s.zones||[]).map(z=>{const role=routeRole(z),color=status.aligned?(roleMeta[role]?.color||'var(--muted)'):'var(--muted)';const d=pathData(z.polygon)+(z.exclude_polygons||[]).map(pathData).join(' ');const gate=pathData(z.confirmation_polygon);return `<path data-route="${esc(role)}" d="${esc(d)}" fill="${color}" fill-opacity=".04" fill-rule="evenodd" stroke="${color}" opacity="${status.aligned?'.6':'.3'}"><title>${esc(z.name)}</title></path>${gate?`<path class="confirmation-zone" data-route="${esc(role)}" d="${esc(gate)}" opacity="${status.aligned?'.9':'.3'}"><title>${esc(z.name)} · alert area</title></path>`:''}`;}).join('');
 $('route-overlay').innerHTML=paths;
}
function hearLane(lane){
 if(!production||!['entrance','exit'].includes(lane))return;
 return runCommand('test-tone',{lane});
}
function renderRoutes(s,status){
 const context=barrierContext(status),arms=barrierSnapshot(s.barriers,context);
 // Mount once: live counters must not replace focused controls.
 if(renderedRoutes!=='mounted'){
  renderedRoutes='mounted';
  $('route-cards').innerHTML=Object.entries(roleMeta).map(([role,meta])=>{
   const head=`<button class="lane-top" data-route="${role}" aria-label="Show ${meta.name} zones on the camera">${icon(meta.icon)}<span class="lane-name">${meta.name}</span><i id="lane-${role}-indicator" class="lane-ind"></i></button>`;
   const status=`<div class="lane-row"><span id="lane-${role}-status">Waiting</span><strong id="lane-${role}-count"></strong></div>`;
   const barrier=role==='pedestrian'?`<div class="lane-row"><span>Separate walking route</span><span class="arm-pill">NO BARRIER</span></div>`:`<div id="lane-${role}-barrier" class="lane-row"><button type="button" class="lane-barrier-link" title="Save this barrier's UP and DOWN positions">Barrier <em id="lane-${role}-setup" class="set-up">· set up</em></button><span class="lane-right"><button type="button" id="lane-${role}-bell" class="bell" data-bell="${role}" aria-pressed="true" title="Barrier-up reminder">${icon('bell')}</button><span id="lane-${role}-arm" class="arm-pill unknown">UNKNOWN</span></span></div>`;
   return `<article class="lane" style="--route-color:${meta.color}">${head}${status}${barrier}</article>`;
  }).join('');
  document.querySelectorAll('.lane-barrier-link').forEach(button=>button.addEventListener('click',()=>navigate('setup','barriers')));
  document.querySelectorAll('[data-bell]').forEach(button=>button.addEventListener('click',()=>toggleUpReminder()));
  document.querySelectorAll('.lane-top').forEach(button=>{
   button.addEventListener('mouseenter',()=>highlightRoute(button.dataset.route));button.addEventListener('mouseleave',()=>highlightRoute(null));
   button.addEventListener('focus',()=>highlightRoute(button.dataset.route));button.addEventListener('blur',()=>highlightRoute(null));
   button.addEventListener('click',()=>{navigate('setup','zones');editor.selectRoute(button.dataset.route);});
  });
 }
 for(const [role] of Object.entries(roleMeta)){
  const zones=(s.zones||[]).filter(z=>routeRole(z)===role),objects=status.aiFresh&&context.connected?zones.reduce((n,z)=>n+(z.ai_objects||0),0):0;
  const activeZone=context.connected&&status.cameraFresh&&status.aligned&&s.armed&&zones.some(z=>(status.aiFresh&&z.ai_active)||z.motion_active||z.light_active);
  const ready=context.connected&&status.cameraFresh&&status.aligned&&status.aiFresh&&s.armed;
  const name=!context.connected?'Unavailable':!status.cameraFresh?'No picture':!status.aligned?'Not aligned':!s.armed?'Paused':activeZone?'Activity now':!status.aiFresh?'AI waiting':'Watching';
  $(`lane-${role}-indicator`).className=`lane-ind ${activeZone?'active':ready?'healthy':''}`;
  $(`lane-${role}-status`).textContent=name;
  $(`lane-${role}-count`).textContent=objects?`${objects} in view`:lastSeenText(role,s.lanes);
  if(role==='pedestrian')continue;
  const arm=arms.find(item=>item.id===role),pill=$(`lane-${role}-arm`);
  pill.className=`arm-pill ${arm.state}`;pill.title=arm.reason;$(`lane-${role}-setup`).hidden=arm.calibrated&&arm.state!=='unknown';
  const reminder=s.barriers?.reminder||{},leftUp=s.barriers?.left_up?.[role];
  const upSeconds=arm.state==='up'?(Number.isFinite(leftUp?.up_s)?leftUp.up_s:arm.upFor):null;
  const alerting=arm.state==='up'&&reminder.sounding===true&&leftUp&&!leftUp.muted&&Number.isFinite(reminder.after_s)&&upSeconds>=reminder.after_s;
  pill.className=`arm-pill ${arm.state}${alerting?' alerting':''}`;
  pill.textContent=!arm.calibrated?'NOT SET UP':arm.state==='unknown'?'UNKNOWN':arm.state.toUpperCase()+(arm.state==='up'&&Number.isFinite(upSeconds)?` ${clockText(upSeconds)}`:'');
  const bell=$(`lane-${role}-bell`),enabled=reminder.enabled!==false;
  bell.className=`bell${enabled?'':' off'}`;bell.ariaPressed=String(enabled);
  bell.innerHTML=icon(enabled?'bell':'bell-off');bell.disabled=!production||!context.connected||commandBusy;
  bell.title=enabled?`Barrier-up reminder ON: short beep after ${Math.round((reminder.after_s||60)/60*10)/10} min up, then every ${Math.round(reminder.every_s||10)} s. Click to turn off.`:'Barrier-up reminder OFF. Click to turn on.';
 }
 const soundReadyNow=production&&context.connected&&!commandBusy;
 $('hear-entrance').disabled=!soundReadyNow;$('hear-exit').disabled=!soundReadyNow;
}
// Barrier up-timer and last-vehicle text: short, for the lane tiles.
function clockText(seconds){const s=Math.max(0,Math.floor(seconds));if(s>=3600)return `${Math.floor(s/3600)}h ${String(Math.floor(s%3600/60)).padStart(2,'0')}m`;return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;}
function agoText(seconds){const s=Math.max(0,Math.round(seconds));return s<60?`${s}s`:s<3600?`${Math.floor(s/60)}m`:`${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m`;}
function lastSeenText(role,lanes){const seen=lanes&&lanes[role],noun=role==='pedestrian'?'person':'car';return seen&&Number.isFinite(seen.last_seen_s)?`last ${noun} ${agoText(seen.last_seen_s)} ago`:`no ${noun} yet`;}
function toggleUpReminder(){const enabled=currentState().barriers?.reminder?.enabled!==false;return runCommand('settings',{changes:{'barriers.reminder':!enabled}});}
function highlightRoute(role){document.querySelectorAll('#route-overlay path').forEach(p=>p.classList.toggle('highlight',!!role&&p.dataset.route===role));}

// EVENTS
function eventData(event){
 const evidence=event.evidence||{};const source=evidence.source||event.source||(event.kind==='fault'?'fault':'motion');
 const rawName=evidence.zone_name||(event.reason||event.detail||'').split(':')[0];
 const title=event.kind==='fault'?'System fault':shortZoneName(rawName)||'Campus event';
 const timestamp=event.ts||event.at||(event.date&&event.time?`${event.date}T${event.time}`:event.time)||'';
 const date=timestamp?new Date(timestamp):null;const valid=date&&!Number.isNaN(date.getTime());
 const time=valid?date.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit'}):'—';
 const day=valid?date.toLocaleDateString('en-GB',{day:'2-digit',month:'short'}):'';
 let snapshot=event.snapshot||'';
 if(!/^[A-Za-z0-9_-]+\.jpg$/.test(snapshot))snapshot='';
 return {title,source:sourceLabel(source),sourceKey:source,time,day,role:routeRole({...event,...evidence,name:rawName}),image:snapshot?transport.path(`/api/alerts/${encodeURIComponent(snapshot)}`):null,detail:event.reason||event.detail||'',acked:event.acked||null,id:typeof event.id==='string'&&event.id?event.id:null,kind:event.kind||'',video:typeof event.recording==='string'&&/^[0-9]{8}-[0-9]{6}-[A-Za-z0-9_-]{1,40}[.]mp4$/.test(event.recording)?transport.path('/api/recordings/'+encodeURIComponent(event.recording)):null,videoPending:event.recording_pending===true,crossLane:evidence.cross_lane&&typeof evidence.cross_lane.text==='string'?evidence.cross_lane.text:'',openArm:evidence.own_arm_up&&typeof evidence.own_arm_up.text==='string'?evidence.own_arm_up.text:'',cameFrom:typeof evidence.came_from==='string'&&evidence.came_from?`${String(evidence.label||'object').replace(/^./,c=>c.toUpperCase())} first seen ${evidence.came_from==='outside the zones'?'outside the zones':'in '+evidence.came_from}${typeof evidence.tracked_s==='number'?`, followed for ${Math.round(evidence.tracked_s)} s`:''}.`:'',feedback:event.feedback&&['wrong','real'].includes(event.feedback.verdict)?event.feedback.verdict:null};
}
function outcome(d){
 if(d.acked&&d.acked.via==='replaced')return {open:true,text:'Not answered · newer alarm'};
 if(!d.acked)return {open:true,text:'Not acknowledged'};
 return {open:false,text:d.acked.via==='barrier-raised'?'Arm raised':d.acked.via==='barrier-lowered'?'Arm lowered':d.acked.via==='auto-road-clear'?'Road cleared':'Acknowledged'};
}
function thumbnail(d){return d.image?`<img class="event-thumb" src="${d.image}" alt="" loading="lazy">`:`<span class="event-thumb">${icon('camera')}</span>`;}
function renderEvents(){
 const key=JSON.stringify(events);if(key!==renderedEvents){renderedEvents=key;
  $('recent-events').innerHTML=events.length?events.slice(0,12).map((event,index)=>{const d=eventData(event);return `<button class="event-card" data-event="${index}">${thumbnail(d)}<span class="event-info"><strong>${esc(d.title)}</strong><span class="event-meta"><span><i class="source-dot ${esc(d.sourceKey)}"></i>${esc(d.source)}</span><time>${d.time}</time></span></span></button>`;}).join(''):'<div class="empty-row">No events yet tonight</div>';
  document.querySelectorAll('#recent-events [data-event]').forEach(b=>b.addEventListener('click',()=>openEvent(Number(b.dataset.event))));
 }
 if(page==='events')renderActivity();
}
function matchesFilter(d){return activityFilter==='all'||(activityFilter==='system'?d.kind==='fault'||!d.role:d.role===activityFilter);}
function renderActivity(){
 const key=JSON.stringify([events,activityFilter]);if(key===renderedActivity)return;renderedActivity=key;
 const filtered=events.map((event,index)=>({event,index,data:eventData(event)})).filter(item=>matchesFilter(item.data));
 $('event-count').textContent=`${filtered.length} event${filtered.length===1?'':'s'} · newest first`;
 $('activity-list').innerHTML=filtered.length?filtered.map(({index,data:d})=>{
  const o=outcome(d),meta=roleMeta[d.role];
  const tags=[d.video?'<em class="tag video">Video</em>':'',d.crossLane?'<em class="tag note">Other lane?</em>':'',d.openArm?'<em class="tag note">Barrier open</em>':'',d.feedback?`<em class="tag ${d.feedback}">${d.feedback==='wrong'?'Marked wrong':'Marked real'}</em>`:''].join('');
  return `<button class="event-row" data-event="${index}">${thumbnail(d)}<span class="event-main"><strong>${esc(d.title)}${tags}</strong><span><i class="source-dot ${esc(d.sourceKey)}"></i>${esc(d.source)}</span></span><span class="route-tag" style="--c:${meta?.color||'var(--faint)'}"><i></i>${meta?meta.name:'System'}</span><span class="outcome ${o.open?'open':''}">${o.open?'':icon('check')}${esc(o.text)}</span><time class="event-time">${d.time}<small>${d.day}</small></time>${icon('chevron')}</button>`;
 }).join(''):'<div class="empty-row">No events for this route</div>';
 document.querySelectorAll('#activity-list [data-event]').forEach(b=>b.addEventListener('click',()=>openEvent(Number(b.dataset.event))));
}
let openEventId=null,feedbackBusy=false;
function renderFeedback(d){
 const box=$('event-feedback');const show=!!d.id&&d.kind!=='fault';box.hidden=!show;if(!show)return;
 box.querySelectorAll('[data-verdict]').forEach(b=>{const on=b.dataset.verdict===d.feedback;b.classList.toggle('selected',on);b.setAttribute('aria-pressed',String(on));b.disabled=feedbackBusy;});
 $('event-feedback-question').textContent=d.feedback?(d.feedback==='wrong'?'You marked this a wrong alarm.':'You marked this real.'):'Was this alarm right?';
 $('event-feedback-note').textContent=d.feedback?'Press it again to undo. The picture is kept so the system can learn from it.':'Your answer helps the local learning and is kept for review. It never silences alarms by itself.';
}
async function sendFeedback(verdict){
 const event=events.find(item=>item&&item.id===openEventId);if(!event||feedbackBusy)return;const d=eventData(event);if(!d.id)return;
 const next=d.feedback===verdict?'clear':verdict;feedbackBusy=true;renderFeedback(d);
 try{
  await transport.command('feedback',{alarm_id:d.id,verdict:next});
  event.feedback=next==='clear'?null:{verdict:next};renderedActivity='';renderedEvents='';renderEvents();
  toast(next==='clear'?'Mark removed':next==='wrong'?'Marked as a wrong alarm · thank you':'Marked as real · thank you');
 }catch(error){toast(error.message);}
 finally{feedbackBusy=false;const current=events.find(item=>item&&item.id===openEventId);if(current)renderFeedback(eventData(current));}
}
function openEvent(index){
 const event=events[index];if(!event)return;const d=eventData(event);
 $('event-dialog-time').textContent=[d.day,d.time].filter(Boolean).join(' · ');$('event-dialog-source').textContent=[roleMeta[d.role]?.name,d.source].filter(Boolean).join(' · ').toUpperCase();$('event-dialog-title').textContent=d.title;
 $('event-dialog-detail').textContent=d.detail;
 $('event-dialog-ack').textContent=d.acked&&d.acked.via==='replaced'?'Not answered: a newer alarm took over':d.acked?`${outcome(d).text}${finite(d.acked.after_s)?` after ${d.acked.after_s.toFixed(1)} s`:''}`:'No acknowledgement recorded';
 $('event-dialog-image').hidden=!d.image;if(d.image)$('event-dialog-image').src=d.image;
 const video=$('event-dialog-video');
 if(d.video){if(video.getAttribute('src')!==d.video)video.src=d.video;video.hidden=false;$('event-dialog-image').hidden=true;}
 else{video.pause();video.removeAttribute('src');video.hidden=true;}
 $('event-video-note').hidden=!!d.video||!d.videoPending;
 $('event-video-note').textContent='The video of this alarm is still being recorded. Open it again in about a minute.';
 $('event-dialog-cross').textContent=[d.crossLane,d.openArm,d.cameFrom].filter(Boolean).join(' ');$('event-dialog-cross').hidden=!(d.crossLane||d.openArm||d.cameFrom);
 openEventId=d.id;renderFeedback(d);
 $('event-dialog').showModal();
}

// HEALTH
function entranceCheckStatus(ml,current){
 const v=ml?.entrance_verifier;if(!v||!v.state)return null;
 if(!current)return {state:'unavailable',label:'Entrance check status unavailable',detail:'Awaiting current data',reason:'Waiting for fresh AI observations.'};
 if(v.state==='disabled')return {state:'disabled',label:'Entrance check off',detail:'',reason:'Optional entrance checker is disabled.'};
 if(v.state==='cooldown')return {state:'unavailable',label:'Entrance check retrying',detail:finite(v.retry_in_s)?`Retry in ${Math.ceil(Math.max(0,v.retry_in_s))} s`:'Primary detector continues',reason:v.error||'Primary detection continues while the optional check recovers.'};
 if(v.state==='deferred')return {state:'unavailable',label:'Entrance check deferred',detail:'Primary detector continues',reason:v.error||'Keeping the current primary observation fresh.'};
 if(v.state!=='ready')return {state:'unavailable',label:'Entrance check unavailable',detail:'Primary detector continues',reason:v.error||'The optional entrance checker is unavailable.'};
 const provider=String(v.provider||'').includes('OpenVINO')?'GPU':v.provider||'Local';
 const details=[provider,v.precision,finite(v.inference_ms)?`${Math.round(v.inference_ms)} ms`:null].filter(Boolean);
 return {state:'ready',label:'Entrance check ready',detail:details.join(' · '),reason:v.model||'Optional entrance check'};
}
function aiComputeLabel(ml){
 const provider=String(ml?.provider||''),extra=ml?.entrance_verifier||{};
 const gpu=provider.includes('OpenVINO'),cpu=provider==='CPU';
 if(cpu&&extra.state==='ready'&&String(extra.provider||'').includes('OpenVINO'))return 'CPU + GPU processing';
 return gpu?'GPU processing':cpu?'CPU processing':'AI processing';
}
function renderSystem(s){
 const status=deriveStatus(s,connectionOk);
 const entranceCheck=entranceCheckStatus(s.ml,status.aiFresh),extraUnavailable=entranceCheck?.state==='unavailable';
 $('system-state').className=`state-pill ${status.tone==='healthy'?'':status.tone}`;$('system-state').textContent=status.label;
 const audio=s.audio||s.loudness||{};
 const playbackError=typeof audio.playback?.error==='string'?audio.playback.error.trim():'';
 const ready=connectionOk&&!playbackError&&audio.available===true&&audio.muted===false&&finite(audio.volume)&&audio.volume>0;
 $('speaker-state').className=`chip ${ready?'good':'warn'}`;$('speaker-state').innerHTML=icon('sound')+`<span>${!connectionOk?'Speaker unavailable':playbackError?'Alarm playback failed · retrying':audio.available!==true?'Speaker unavailable':audio.muted?'Speaker muted':finite(audio.volume)?`Speaker ${Math.round(audio.volume*100)}%`:'Speaker unknown'}</span>`;
 $('speaker-state').title=connectionOk&&playbackError?playbackError:'';
 const backgroundReady=finite(s.background_warmup_remaining_s)&&s.background_warmup_remaining_s>=0;
 const backgroundNote=status.cameraFresh&&backgroundReady?{
  label:'Image background',state:s.background_warmup_remaining_s>0?'updating':'ready',
  detail:s.background_warmup_remaining_s>0?`Settling · ${Math.ceil(s.background_warmup_remaining_s)} s`:'Ready',
  reason:s.background_warmup_remaining_s>0?'Pixel change checks are settling; AI and barrier readiness are shown separately.':'Pixel change background is ready.'
 }:null;
 const cards=[
  {label:'Camera',icon:'camera',value:status.cameraFresh?'Live':'Unavailable',good:status.cameraFresh,left:'Picture age',right:finite(s.last_frame_age)?`${s.last_frame_age.toFixed(1)} s`:'—',note:backgroundNote},
  {label:'AI detection',icon:'system',value:status.aiFresh?(extraUnavailable?'Primary ready':'Ready'):!status.cameraFresh||!status.aligned?'Waiting':aiReading(s.ml).value,good:status.aiFresh&&!extraUnavailable,left:aiComputeLabel(s.ml),right:status.aiFresh&&finite(s.ml?.inference_ms)?`${Math.round(s.ml.inference_ms)} ms total`:'—',note:entranceCheck},
  {label:'Zone alignment',icon:'zones',value:status.aligned?'Locked':'Unverified',good:status.aligned,left:'Landmarks matched',right:finite(s.alignment?.matches)?String(s.alignment.matches):'—'},
  {label:'Keys & clicks',icon:'check',value:status.inputReady?'Ready':'Unavailable',good:status.inputReady,left:'Acknowledge from any app',right:status.inputReady?'On':'Off'},
 ];
 $('system-grid').innerHTML=cards.map(c=>`<section class="card tile"><div class="tile-head"><span>${c.label}</span>${icon(c.icon)}</div><div class="tile-value ${c.good?'good':'warn'}">${esc(c.value)}</div><div class="tile-foot"><span>${esc(c.left)}</span><span>${esc(c.right)}</span></div>${c.note?`<div class="tile-foot" title="${esc(c.note.reason)}"><span class="${c.note.state==='unavailable'?'warn-text':''}">${esc(c.note.label)}</span><span>${esc(c.note.detail)}</span></div>`:''}</section>`).join('');
 $('uptime').textContent=duration(s.uptime_s);
 $('source-name').textContent=typeof s.camera_view==='string'&&s.camera_view?`${s.camera_view} · capture stick`:'—';
 const points=historyPoints.slice(-60),max=Math.max(150,...points)*1.15;
 if(points.length<2){$('performance-chart').innerHTML='<text x="350" y="95" fill="var(--muted)" stroke="none" text-anchor="middle" font-size="12">Waiting for readings</text>';return;}
 const coords=points.map((value,i)=>[i/(points.length-1)*700,170-value/max*150]);
 const line=coords.map(([x,y],i)=>`${i?'L':'M'}${x.toFixed(2)} ${y.toFixed(2)}`).join(' ');
 const budget=(170-100/max*150).toFixed(1);
 $('performance-chart').innerHTML=`<defs><linearGradient id="chart-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="var(--accent)" stop-opacity=".18"/><stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs><path class="chart-grid" d="M0 ${budget}H700" stroke-dasharray="4 5"/><text x="696" y="${(budget-5)}" fill="var(--faint)" stroke="none" text-anchor="end" font-size="10">100 ms budget</text><path class="chart-area" d="${line} L700 180L0 180Z"/><path class="chart-line" d="${line}"/>`;
 $('chart-range').textContent=`Last ${points.length} readings`;$('chart-latest').textContent=`${Math.round(points.at(-1))} ms now`;
}
function duration(seconds){if(!finite(seconds)||seconds<0)return '—';const h=Math.floor(seconds/3600),m=Math.floor(seconds%3600/60);return h?`${h} h ${m} min`:`${m} min`;}

// PICTURE_CROP_START
// The keyboard's video output puts black margins around the camera picture. The Live view trims rows and columns
// that stayed black in the initial checks (at most a quarter per side). Once fitted, darker traffic/lighting
// must not keep zooming the display in. New content may expand the view, but only a camera change starts a new
// inward fit. Picture and outlines share one layer. Display only: detection and zones are untouched.
let cropHistory=[],cropView='',cropMeasuredAt=-Infinity,cropCanvas=null,cropBounds=null;
function cropBox(lum,w,h){
 const busy=values=>{let lit=0;for(const v of values)if(v>24)lit++;return lit>values.length*.04;};
 const row=y=>busy(lum.subarray(y*w,(y+1)*w)),col=x=>{const v=new Uint8Array(h);for(let y=0;y<h;y++)v[y]=lum[y*w+x];return busy(v);};
 let top=0,bottom=h-1,left=0,right=w-1;
 while(top<h*.25&&!row(top))top++;while(bottom>h*.75&&!row(bottom))bottom--;
 while(left<w*.25&&!col(left))left++;while(right>w*.75&&!col(right))right--;
 return [left/w,top/h,(right+1)/w,(bottom+1)/h];
}
function applyCrop(){
 const box=cropHistory.reduce((a,b)=>[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[2],b[2]),Math.max(a[3],b[3])],[1,1,0,0]);
 let [x0,y0,x1,y1]=cropHistory.length>=3?box:[0,0,1,1];
 if((x1-x0)*(y1-y0)>.97||x1<=x0||y1<=y0)[x0,y0,x1,y1]=[0,0,1,1];
 if(cropHistory.length>=3){
  if(cropBounds){x0=Math.min(x0,cropBounds[0]);y0=Math.min(y0,cropBounds[1]);x1=Math.max(x1,cropBounds[2]);y1=Math.max(y1,cropBounds[3]);}
  cropBounds=[x0,y0,x1,y1];
 }
 const root=document.documentElement.style,w=(x1-x0)*640,h=(y1-y0)*360;
 root.setProperty('--crop-x',(x0*640).toFixed(1));root.setProperty('--crop-y',(y0*360).toFixed(1));
 root.setProperty('--crop-w',w.toFixed(1));root.setProperty('--crop-h',h.toFixed(1));root.setProperty('--crop-aspect',(w/h).toFixed(4));
}
function measurePictureCrop(){
 const img=$('feed'),view=liveState.camera_view||'';
 const pictureKey=`${view}:${img.naturalWidth}x${img.naturalHeight}`;
 if(pictureKey!==cropView){cropView=pictureKey;cropHistory=[];cropBounds=null;cropMeasuredAt=-Infinity;applyCrop();}
 if(!img.naturalWidth||performance.now()-cropMeasuredAt<(cropHistory.length<3?1000:4000))return;
 cropMeasuredAt=performance.now();
 try{
  const w=160,h=90;cropCanvas=cropCanvas||document.createElement('canvas');cropCanvas.width=w;cropCanvas.height=h;
  const context=cropCanvas.getContext('2d',{willReadFrequently:true});context.drawImage(img,0,0,w,h);
  const data=context.getImageData(0,0,w,h).data,lum=new Uint8Array(w*h);
  for(let i=0;i<w*h;i++)lum[i]=(data[i*4]*299+data[i*4+1]*587+data[i*4+2]*114)/1000;
  cropHistory.push(cropBox(lum,w,h));cropHistory=cropHistory.slice(-6);applyCrop();
 }catch(_){/* Trimming is cosmetic; the full picture stays usable. */}
}
// PICTURE_CROP_END

// DATA
async function getJSON(path,signal){return transport.read(path,signal);}
async function readState(){
 if(!active)return;const controller=new AbortController();stateController=controller;const timeout=setTimeout(()=>controller.abort(),4000);
 try{
  const s=await getJSON('/api/state',controller.signal);if(!active||stateController!==controller)return;liveState=s;connectionOk=true;lastStateAt=performance.now();lastAckAt=lastStateAt;maybeReloadForUpdate(s);
  if(finite(s.local_pipeline_ms)&&s.frames!==lastHistoryFrame){historyPoints.push(s.local_pipeline_ms);historyPoints=historyPoints.slice(-60);lastHistoryFrame=s.frames;}
  if(s.alerts_version!==lastEventVersion||performance.now()-lastEventsAt>10000){lastEventVersion=s.alerts_version;lastEventsAt=performance.now();readEvents();}
  render();
 }catch{if(active&&stateController===controller){connectionOk=false;render();}}
 finally{clearTimeout(timeout);if(stateController===controller){stateController=null;if(active)setTimeout(readState,document.hidden||!connectionOk?600:250);}}
}
async function readEvents(){try{const data=await getJSON('/api/alerts',AbortSignal.timeout(4000));events=Array.isArray(data)?data:Array.isArray(data.alerts)?data.alerts:[];renderEvents();}catch{if(!events.length){$('recent-events').innerHTML='<div class="empty-row">Events unavailable</div>';$('activity-list').innerHTML='<div class="empty-row">Events unavailable</div>';}}}
async function readConfiguration(){if(configurationBusy||!active)return;configurationBusy=true;clearTimeout(configurationTimer);try{configuration=await getJSON('/api/config',AbortSignal.timeout(4500));renderedOverlay='';render();}catch{/* Keep current state authoritative during a restart. */}finally{configurationBusy=false;if(active)configurationTimer=setTimeout(readConfiguration,10000);}}
async function readFrame(){
 if(!active||frameRunning)return;frameRunning=true;const controller=new AbortController();frameController=controller;const timeout=setTimeout(()=>controller.abort(),4000);let candidate=null;
 try{
  const response=await fetch(transport.path(`/api/scene.jpg?t=${Date.now()}`),{cache:'no-store',signal:controller.signal});if(!response.ok)throw new Error('No frame');
  candidate=URL.createObjectURL(await response.blob());const decoded=new Image();decoded.src=candidate;
  // Aborting fetch alone cannot release an Image.decode() that is still pending.
  await new Promise((resolve,reject)=>{
   const aborted=()=>finish(new Error('Frame request expired'));
   const finish=error=>{controller.signal.removeEventListener('abort',aborted);error?reject(error):resolve();};
   if(controller.signal.aborted){aborted();return;}
   controller.signal.addEventListener('abort',aborted,{once:true});
   decoded.decode().then(()=>finish(),finish);
  });
  if(!active||frameController!==controller)return;const previous=displayedFrame;displayedFrame=candidate;candidate=null;lastFrameAt=performance.now();$('feed').src=displayedFrame;
  if(page==='setup'&&setupTab==='barriers')$('barrier-capture-image').src=displayedFrame;
  if(page==='setup'&&setupTab==='zones')editor.updateImage(displayedFrame);
  if(previous)URL.revokeObjectURL(previous);
  // Camera updates never schedule commands or open additional tabs.
  const s=currentState(),status=deriveStatus(s,connectionOk);
  if(status.cameraFresh){$('camera-empty').hidden=true;$('camera-stage').classList.remove('stale');}
  if(page==='setup'&&setupTab==='camera'){renderSetupPreview(s,status);renderFraming(s,status);}
 }catch{if(active&&frameController===controller&&performance.now()-lastFrameAt>4500)render();}
 finally{clearTimeout(timeout);if(candidate)URL.revokeObjectURL(candidate);if(frameController===controller){frameController=null;frameRunning=false;if(active)setTimeout(readFrame,page==='live'||page==='setup'?180:900);}}
}

// WIRING
document.querySelectorAll('[data-page]').forEach(button=>button.addEventListener('click',()=>navigate(button.dataset.page)));
document.querySelectorAll('[data-go]').forEach(button=>button.addEventListener('click',()=>navigate(button.dataset.go,button.dataset.setup)));
document.querySelectorAll('[data-setup-tab]').forEach(button=>button.addEventListener('click',()=>chooseSetupTab(button.dataset.setupTab)));
document.querySelector('.brand').addEventListener('click',e=>{e.preventDefault();navigate('live');});
$('feed').addEventListener('load',measurePictureCrop);
$('toggle-zones').addEventListener('click',()=>{showZones=!showZones;$('toggle-zones').classList.toggle('selected',showZones);$('toggle-zones').setAttribute('aria-pressed',String(showZones));$('route-overlay').style.opacity=showZones?'1':'0';});
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('camera-stage').requestFullscreen();}catch{toast('Full screen is unavailable in this window');}});
for(const key of ['view','exit','entrance','pedestrian'])$(`setup-check-${key}`).addEventListener('change',event=>{setupChecks[key]=event.target.checked===true;setupCompletedAt='';render();});
$('setup-empty-confirm').addEventListener('change',()=>render());
$('setup-finish').addEventListener('click',finishCameraSetup);
$('setup-capture').addEventListener('click',captureSetupReference);
$('setup-pause').addEventListener('click',()=>runCommand('disarm'));
$('setup-preview-aligned').addEventListener('click',()=>chooseSetupPreview('aligned'));
$('setup-preview-raw').addEventListener('click',()=>chooseSetupPreview('raw'));
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopSetupRawPreview();else if(setupVisible('camera')&&setupPreviewMode==='raw')readSetupRawPreview();});
window.addEventListener('pagehide',stopSetupRawPreview);
$('acknowledge').addEventListener('click',()=>runCommand('ack',{alarm_id:liveState.alarm_id}));
$('pause-alerts').addEventListener('click',()=>runCommand(liveState.armed?'disarm':'arm'));
$('hero-resume').addEventListener('click',()=>runCommand('arm'));
$('test-alarm').addEventListener('click',()=>runCommand('test-tone'));
document.querySelectorAll('[data-lane-sound]').forEach(button=>button.addEventListener('click',()=>hearLane(button.dataset.laneSound)));
for(const id of ['power-off','power-off-health'])$(id).addEventListener('click',()=>$('power-dialog').showModal());
$('cancel-power').addEventListener('click',()=>$('power-dialog').close());
$('confirm-power').addEventListener('click',()=>{$('power-dialog').close();runCommand('power/off');});
$('close-event').addEventListener('click',()=>$('event-dialog').close());
$('event-dialog').addEventListener('close',()=>{try{$('event-dialog-video').pause();}catch{}});
$('event-dialog').addEventListener('click',event=>{if(event.target===$('event-dialog'))$('event-dialog').close();});
if(/[?&]app=1\b/.test(location.search))window.addEventListener('beforeunload',event=>{if(connectionOk&&!shutdownRequested&&!reloadingForUpdate){event.preventDefault();event.returnValue='Closing this window switches the whole alarm system off.';}});
document.querySelectorAll('#event-feedback [data-verdict]').forEach(b=>b.addEventListener('click',()=>sendFeedback(b.dataset.verdict)));
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{activityFilter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('selected',b===button));renderActivity();}));
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopVehicleStatus();else if(page==='vehicles')readVehicleStatus();});
$('cars-frame').addEventListener('load',()=>{vehicleReportVersion='';if(vehiclePageVisible())readVehicleStatus();});
window.addEventListener('pagehide',()=>{active=false;stopVehicleStatus();clearTimeout(configurationTimer);stateController?.abort();stateController=null;frameController?.abort();frameController=null;frameRunning=false;if(displayedFrame)URL.revokeObjectURL(displayedFrame);});
window.addEventListener('pageshow',e=>{if(e.persisted){active=true;connectionOk=false;lastStateAt=0;lastAckAt=0;lastFrameAt=0;displayedFrame=null;render();readState();readFrame();readConfiguration();if(page==='vehicles')readVehicleStatus();}});
setInterval(()=>{if(!active)return;if(connectionOk&&performance.now()-lastStateAt>=1800){connectionOk=false;render();}updateClock();const s=currentState();renderBarriers(s,deriveStatus(s,connectionOk));if(production&&s.ack_remaining_s>0)render();},100);
if(!production){document.querySelector('.app-place').textContent='Preview · live data, controls disabled';}
const [startPage,startTab]=(location.hash.slice(1)||'live').split('/');
updateClock();navigate(startPage,startTab);render();readState();readConfiguration();readFrame();

return Object.freeze({});
})();
