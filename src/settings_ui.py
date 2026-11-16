"""The live Settings page (served at /settings). Plain HTML, CSS and JavaScript."""

PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>Settings</title><link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E%3Crect x='4' y='7' width='8' height='27' rx='2.5' fill='%23d8f78a'/%3E%3Crect x='2' y='32' width='12' height='5' rx='2' fill='%23d8f78a'/%3E%3Crect x='12' y='10' width='26' height='8' rx='2.5' fill='%23eef0e9'/%3E%3Crect x='16.5' y='10' width='4.5' height='8' fill='%23ff6b57'/%3E%3Crect x='25.5' y='10' width='4.5' height='8' fill='%23ff6b57'/%3E%3Crect x='34' y='10' width='4' height='8' rx='1.5' fill='%23ff6b57'/%3E%3Ccircle cx='8' cy='14' r='2.2' fill='%23101210'/%3E%3C/svg%3E">
<style>
:root{color-scheme:dark;--bg:#090c10;--surface:#11161d;--raised:#171e27;--line:#293340;--soft:#202933;--text:#eef4fb;--muted:#96a2b2;--faint:#748193;--accent:#70d9ed;--warn:#f2bf72;--danger:#ff947e;font:14px/1.5 "Segoe UI Variable Text","Segoe UI",Arial,sans-serif}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--text)}a{color:inherit}button,input,select{font:inherit;color:inherit}button{cursor:pointer}button:disabled{opacity:.5;cursor:default}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:3px}header{padding:24px 28px 20px;display:flex;align-items:center;gap:18px;flex-wrap:wrap;border-bottom:1px solid var(--line)}h1{margin:0;font-size:30px;font-weight:620;letter-spacing:-.04em}.page-subtitle{margin:3px 0 0;color:var(--muted);font-size:13px}.back{text-decoration:none;font-size:12px;color:var(--muted)}.status{margin-left:auto;display:flex;gap:9px;align-items:center;flex-wrap:wrap}.chip{font-size:12px;border:1px solid var(--line);border-radius:7px;padding:8px 12px;color:var(--muted)}.chip.on{color:var(--accent);background:#10232a;border-color:#284b57}.btn{border:1px solid var(--line);background:var(--raised);border-radius:7px;padding:9px 13px;font-size:12px}.btn:hover:not(:disabled){border-color:var(--muted)}.btn.accent{color:var(--accent);border-color:#305460;background:#10232a}.btn.warn{color:var(--warn)}.btn.danger{color:var(--danger);border-color:#785044}.settings-layout{max-width:1320px;margin:auto;display:grid;grid-template-columns:205px minmax(0,1fr);gap:28px;padding:24px 28px 40px}.settings-aside{position:sticky;top:20px;align-self:start}.search-label{display:block;color:var(--muted);font-size:11px;font-weight:600;letter-spacing:.08em;margin-bottom:8px;text-transform:uppercase}#setting-search{width:100%;min-width:0;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:10px 12px;font-size:13px}#category-nav{display:grid;gap:4px;margin-top:18px}#category-nav button{text-align:left;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:11px 12px;border:1px solid transparent;border-radius:8px;background:none;color:var(--muted);font-size:13px}#category-nav button[aria-current=true]{background:#13232c;border-color:#294653;color:var(--accent)}#category-nav small{color:var(--faint);font-variant-numeric:tabular-nums}.aside-note{font-size:12px;line-height:1.6;color:var(--faint);margin:24px 10px}main{min-width:0;display:grid;gap:18px;align-content:start}.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:23px 25px;min-width:0}.card h2{margin:0 0 5px;font-size:20px;font-weight:570;letter-spacing:-.02em}.card .sub{color:var(--muted);font-size:13px;margin:0 0 18px;max-width:70ch}.zone-title{margin:22px 0 6px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}.row{display:grid;grid-template-columns:minmax(0,1fr) minmax(170px,250px);gap:24px;align-items:center;padding:17px 0;border-top:1px solid var(--soft)}.label strong{display:block;font-size:14px;font-weight:550}.label small{display:block;color:var(--muted);font-size:12px;margin-top:4px;line-height:1.5}.control{display:flex;align-items:center;gap:12px;justify-content:flex-end;min-width:0}input[type=range]{width:100%;accent-color:var(--accent);min-width:0}.value{font-size:13px;font-variant-numeric:tabular-nums;color:var(--accent);min-width:56px;text-align:right}select,input.text{background:var(--raised);border:1px solid var(--line);border-radius:7px;padding:9px 10px;width:100%;min-width:0}.switch{width:44px;height:25px;border-radius:20px;border:1px solid var(--line);background:#293340;position:relative;flex:none}.switch::after{content:"";position:absolute;top:3px;left:3px;width:17px;height:17px;border-radius:50%;background:#a8b4c2;transition:transform .15s}.switch[aria-checked=true]{background:#193d48;border-color:#356678}.switch[aria-checked=true]::after{transform:translateX(19px);background:var(--accent)}.saving{opacity:.65}.row.saved .label strong::after{content:"  Saved";color:var(--accent);font-size:11px;font-weight:400}.learn-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.learn{background:var(--raised);border:1px solid var(--soft);border-radius:8px;padding:16px}.learn b{display:block;font-size:23px;font-weight:550;font-variant-numeric:tabular-nums}.learn span{font-size:12px;color:var(--muted)}.toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);background:#1c2d36;border:1px solid #355563;border-radius:8px;padding:12px 18px;font-size:13px;opacity:0;pointer-events:none;max-width:90vw;z-index:20}.toast.show{opacity:1}.toast.error{border-color:#785044;color:var(--danger)}.empty{color:var(--muted);font-size:14px}.connection-warning{margin:18px 28px 0;padding:12px 15px;border:1px solid #785044;background:#271c18;color:var(--danger);border-radius:8px;font-size:13px}#search-empty{padding:30px;border:1px dashed var(--line);border-radius:12px;color:var(--muted)}@media(max-width:800px){header{padding:20px}.settings-layout{grid-template-columns:1fr;padding:20px;gap:18px}.settings-aside{position:static}#category-nav{display:flex;overflow:auto;gap:6px;margin-top:12px}#category-nav button{flex:none}.aside-note{display:none}.row{grid-template-columns:minmax(0,1fr) minmax(150px,230px)}.status{margin-left:0}}@media(max-width:540px){.row{grid-template-columns:1fr;gap:12px}.control{justify-content:stretch}.card{padding:19px}.status{width:100%}.status .chip{margin-right:auto}h1{font-size:26px}.settings-layout{padding:16px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<header>
 <a class="back" href="/">← Dashboard</a>
 <div><h1>Settings</h1><p class="page-subtitle">Your preferences. Applied on this PC.</p></div>
 <div class="status">
  <span id="armed" class="chip">…</span>
  <button id="test" class="btn accent" type="button">Test sound</button>
  <button id="refresh" class="btn warn" type="button">Refresh system</button>
 </div>
</header>
<p id="connection-warning" class="connection-warning" role="alert" hidden></p>
<div class="settings-layout"><aside class="settings-aside"><label class="search-label" for="setting-search">Find a setting</label><input id="setting-search" type="search" placeholder="Sound, barriers, quiet…" autocomplete="off"><nav id="category-nav" aria-label="Settings categories"></nav><p class="aside-note">Changes save individually.<br>Detection continues while you adjust settings.</p></aside><main id="main"><div class="card"><p class="empty">Loading settings…</p></div></main></div>
<div id="toast" class="toast" role="status" aria-live="polite"></div>
<script>
const $=id=>document.getElementById(id);
let toastTimer=null,refreshArmed=0,sections=[],activeCategory="sound",loadVersion=0;
function toast(text,error=false){const t=$('toast');t.textContent=text;t.className='toast show'+(error?' error':'');clearTimeout(toastTimer);toastTimer=setTimeout(()=>t.className='toast',error?6000:2600);}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function shown(item,v){if(v===null||v===undefined)return '—';if(item.kind==='percent')return Math.round(v*100)+'%';
 if(item.kind==='seconds'){if((item.key==='power.stop_when_camera_gone_s'||item.key==='alarms.fault_remind_s')&&v===0)return 'off';return (v>=10?Math.round(v):Math.round(v*100)/100)+' s';}
 return String(Math.round(v*100)/100);}
function rowFor(item){
 const row=document.createElement('div');row.className='row';row.dataset.key=item.key;row.dataset.search=[item.label,item.help,item.zone,item.key].filter(Boolean).join(' ').toLowerCase();
 const label=`<div class="label"><strong>${esc(item.label)}</strong>${item.help?`<small>${esc(item.help)}</small>`:''}</div>`;
 let control='';
 if(item.kind==='bool')control=`<button class="switch" type="button" role="switch" aria-checked="${item.value===true}" aria-label="${esc(item.label)}"></button>`;
 else if(item.kind==='choice')control=`<select aria-label="${esc(item.label)}">${item.choices.map(([v,l])=>`<option value="${esc(v)}"${v===item.value?' selected':''}>${esc(l)}</option>`).join('')}</select>`;
 else if(item.kind==='text')control=`<input type="text" class="text" maxlength="${item.max_length||24}" value="${esc(item.value??item.default??'')}" aria-label="${esc(item.label)}" spellcheck="false" autocomplete="off">`;
 else{const scale=item.kind==='percent'?100:1;const v=(item.value??item.default??item.min/scale)*scale;
  control=`<input type="range" min="${item.min}" max="${item.max}" step="${item.step}" value="${v}" aria-label="${esc(item.label)}"><span class="value">${shown(item,item.value)}</span>`;}
 row.innerHTML=label+`<div class="control">${control}</div>`;
 const input=row.querySelector('input,select,.switch');
 if(item.kind==='bool')input.addEventListener('click',()=>save(item,input.getAttribute('aria-checked')!=='true',row));
 else if(item.kind==='choice')input.addEventListener('change',()=>save(item,input.value,row));
 else if(item.kind==='text')input.addEventListener('change',()=>{const v=input.value.trim().replace(/\s+/g,' ');if(!v){input.value=item.value??item.default??'';toast('This setting cannot be empty',true);}else if(v!==item.value)save(item,v,row);});
 else{const scale=item.kind==='percent'?100:1;
  input.addEventListener('input',()=>{row.querySelector('.value').textContent=shown(item,Number(input.value)/scale);});
  input.addEventListener('change',()=>save(item,Number(input.value)/scale,row));}
 return row;}
function render(data){
 if(data.name)document.title=data.name+' / Settings';
 sections=data.sections||[];const main=$('main');main.innerHTML='';
 for(const section of sections){
  const card=document.createElement('section');card.className='card';card.dataset.section=section.id;card.dataset.title=section.title;
  card.innerHTML=`<h2>${esc(section.title)}</h2><p class="sub">${esc({sound:'Changes apply to the next alarm.',barriers:'Arms must be set up in the Barriers drawer first.',system:'Applied straight away.',recordings:'Kept in the program’s private folder on this PC; about 1 MB each. Videos of alarms you mark Real or Wrong are kept for good.',zones:'Only sensitivity; zone outlines are edited on the Zones page.'}[section.id]||'')}</p>`;
  let zone=null;
  for(const item of section.items){
   if(item.zone&&item.zone!==zone){zone=item.zone;const h=document.createElement('div');h.className='zone-title';h.textContent=zone;card.append(h);}
   card.append(rowFor(item));}
  main.append(card);}
 const learning=data.learning||{};
 const card=document.createElement('section');card.className='card';card.dataset.section='learning';card.dataset.title='Learning & feedback';
 const routes=learning.routes||[],suggestions=learning.suggestions||[],arms=learning.barrier_learning||[];
 let html='<h2>What the system has learned</h2>';
 if(!routes.length)html+='<p class="empty">Learning starts collecting with the next alerts.</p>';
 else{
  html+=`<p class="sub">${esc(learning.summary||'')} A kind is judged after ${learning.min_evidence||15} alerts.</p><div class="learn-grid">`;
  html+=routes.map(r=>{const pct=r.rate===null||r.rate===undefined?'learning':Math.round(r.rate*100)+'% real';
   return `<div class="learn"><b>${r.followed}/${r.alerts}</b><span>${esc(r.name)} · ${pct}${r.reliable?' · climbs faster':''}${r.marked_wrong?` · you marked ${r.marked_wrong} wrong`:''}${r.marked_real?` · ${r.marked_real} real`:''}</span></div>`;}).join('');
  html+='</div>';}
 if(suggestions.length){html+='<div class="zone-title">Suggestions</div>'+suggestions.map((s,i)=>`<div class="row"><div class="label"><strong>${esc(s.text)}</strong></div><div class="control">${s.key?`<button class="btn accent" type="button" data-suggest="${i}">Apply</button>`:'<span class="sub">Advice only</span>'}</div></div>`).join('');}
 if(arms.length){html+='<div class="zone-title">Barrier examples learned by itself</div><p class="sub">'+arms.map(a=>`${esc(a.at)} ${esc(a.arm)} ${esc(a.position)} (match was ${a.match})`).join(' · ')+'</p>';}
 const skips=learning.barrier_learning_skips||[];if(skips.length){html+='<p class="sub">Not learned (glare or changing light): '+skips.map(s=>`${esc(s.at)} ${esc(s.arm)} ${esc(s.position)}`).join(' · ')+'</p>';}
 if(learning.barrier_undo_available){html+='<div class="row"><div class="label"><strong>Undo the latest learned barrier example</strong><small>Use this if an arm started reading wrongly after it learned something.</small></div><div class="control"><button class="btn warn" type="button" data-undo-barrier>Undo</button></div></div>';}
 card.innerHTML=html;
 card.querySelectorAll('[data-undo-barrier]').forEach(button=>button.addEventListener('click',async()=>{button.disabled=true;
  try{const r=await post('/barrier-learning/undo',{});toast(r.reason||'Undone');}catch(e){toast(e.message,true);}finally{load();}}));
 card.querySelectorAll('[data-suggest]').forEach(button=>button.addEventListener('click',async()=>{const s=suggestions[Number(button.dataset.suggest)];button.disabled=true;
  try{await post('/settings',{changes:{[s.key]:s.value}});toast('Applied · live now');}catch(e){toast(e.message,true);}finally{load();}}));
 main.append(card);buildCategories();filterSettings();
 const live=data.live||{};$('armed').textContent=live.armed?'Watching':'Paused';$('armed').className='chip'+(live.armed?' on':'');}
function buildCategories(){const nav=$('category-nav');nav.replaceChildren();for(const card of $('main').querySelectorAll('[data-section]')){const button=document.createElement('button');button.type='button';button.dataset.category=card.dataset.section;button.textContent=card.dataset.title;const count=document.createElement('small');count.textContent=card.querySelectorAll('[data-key]').length||'';button.append(count);button.addEventListener('click',()=>{activeCategory=card.dataset.section;$('setting-search').value='';filterSettings();});nav.append(button);}if(!$('main').querySelector('[data-section="'+activeCategory+'"]'))activeCategory=sections[0]?.id||'learning';}
function filterSettings(){const query=$('setting-search').value.trim().toLowerCase();let count=0;for(const card of $('main').querySelectorAll('[data-section]')){const selected=card.dataset.section===activeCategory;const title=card.dataset.title.toLowerCase();let matches=0;for(const row of card.querySelectorAll('[data-key]')){const match=!query||title.includes(query)||row.dataset.search.includes(query);row.hidden=!match;if(match)matches++;}const visible=query?(matches>0||title.includes(query)):selected;card.hidden=!visible;if(visible)count++;for(const heading of card.querySelectorAll('.zone-title'))heading.hidden=!!query;}for(const button of $('category-nav').querySelectorAll('button'))button.setAttribute('aria-current',String(!query&&button.dataset.category===activeCategory));let empty=$('search-empty');if(!empty){empty=document.createElement('p');empty.id='search-empty';empty.textContent='No settings match this search.';$('main').append(empty);}empty.hidden=count>0;}
function updateControl(item,row){const input=row.querySelector('input,select,.switch');if(item.kind==='bool')input.setAttribute('aria-checked',String(item.value===true));else if(item.kind==='text'||item.kind==='choice')input.value=item.value??'';else{input.value=item.value*(item.kind==='percent'?100:1);row.querySelector('.value').textContent=shown(item,item.value);}}
async function load(){const version=++loadVersion;const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),8000);try{const r=await fetch('/settings.json',{cache:'no-store',signal:controller.signal});if(!r.ok)throw new Error('Settings unavailable');const data=await r.json();if(version!==loadVersion)return;render(data);$('connection-warning').hidden=true;}catch{if(version!==loadVersion)return;$('connection-warning').hidden=false;$('connection-warning').textContent='Settings are unavailable. Check the Live page for monitoring status.';toast('Cannot reach the watcher. Existing settings have not changed.',true);}finally{clearTimeout(timer);}}
async function post(path,body){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),10000);try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Via':'web'},body:JSON.stringify(body||{}),signal:controller.signal});let d={};try{d=await r.json();}catch(e){if(e.name==='AbortError')throw e;}if(!r.ok||d.ok!==true)throw new Error(d.error||'The watcher did not confirm this change.');return d;}catch(e){if(e.name==='AbortError')throw new Error('Response timed out. The change may have applied; check this setting before retrying.');throw e;}finally{clearTimeout(timer);}}
async function save(item,value,row){if(row.classList.contains('saving'))return;row.classList.add('saving');const input=row.querySelector('input,select,.switch');input.disabled=true;++loadVersion;
 try{await post('/settings',{changes:{[item.key]:value}});item.value=value;updateControl(item,row);row.classList.add('saved');setTimeout(()=>row.classList.remove('saved'),2600);toast('Saved');}
 catch(e){updateControl(item,row);toast(e.message,true);}
 finally{row.classList.remove('saving');input.disabled=false;}}
$('test').addEventListener('click',async()=>{try{await post('/test-tone');toast('Playing the alarm sound for 2 seconds');}catch(e){toast(e.message,true);}});
$('refresh').addEventListener('click',async()=>{
 if(Date.now()-refreshArmed>4000){refreshArmed=Date.now();$('refresh').textContent='Press again to refresh';$('refresh').className='btn danger';setTimeout(()=>{if(Date.now()-refreshArmed>=4000){$('refresh').textContent='Refresh system';$('refresh').className='btn warn';}},4100);return;}
 refreshArmed=0;$('refresh').disabled=true;$('refresh').textContent='Refreshing…';
 try{await post('/refresh');toast('Restarting the watcher. Back in about 30 seconds.');}catch(e){toast(e.message,true);}
 let tries=0;const wait=setInterval(async()=>{tries++;try{const r=await fetch('/settings.json',{cache:'no-store'});if(r.ok&&tries>3){clearInterval(wait);$('refresh').disabled=false;$('refresh').textContent='Refresh system';$('refresh').className='btn warn';render(await r.json());toast('Settings reconnected. Check Live for camera and AI readiness.');}}catch{}if(tries>90){clearInterval(wait);$('refresh').disabled=false;$('refresh').textContent='Refresh system';toast('Still restarting. Check the Live page.',true);}},2000);
});
$('setting-search').addEventListener('input',filterSettings);
load();
</script>
</body>
</html>
'''
