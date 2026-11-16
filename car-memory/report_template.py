"""Local vehicle history presentation; no tracking, identity or occupancy decisions."""
import base64
import html
import math
import time


def _escape(value):
    return html.escape(str(value if value is not None else ''), quote=True)


def _clock(value, fmt='%d %b %Y · %H:%M:%S'):
    try:
        return time.strftime(fmt, time.localtime(float(value)))
    except (TypeError, ValueError, OverflowError, OSError):
        return 'Time unavailable'


def _description(row):
    kind = row.get('learned_type') or row.get('type') or 'Vehicle'
    colour = row.get('colour') if row.get('night') else row.get('learned_colour') or row.get('colour')
    return ' '.join(str(part) for part in (colour, kind) if part)


def _image(row, extra=''):
    thumb = row.get('thumb')
    if not isinstance(thumb, (bytes, bytearray)) or not thumb:
        return '<div class="photo-empty">Photo unavailable</div>'
    encoded = base64.b64encode(thumb).decode('ascii')
    return f'<img src="data:image/jpeg;base64,{encoded}" alt="{_escape(_description(row))}" loading="lazy" {extra}>'


def render_report(rows, pairs, generated_at, keep_hours, authorisation=''):
    """Render every saved sighting, including unclear direction, with review-only pairs."""
    rows = sorted(rows, key=lambda row: row.get('at') or 0, reverse=True)
    by_id = {row['id']: row for row in rows}
    pair_by_out = {pair.get('out'): pair for pair in pairs}
    counts = {direction: sum(row.get('direction') == direction for row in rows) for direction in ('in', 'out')}
    counts['unclear'] = len(rows) - counts['in'] - counts['out']
    suggestions = sum(pair.get('band') in ('strong', 'possible') and pair.get('in') in by_id for pair in pairs)
    directions = {'in': 'Entrance', 'out': 'Exit', 'unclear': 'Direction unclear'}
    cards = []
    for index, row in enumerate(rows):
        direction = row.get('direction') if row.get('direction') in directions else 'unclear'
        description = _description(row)
        observed = _clock(row.get('at'))
        light = 'Infrared · colour unsure' if row.get('colourless') else 'Night · colour unsure' if row.get('night') else 'Daylight observation'
        origin = 'Clean sighting' if str(row.get('clip', '')).startswith('live:') else 'Alarm recording'
        pair = pair_by_out.get(row['id'], {})
        entry = by_id.get(pair.get('in'))
        match = ''
        if pair.get('band') in ('strong', 'possible') and entry:
            band = 'Stronger appearance suggestion' if pair['band'] == 'strong' else 'Possible appearance suggestion'
            try:
                score = float(pair.get('score'))
                score_text = f'{score:.2f}' if math.isfinite(score) else 'unavailable'
            except (TypeError, ValueError):
                score_text = 'unavailable'
            light_note = ' Lighting differs between these observations.' if pair.get('light_differs') else ''
            match = f'''<details class="suggestion"><summary>{band}<span>Review entry photo</span></summary><div class="suggestion-body">{_image(entry)}<div><strong>{_escape(_description(entry))}</strong><time>{_escape(_clock(entry.get('at')))}</time><p>Appearance score {score_text}; not an identity probability.{light_note} Similar vehicles can be confused.</p></div></div></details>'''
        elif direction == 'out':
            match = '<p class="match-note">No entry suggestion in this report.</p>'
        search = f'{description} {directions[direction]} {observed} {light} {origin}'
        cards.append(f'''<article class="vehicle-card" data-sighting="{_escape(row['id'])}" data-direction="{direction}" data-search="{_escape(search.lower())}" data-index="{index}"><button class="photo-button" type="button" data-photo="{_escape(row['id'])}" aria-label="Enlarge {_escape(description)} photo">{_image(row)}</button><div class="card-copy"><div class="card-heading"><span class="direction {direction}">{directions[direction]}</span><time>{_escape(_clock(row.get('at'), '%H:%M:%S'))}</time></div><h2>{_escape(description)}</h2><p class="observed">Observed {_escape(observed)}</p><div class="card-meta"><span>{_escape(light)}</span><span>{origin}</span></div>{match}</div></article>''')
    try:
        retention = float(keep_hours)
        retention_text = f'{retention:g}' if math.isfinite(retention) and retention > 0 else 'configured'
    except (TypeError, ValueError):
        retention_text = 'configured'
    summary = ''.join(f'<div><strong>{count}</strong><span>{label}</span></div>' for count, label in (
        (len(rows), 'Saved observations'), (counts['in'], 'Entrance'), (counts['out'], 'Exit'),
        (counts['unclear'], 'Direction unclear'), (suggestions, 'Appearance suggestions')))
    page = TEMPLATE.replace('__STYLES__', STYLES).replace('__SCRIPT__', SCRIPT)
    replacements = {'__SUMMARY__': summary, '__CARDS__': ''.join(cards), '__GENERATED__': _escape(_clock(generated_at)),
                    '__RETENTION__': retention_text, '__AUTHORISED__': _escape(authorisation) or 'Recorded locally',
                    '__EMPTY_HIDDEN__': ' hidden' if rows else '', '__COUNT__': str(len(rows))}
    for token, value in replacements.items():
        page = page.replace(token, value)
    return page


STYLES = r'''
:root{color-scheme:dark;--bg:#090c10;--surface:#11161d;--raised:#171e27;--line:#293340;--text:#eef4fb;--muted:#96a2b2;--accent:#70d9ed;--entry:#f2bf72;--exit:#75d8b6;--unclear:#b9a1ee;font:14px/1.5 'Segoe UI',system-ui,sans-serif}*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--text)}button,input,select{font:inherit}button{cursor:pointer;color:inherit}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}button:disabled{opacity:.45;cursor:default}main{max-width:1900px;margin:auto;padding:26px 28px 36px}.heading{display:flex;align-items:flex-start;justify-content:space-between;gap:25px;margin-bottom:20px}.eyebrow{font-size:10px;letter-spacing:.12em;color:var(--accent);text-transform:uppercase}h1{font-size:30px;letter-spacing:-.025em;margin:4px 0 5px;font-weight:600}.intro{color:var(--muted);margin:0;font-size:13px;max-width:78ch}.generated{text-align:right;color:var(--muted);font-size:11px;white-space:nowrap;padding-top:6px}.generated strong{display:block;font-size:12px;font-weight:500;color:var(--text);margin-top:4px}.summary{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:0 0 20px}.summary>div{border:1px solid var(--line);background:var(--surface);border-radius:10px;padding:16px 18px}.summary strong{font-size:28px;font-weight:500;display:block;line-height:1.1}.summary span{font-size:12px;color:var(--muted);display:block;margin-top:9px}.summary>div:nth-child(2) strong{color:var(--entry)}.summary>div:nth-child(3) strong{color:var(--exit)}.summary>div:nth-child(4) strong{color:var(--unclear)}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:14px 0;margin-bottom:10px;position:sticky;top:0;z-index:3;background:#090c10f5;backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}.search{display:flex;align-items:center;gap:8px;flex:1;min-width:220px;max-width:480px;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:0 12px}.search svg{width:18px;height:18px;stroke:var(--muted);fill:none;stroke-width:1.8}.search input{width:100%;min-height:42px;background:none;border:0;color:var(--text);outline-offset:0}.search input::placeholder{color:var(--muted)}.filters{display:flex;gap:3px;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:4px}.filters button{padding:7px 12px;background:none;border:0;border-radius:5px;color:var(--muted);font-size:12px;white-space:nowrap}.filters button[aria-pressed=true]{background:#19343e;color:var(--accent)}.group-control{display:flex;gap:8px;align-items:center;color:var(--muted);font-size:12px;margin-left:auto}.group-control select{background:var(--surface);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:9px;font-size:12px}.results-heading{display:flex;gap:12px;align-items:center;justify-content:space-between;margin:18px 0 12px;color:var(--muted);font-size:12px}.results-heading h2{font-size:16px;font-weight:600;color:var(--text);margin:0}.vehicle-list{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;align-items:start}.vehicle-card{min-width:0;background:var(--surface);border:1px solid var(--line);border-radius:11px;overflow:hidden}.photo-button{display:block;border:0;padding:0;width:100%;background:#05080c;aspect-ratio:16/10;overflow:hidden}.photo-button img{width:100%;height:100%;object-fit:contain;display:block;transition:transform .18s}.photo-button:hover img{transform:scale(1.03)}.photo-empty{display:grid;place-items:center;min-height:100px;height:100%;font-size:12px;color:var(--muted)}.card-copy{padding:14px}.card-heading{display:flex;align-items:center;justify-content:space-between;gap:8px}.direction{font-size:10px;letter-spacing:.035em;text-transform:uppercase;display:inline-flex;border:1px solid #584b36;background:#2c2419;border-radius:5px;color:var(--entry);padding:3px 7px}.direction.out{color:var(--exit);background:#152c25;border-color:#2a5142}.direction.unclear{color:var(--unclear);background:#292137;border-color:#4a3c60}.card-heading>time{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}.card-copy h2{font-size:16px;font-weight:600;line-height:1.3;margin:12px 0 5px;text-transform:capitalize}.observed{font-size:11px;color:var(--muted);margin:0}.card-meta{display:flex;flex-wrap:wrap;gap:5px 12px;color:var(--muted);font-size:10px;margin-top:10px}.suggestion{margin-top:14px;border-top:1px solid var(--line);padding-top:11px}.suggestion summary{font-size:12px;color:var(--accent);cursor:pointer}.suggestion summary>span{display:block;font-size:10px;color:var(--muted);margin:3px 0 0 15px}.suggestion-body{display:flex;gap:10px;align-items:flex-start;margin-top:12px}.suggestion-body>img{width:95px;max-width:40%;height:80px;object-fit:contain;border-radius:5px;background:#080b10}.suggestion-body strong{font-size:12px;font-weight:500}.suggestion-body time{display:block;font-size:10px;color:var(--muted);margin-top:4px}.suggestion-body p{font-size:11px;color:var(--muted);margin:8px 0 0}.match-note{font-size:11px;color:var(--muted);border-top:1px solid var(--line);padding-top:11px;margin:14px 0 0}.group-heading{grid-column:1/-1;margin:12px 0 0;font-size:15px;font-weight:500;color:var(--muted)}.empty{padding:50px 20px;text-align:center;border:1px dashed var(--line);border-radius:10px;color:var(--muted);grid-column:1/-1}.empty strong{display:block;font-size:17px;font-weight:500;color:var(--text);margin-bottom:7px}.refresh-note{margin-bottom:14px;padding:11px 14px;border:1px solid #695033;background:#241e15;color:var(--entry);border-radius:8px;font-size:12px}.refresh-note button{border:1px solid #695033;background:transparent;border-radius:5px;padding:4px 9px;margin-left:12px}footer{margin-top:25px;padding-top:20px;border-top:1px solid var(--line);font-size:11px;color:var(--muted);display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}footer details{max-width:80ch}footer summary{cursor:pointer}dialog{color:var(--text);background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:0;width:min(950px,calc(100vw - 30px));max-height:90vh}dialog::backdrop{background:#030609d9;backdrop-filter:blur(6px)}.photo-dialog-heading{padding:14px 18px;display:flex;justify-content:space-between;align-items:center;gap:15px}.photo-dialog-heading h2{font-size:16px;margin:0;font-weight:500}.photo-dialog-heading button{background:var(--raised);border:1px solid var(--line);border-radius:6px;padding:7px 12px}#photo-large{display:block;width:100%;max-height:66vh;object-fit:contain;background:#030609}#photo-context{font-size:12px;color:var(--muted);padding:14px 18px;margin:0}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(min-width:1650px){.vehicle-list{grid-template-columns:repeat(5,minmax(0,1fr))}}@media(max-width:1250px){.vehicle-list{grid-template-columns:repeat(3,minmax(0,1fr))}.toolbar{gap:10px}.group-control{margin-left:0}.summary>div{padding:14px}.summary span{font-size:11px}}@media(max-width:850px){main{padding:20px}.heading{gap:15px}.generated{font-size:10px}.summary{gap:8px}.summary>div{padding:13px 11px}.summary strong{font-size:25px}.vehicle-list{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.toolbar{position:static}.search{max-width:none}.filters{order:3}.group-control{margin-left:auto}}@media(max-width:560px){main{padding:18px 12px 26px}.heading{display:block}.generated{text-align:left;margin-top:14px;padding:0;display:flex;gap:8px;flex-wrap:wrap}.generated strong{margin:0}h1{font-size:27px}.summary{grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}.summary>div{padding:12px}.summary strong{font-size:24px}.summary span{font-size:10px;margin-top:7px}.summary>div:nth-child(4),.summary>div:nth-child(5){grid-column:auto}.toolbar{gap:9px}.search{flex-basis:100%}.filters{order:0;width:100%;justify-content:space-between}.filters button{padding:7px 9px;font-size:11px}.group-control{margin-left:0}.vehicle-list{grid-template-columns:1fr}.photo-button{aspect-ratio:16/9}.card-copy h2{font-size:17px}.card-meta{font-size:11px}.card-heading>time{font-size:13px}.direction{font-size:11px}footer{display:block}footer details{margin-top:12px}.results-heading{font-size:11px}.results-heading h2{font-size:15px}}@media(prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important}}
'''

TEMPLATE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><title>Vehicles · BarrierWatcher</title><style>__STYLES__</style></head><body><main><header class="heading"><div><span class="eyebrow">Local observation history</span><h1>Vehicles</h1><p class="intro">Review arrivals, departures and unclear sightings. Appearance suggestions help compare photos; they do not identify a vehicle or confirm who is on site.</p></div><div id="report-metadata" class="generated">Report generated<strong>__GENERATED__</strong></div></header><section id="summary" class="summary" aria-label="Saved observations in this report">__SUMMARY__</section><div class="toolbar" aria-label="Filter vehicle observations"><label class="search"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/></svg><span class="sr-only">Search observations</span><input id="search" type="search" placeholder="Search type, colour or time" autocomplete="off"></label><div class="filters" role="group" aria-label="Observed direction"><button data-direction-filter="all" aria-pressed="true">All sightings</button><button data-direction-filter="in" aria-pressed="false">Entrance</button><button data-direction-filter="out" aria-pressed="false">Exit</button><button data-direction-filter="unclear" aria-pressed="false">Unclear</button></div><label class="group-control">Arrange<select id="grouping"><option value="recent">Newest first</option><option value="direction">By direction</option></select></label></div><div id="refresh-note" class="refresh-note" role="status" hidden>New history could not be loaded. This report remains available.<button id="retry-refresh" type="button">Try again</button></div><div class="results-heading"><h2>Saved sightings</h2><span id="result-count" aria-live="polite">__COUNT__ observations</span></div><section id="vehicle-list" class="vehicle-list" aria-label="Saved vehicle photos">__CARDS__</section><div id="empty" class="empty"__EMPTY_HIDDEN__><strong>No sightings to show</strong><span>Try a different filter or search. New observations appear when collected.</span></div><footer><span>Local history · __RETENTION__ hour retention · no plate reading</span><details><summary>Review and authorisation</summary><p>Suggestions compare appearance only. Lighting, similar vehicles and missed sightings can change a match. This is not a live occupancy list.</p><p>Authorisation: __AUTHORISED__</p></details></footer></main><dialog id="photo-dialog" aria-labelledby="photo-title"><div class="photo-dialog-heading"><h2 id="photo-title">Vehicle observation</h2><button id="close-photo" type="button">Close</button></div><img id="photo-large" alt=""><p id="photo-context"></p></dialog><script>__SCRIPT__</script></body></html>'''

SCRIPT = r'''
(()=>{'use strict';
const $=id=>document.getElementById(id),prefs={query:'',direction:'all',grouping:'recent'};
try{Object.assign(prefs,JSON.parse(sessionStorage.getItem('bw-vehicle-filters')||'{}'));}catch{}
if(!['all','in','out','unclear'].includes(prefs.direction))prefs.direction='all';
if(!['recent','direction'].includes(prefs.grouping))prefs.grouping='recent';
if(typeof prefs.query!=='string')prefs.query='';
$('search').value=prefs.query;$('grouping').value=prefs.grouping;
function applyFilters(){
 const query=prefs.query.trim().toLowerCase(),list=$('vehicle-list');let visible=0;
 list.querySelectorAll('.group-heading').forEach(node=>node.remove());
 const groupCounts={in:0,out:0,unclear:0},ranks={in:0,out:1,unclear:2};
 list.querySelectorAll('.vehicle-card').forEach((card,index)=>{
  const direction=card.dataset.direction,shown=(prefs.direction==='all'||direction===prefs.direction)&&(!query||card.dataset.search.includes(query));
  card.hidden=!shown;if(shown){visible++;groupCounts[direction]++;}
  card.style.order=String(prefs.grouping==='direction'?(ranks[direction]+1)*100000+index:index);
 });
 if(prefs.grouping==='direction')for(const [direction,label] of Object.entries({in:'Entrance observations',out:'Exit observations',unclear:'Direction unclear'})){
  if(!groupCounts[direction])continue;const heading=document.createElement('h2');heading.className='group-heading';heading.textContent=label;heading.style.order=String((ranks[direction]+1)*100000-1);list.append(heading);
 }
 $('empty').hidden=visible>0;$('result-count').textContent=`${visible} of ${list.querySelectorAll('.vehicle-card').length} observations`;
 document.querySelectorAll('[data-direction-filter]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.directionFilter===prefs.direction)));
 try{sessionStorage.setItem('bw-vehicle-filters',JSON.stringify(prefs));}catch{}
}
$('search').addEventListener('input',()=>{prefs.query=$('search').value;applyFilters();});
$('grouping').addEventListener('change',()=>{prefs.grouping=$('grouping').value;applyFilters();});
document.querySelectorAll('[data-direction-filter]').forEach(button=>button.addEventListener('click',()=>{prefs.direction=button.dataset.directionFilter;applyFilters();}));
let photoReturn=null;
$('vehicle-list').addEventListener('click',event=>{
 const button=event.target.closest('[data-photo]');if(!button)return;const card=button.closest('.vehicle-card'),image=button.querySelector('img');if(!image)return;
 photoReturn={id:card.dataset.sighting};$('photo-large').src=image.src;$('photo-large').alt=image.alt;
 $('photo-title').textContent=card.querySelector('h2').textContent;$('photo-context').textContent=card.querySelector('.observed').textContent+' · '+card.querySelector('.direction').textContent;
 $('photo-dialog').showModal();
});
$('close-photo').addEventListener('click',()=>$('photo-dialog').close());
$('photo-dialog').addEventListener('close',()=>{
 $('photo-large').removeAttribute('src');const card=[...$('vehicle-list').querySelectorAll('.vehicle-card')].find(card=>card.dataset.sighting===photoReturn?.id);card?.querySelector('[data-photo]')?.focus({preventScroll:true});photoReturn=null;
 if(requestedVersion!==appliedVersion)refreshReport();
});
let requestedVersion='',appliedVersion='',refreshing=false,retries=0,retryTimer=null;
async function refreshReport(){
 if(refreshing||!requestedVersion||requestedVersion===appliedVersion||$('photo-dialog').open)return;
 refreshing=true;const version=requestedVersion,controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),6000);
 try{
  const response=await fetch('/car-memory?t='+Date.now(),{cache:'no-store',signal:controller.signal});if(!response.ok)throw new Error('Report unavailable');
  const parsed=new DOMParser().parseFromString(await response.text(),'text/html');
  const nextList=parsed.getElementById('vehicle-list'),nextSummary=parsed.getElementById('summary'),nextMetadata=parsed.getElementById('report-metadata');
  if(!nextList||!nextSummary||!nextMetadata)throw new Error('Incomplete report');
  // Defer result replacement while a result has keyboard focus; the toolbar never changes.
  if($('vehicle-list').contains(document.activeElement))return;
  const scrollX=window.scrollX,scrollY=window.scrollY;
  const openIds=new Set([...$('vehicle-list').querySelectorAll('.vehicle-card')].filter(card=>card.querySelector('details[open]')).map(card=>card.dataset.sighting));
  $('vehicle-list').replaceChildren(...nextList.childNodes);$('summary').replaceChildren(...nextSummary.childNodes);$('report-metadata').replaceChildren(...nextMetadata.childNodes);
  $('vehicle-list').querySelectorAll('.vehicle-card').forEach(card=>{if(openIds.has(card.dataset.sighting)){const details=card.querySelector('details');if(details)details.open=true;}});
  applyFilters();window.scrollTo(scrollX,scrollY);appliedVersion=version;retries=0;$('refresh-note').hidden=true;
 }catch{$('refresh-note').hidden=false;if(retries++<2){clearTimeout(retryTimer);retryTimer=setTimeout(refreshReport,5000);}}
 finally{clearTimeout(timeout);refreshing=false;if(requestedVersion!==version)refreshReport();}
}
$('vehicle-list').addEventListener('focusout',()=>{setTimeout(()=>{if(!document.hidden&&requestedVersion!==appliedVersion)refreshReport();},0);});
$('retry-refresh').addEventListener('click',()=>{retries=0;refreshReport();});
window.addEventListener('message',event=>{
 if(event.source!==window.parent||event.origin!==location.origin||event.data?.type!=='barrierwatcher:report-refresh'||typeof event.data.reportVersion!=='string'||event.data.reportVersion.length>150)return;
 if(event.data.reportVersion!==requestedVersion)retries=0;requestedVersion=event.data.reportVersion;refreshReport();
});
window.addEventListener('pagehide',()=>clearTimeout(retryTimer));
applyFilters();
})();
'''
