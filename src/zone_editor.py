"""Local zone editor and strict, atomic configuration updates.

The HTTP handler must hold the watcher's state lock while checking armed state,
saving, and replacing the in-memory zones. No camera reader is created here.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import uuid


FIELDS = {"name", "rect", "motion", "light", "severity", "light_severity",
          "motion_pct", "light_delta", "on_frames", "off_frames"}


def _number(value, label: str, low: float, high: float, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    if not math.isfinite(value) or not low <= value <= high or (positive and value <= 0):
        raise ValueError(f"{label} must be {'greater than 0 and ' if positive else ''}between {low} and {high}.")
    return value


def validate_config(payload: dict, original: dict | None = None) -> dict:
    """Validate editable fields and retain the file's existing top-level metadata."""
    if not isinstance(payload, dict) or not isinstance(payload.get("zones"), list):
        raise ValueError("Configuration must contain a zones list.")
    if not 1 <= len(payload["zones"]) <= 32:
        raise ValueError("Keep between 1 and 32 zones.")
    result = copy.deepcopy(original or {})
    result["zones"] = copy.deepcopy(payload["zones"])
    names = set()
    for index, zone in enumerate(result["zones"], 1):
        label = f"Zone {index}"
        if not isinstance(zone, dict) or set(zone) - FIELDS:
            raise ValueError(f"{label} contains unsupported fields.")
        name = zone.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError(f"{label} needs a label of 1–80 characters.")
        name = name.strip()
        if name.casefold() in names:
            raise ValueError("Zone labels must be unique.")
        names.add(name.casefold())
        zone["name"] = name
        rect = zone.get("rect")
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            raise ValueError(f"{label} rectangle must contain x, y, width, height.")
        x, y, width, height = [_number(v, f"{label} rectangle", 0, 1) for v in rect]
        if width <= 0 or height <= 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
            raise ValueError(f"{label} rectangle must have positive dimensions and stay inside the image.")
        zone["rect"] = [x, y, width, height]
        for field in ("motion", "light"):
            if field in zone and not isinstance(zone[field], bool):
                raise ValueError(f"{label} {field} must be true or false.")
        for field in ("severity", "light_severity"):
            if field in zone and zone[field] not in ("early", "main", "fault"):
                raise ValueError(f"{label} {field} must be early, main, or fault.")
        for field, maximum in (("motion_pct", 100), ("light_delta", 255)):
            if field in zone:
                _number(zone[field], f"{label} {field}", 0, maximum, positive=True)
        for field in ("on_frames", "off_frames"):
            if field in zone and (type(zone[field]) is not int or not 1 <= zone[field] <= 1000):
                raise ValueError(f"{label} {field} must be an integer from 1 to 1000.")
    return result


def read_config(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, list):
        raw = {"zones": raw}
    return validate_config(raw, raw)


def _atomic_write(path: Path, contents: bytes) -> None:
    filename = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            filename = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(filename, path)
    finally:
        if filename is not None:
            filename.unlink(missing_ok=True)


def save_config(path: Path, payload: dict, *, armed: bool) -> dict:
    """Save only while disarmed, retaining an exact dated backup before replacement."""
    if armed:
        raise PermissionError("Disarm the watcher before saving zone changes.")
    original = read_config(path)
    updated = validate_config(payload, original)
    contents = (json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.bak-{stamp}-{uuid.uuid4().hex[:8]}")
    _atomic_write(backup, path.read_bytes())
    _atomic_write(path, contents)
    return updated


def render_page(path: Path) -> str:
    config = json.dumps(read_config(path), ensure_ascii=False, allow_nan=False)
    # A label containing </script> must remain inert JSON, never HTML markup.
    return PAGE.replace("__ZONE_CONFIG__", config.replace("<", "\\u003c"))


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>c500-guard · Edit zones</title>
<style>
:root{color-scheme:dark;font:15px system-ui,sans-serif;background:#10151b;color:#e7edf4}*{box-sizing:border-box}body{margin:0;padding:24px;max-width:1280px;margin:auto}a{color:#9bc9ff}h1{font-size:28px;margin:16px 0 6px}p{color:#b9c6d6;line-height:1.5}.layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:22px;margin-top:20px}.preview{position:relative;align-self:start;border:1px solid #445064;background:#000;line-height:0}.preview img{width:100%;height:auto;display:block;min-height:180px}.preview canvas{position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair}.panel{background:#1b2430;border-radius:10px;padding:18px}label{display:block;margin:14px 0 5px;font-weight:600}input,select,button{font:inherit;border:1px solid #526073;border-radius:6px;padding:9px;background:#121a24;color:inherit}input:not([type=checkbox]),select{width:100%}input[type=checkbox]{margin-right:8px}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}.buttons{display:flex;gap:8px;margin:12px 0}#save{background:#a8d5ff;color:#0d2032;font-weight:700;width:100%;margin-top:15px}#status,#message{padding:10px 0;line-height:1.4}#message{color:#a8d5ff;white-space:pre-wrap}.small{font-size:13px;color:#bac7d6}.coords{font:12px ui-monospace,monospace;line-height:1.6;overflow-wrap:anywhere}@media(max-width:850px){body{padding:16px}.layout{grid-template-columns:1fr}.panel{max-width:none}}
</style></head><body><a href="/">← Dashboard</a><h1>Edit detection zones</h1>
<p>Reposition the camera, then choose a zone and drag a rectangle over the live image. Keep clocks, screen edges and unrelated displays outside your zones.</p>
<div id="status" role="status">Checking watcher state…</div><div class="layout"><div>
<div class="preview"><img id="frame" alt="Live camera preview"><canvas id="overlay" aria-label="Drag to draw the selected zone"></canvas></div>
<p class="small">Drag to replace the selected rectangle. Blue and gold rectangles show your edits; outlines already in the camera preview show the saved zones. Use x, y, width and height below for precise placement. Saving resets zone detection history; the watcher stays disarmed.</p></div>
<section class="panel" aria-label="Zone settings"><label for="zone">Selected zone</label><select id="zone"></select><div class="buttons"><button id="add">Add zone</button><button id="remove">Remove</button></div>
<label for="name">Label</label><input id="name" maxlength="80">
<label><input id="motion" type="checkbox">Detect motion</label><label for="motion_pct">Changed pixels (%)</label><input id="motion_pct" type="number" min="0.01" max="100" step="0.1">
<label><input id="light" type="checkbox">Detect light changes</label><label for="light_delta">Light change threshold (1–255)</label><input id="light_delta" type="number" min="0.01" max="255" step="1">
<label for="rect">Rectangle: x, y, width, height</label><input id="rect" class="coords" aria-describedby="rect-help"><p id="rect-help" class="small">Normalized coordinates from 0 to 1.</p>
<button id="save" disabled>Save zones</button><div id="message" role="status"></div></section></div>
<script id="initial-config" type="application/json">__ZONE_CONFIG__</script>
<script>
'use strict';
const config=JSON.parse(document.getElementById('initial-config').textContent),zones=config.zones;
const $=id=>document.getElementById(id),canvas=$('overlay'),ctx=canvas.getContext('2d');
let selected=0,armed=true,dirty=false,drag=null,frameReady=false;
function message(text){$('message').textContent=text}
function list(){const picker=$('zone');picker.replaceChildren();zones.forEach((z,i)=>{const o=document.createElement('option');o.value=i;o.textContent=z.name;picker.appendChild(o)});picker.value=selected;$('remove').disabled=zones.length<=1;$('add').disabled=zones.length>=32}
function fields(){const z=zones[selected];$('name').value=z.name;$('motion').checked=z.motion??true;$('light').checked=z.light??false;$('motion_pct').value=z.motion_pct??1.2;$('light_delta').value=z.light_delta??10;$('rect').value=z.rect.map(v=>Number(v.toFixed(5))).join(', ');draw()}
function draw(){const box=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.round(box.width));canvas.height=Math.max(1,Math.round(box.height));ctx.clearRect(0,0,canvas.width,canvas.height);zones.forEach((z,i)=>{let [x,y,w,h]=z.rect;const active=i===selected;ctx.strokeStyle=active?'#a8e8ff':'#ffcf75';ctx.fillStyle=active?'#a8e8ff25':'#ffcf7510';ctx.lineWidth=active?3:1.5;ctx.fillRect(x*canvas.width,y*canvas.height,w*canvas.width,h*canvas.height);ctx.strokeRect(x*canvas.width,y*canvas.height,w*canvas.width,h*canvas.height);ctx.font='bold 13px system-ui';ctx.fillStyle=active?'#d6f6ff':'#ffdf9b';ctx.shadowColor='#000';ctx.shadowBlur=4;ctx.fillText(z.name,x*canvas.width+5,Math.max(16,y*canvas.height+17));ctx.shadowBlur=0})}
function changed(){dirty=true;message('Unsaved changes.');draw()}
function commit(){const z=zones[selected],name=$('name').value.trim(),rect=$('rect').value.split(',').map(s=>Number(s.trim()));if(!name)throw Error('Enter a zone label.');if(rect.length!==4||rect.some(v=>!Number.isFinite(v)||v<0||v>1)||rect[2]<=0||rect[3]<=0||rect[0]+rect[2]>1.000000001||rect[1]+rect[3]>1.000000001)throw Error('Rectangle must fit within the image and have positive width and height.');const motion=Number($('motion_pct').value),light=Number($('light_delta').value);if(!(motion>0&&motion<=100&&light>0&&light<=255))throw Error('Check the motion and light thresholds.');Object.assign(z,{name,rect,motion:$('motion').checked,light:$('light').checked,motion_pct:motion,light_delta:light});list()}
for(const id of ['name','motion','light','motion_pct','light_delta','rect'])$(id).addEventListener('change',()=>{try{commit();changed()}catch(e){message(e.message)}});
$('zone').onchange=()=>{const next=Number($('zone').value);try{commit();selected=next;list();fields()}catch(e){$('zone').value=selected;message(e.message)}};
$('add').onclick=()=>{try{commit();let n=zones.length+1;while(zones.some(z=>z.name===`Zone ${n}`))n++;zones.push({name:`Zone ${n}`,rect:[.25,.25,.25,.25],motion:true,light:false,severity:'main',light_severity:'early',motion_pct:1.2,light_delta:10,on_frames:2,off_frames:8});selected=zones.length-1;list();fields();changed()}catch(e){message(e.message)}};
$('remove').onclick=()=>{if(zones.length>1){zones.splice(selected,1);selected=Math.min(selected,zones.length-1);list();fields();changed()}};
function point(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
canvas.onpointerdown=e=>{if(!frameReady)return;try{commit()}catch(err){message(err.message);return}drag={start:point(e),old:[...zones[selected].rect]};canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drag)return;const [x,y]=point(e),[a,b]=drag.start;zones[selected].rect=[Math.min(a,x),Math.min(b,y),Math.abs(x-a),Math.abs(y-b)].map(v=>Number(v.toFixed(6)));$('rect').value=zones[selected].rect.join(', ');draw()};
function endDrag(){if(!drag)return;if(zones[selected].rect[2]<.002||zones[selected].rect[3]<.002)zones[selected].rect=drag.old;drag=null;fields();changed()}
canvas.onpointerup=endDrag;canvas.onpointercancel=endDrag;window.addEventListener('resize',draw);
async function state(){try{const r=await fetch('/state',{cache:'no-store'});if(!r.ok)throw Error();const s=await r.json();armed=s.armed!==false;$('status').textContent=armed?'Watcher is armed. Disarm it on the dashboard before saving.':'Watcher is disarmed. Zone changes can be saved.';$('save').disabled=armed}catch{armed=true;$('save').disabled=true;$('status').textContent='Watcher is unavailable. Saving is disabled.'}setTimeout(state,1500)}
$('frame').onload=()=>{frameReady=true;draw();setTimeout(nextFrame,1000)};$('frame').onerror=()=>{frameReady=false;setTimeout(nextFrame,2000)};function nextFrame(){$('frame').src='/frame.jpg?t='+Date.now()}
$('save').onclick=async()=>{try{commit();if(armed)throw Error('Disarm the watcher before saving.');$('save').disabled=true;const r=await fetch('/zones',{method:'POST',headers:{'Content-Type':'application/json','X-Via':'zone-editor'},body:JSON.stringify({zones})});const result=await r.json();if(!r.ok||!result.ok)throw Error(result.error||'Could not save zones.');dirty=false;message('Zones saved. Previous configuration backed up. Watcher remains disarmed.')}catch(e){message(e.message)}finally{$('save').disabled=armed}};
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue=''}});list();fields();state();nextFrame();
</script></body></html>'''
