__c500Modules["runtime.js"] = (() => {
'use strict';
// Shared transport; the design preview has no command permission.
const clone = value => JSON.parse(JSON.stringify(value));
const points = zone => zone.polygon || (Array.isArray(zone.rect) ? [[zone.rect[0],zone.rect[1]],[zone.rect[0]+zone.rect[2],zone.rect[1]],[zone.rect[0]+zone.rect[2],zone.rect[1]+zone.rect[3]],[zone.rect[0],zone.rect[1]+zone.rect[3]]] : []);
function geometrySnapshot(zones) {
 return (zones || []).map(z=>({name:z.name,polygon:points(z),exclude_polygons:z.exclude_polygons||[]})).sort((a,b)=>a.name.localeCompare(b.name));
}
function mergeLiveZones(state, config) {
 return {...state,zones:(state.zones||[]).map(zone=>{
  const saved=config?.zones?.find(z=>z.name===zone.name);
  const matching=saved&&JSON.stringify(points(saved))===JSON.stringify(points(zone));
  return {...zone,polygon:points(zone),exclude_polygons:matching?(saved.exclude_polygons||[]):(zone.exclude_polygons||[])};
 })};
}
function mergeGeometryDraft(config,draft,baseline) {
 if(JSON.stringify(geometrySnapshot(config.zones))!==JSON.stringify(geometrySnapshot(baseline)))throw new Error('The live layout changed. Load the update before applying.');
 if(draft.length!==config.zones.length||new Set(draft.map(z=>z.name)).size!==draft.length)throw new Error('Zone names changed. Reload the live layout.');
 const result=clone(config);
 result.zones=result.zones.map(zone=>{
  const edit=draft.find(z=>z.name===zone.name);
  if(!edit||!Array.isArray(edit.polygon)||edit.polygon.length<3||edit.polygon.some(p=>!Array.isArray(p)||p.length!==2||p.some(v=>!Number.isFinite(v)||v<0||v>1)))throw new Error('Invalid zone outline.');
  // Exclusions and detector tuning are preserved from the latest configuration.
  zone.polygon=clone(edit.polygon);
  const all=[...zone.polygon,...(zone.confirmation_polygon||[]),...(zone.light_polygon||[])];
  const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
  const x=Math.min(...xs),y=Math.min(...ys);
  zone.rect=[x,y,Math.max(...xs)-x,Math.max(...ys)-y];
  return zone;
 });
 return result;
}
function createTransport({production=false,fetcher=globalThis.fetch.bind(globalThis),parseConfig}={}) {
 const path=value=>production?value.replace(/^\/api\//,'/'):value;
 async function read(value,signal){
  const config=value==='/api/config'&&production;
  const response=await fetcher(config?'/zones':path(value),{cache:'no-store',signal});
  if(!response.ok)throw new Error(`Read unavailable (${response.status})`);
  return config?parseConfig(await response.text()):response.json();
 }
 async function command(action,payload={}) {
  if(!production)throw new Error('Preview cannot control the live system.');
  if(!['ack','arm','disarm','test-tone','power/off','zones','barrier-calibration','feedback','alignment/calibrate','settings'].includes(action))throw new Error('Unsupported command.');
  const response=await fetcher('/'+action,{method:'POST',headers:{'Content-Type':'application/json','X-Via':'web'},body:JSON.stringify(payload),signal:AbortSignal.timeout(12000)});
  let data;try{data=await response.json();}catch{throw new Error('No command receipt. Check the live system before retrying.');}
  if(!response.ok||data.ok!==true){const error=new Error(data.error||data.message||data.reason||`Command rejected (${response.status})`);error.receipt=data;throw error;}
  if(action==='ack'&&data.acknowledged!==true)throw new Error('That alarm has changed or was already acknowledged.');
  return data;
 }
 return {path,read,command};
}


return Object.freeze({"geometrySnapshot": geometrySnapshot, "mergeLiveZones": mergeLiveZones, "mergeGeometryDraft": mergeGeometryDraft, "createTransport": createTransport});
})();

