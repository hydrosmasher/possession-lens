'use strict';
const Lens=(()=>{
 const colours={H:'#07519a',A:'#a51b46',D:'#b9b4a4',X:'#e0e5ed'};
 const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function validate(d){
  if(!d||d.schema_version!==2||!['synthetic_demo','licensed_local','imported_csv'].includes(d.mode))throw Error('Open a Possession Lens analysis JSON or a frame,label CSV.');
  if(!Number.isInteger(d.n_frames)||d.n_frames<1||d.n_frames>2000000||d.fps!==25)throw Error('This analysis needs a valid 25 fps timeline.');
  if(!Array.isArray(d.timeline)||d.timeline.length>1000000||!Array.isArray(d.blocks)||!d.blocks.length)throw Error('The analysis has no valid timeline.');
  let last=0;for(const seg of d.timeline){if(!Array.isArray(seg)||seg.length!==3||!Number.isInteger(seg[0])||!Number.isInteger(seg[1])||seg[0]!==last||seg[1]<=seg[0]||seg[1]>d.n_frames||!Object.hasOwn(colours,seg[2]))throw Error('Timeline intervals must cover the recording without overlaps or gaps.');last=seg[1];}if(last!==d.n_frames)throw Error('Timeline does not cover the whole recording.');
  let end=0;for(const b of d.blocks){if(!Array.isArray(b)||b.length!==2||!b.every(Number.isInteger)||b[0]<end||b[1]<=b[0]||b[1]>d.n_frames)throw Error('Invalid analysed periods.');end=b[1];}
  if(!d.teams||!['H','A'].every(k=>typeof d.teams[k]==='string'&&d.teams[k].length>0&&d.teams[k].length<=120))throw Error('Team names are missing or invalid.');
  if(d.tracklets){if(!Array.isArray(d.tracklets.rows)||d.tracklets.rows.length>1000)throw Error('Invalid anonymous-track summary.');for(const r of d.tracklets.rows){if(!['H','A'].includes(r.team)||!(r.track_slot==='UNASSIGNED'||(Number.isInteger(r.track_slot)&&r.track_slot>0))||!Number.isFinite(r.associated_control_seconds)||r.associated_control_seconds<0||!(r.percent_of_team_possession===null||(Number.isFinite(r.percent_of_team_possession)&&r.percent_of_team_possession>=0&&r.percent_of_team_possession<=100)))throw Error('Anonymous-track values are invalid.');}}
  return d;
 }
 function stats(d,period='all'){
  let blocks=d.blocks;if(period==='first')blocks=[d.blocks[0]];if(period==='second')blocks=[d.blocks[1]||d.blocks[0]];
  const counts={H:0,A:0,D:0,X:0};let spells=0,total=0,longest=0,longestTeam=null;
  for(const [a,b] of blocks)for(const [s,e,k] of d.timeline){const n=Math.max(0,Math.min(e,b)-Math.max(s,a));if(!n)continue;counts[k]+=n;if(k==='H'||k==='A'){spells++;total+=n;if(n>longest){longest=n;longestTeam=k;}}}
  const live=counts.H+counts.A;return{counts,blocks,spells,average:spells?total/d.fps/spells:0,longest:longest/d.fps,longestTeam,live:live/d.fps,home:live?100*counts.H/live:null,away:live?100*counts.A/live:null,analysed:blocks.reduce((n,[a,b])=>n+b-a,0)/d.fps};
 }
 function csv(text){
  const lines=text.replace(/^\uFEFF/,'').trimEnd().split(/\r?\n/);if(lines[0].trim().toLowerCase()!=='frame,label'||lines.length<2)throw Error('CSV needs the header frame,label and one row per frame.');
  const timeline=[];let current=null,start=0;
  for(let i=1;i<lines.length;i++){const parts=lines[i].split(',');const k=(parts[1]||'').trim().toUpperCase();if(parts.length!==2||!/^\d+$/.test(parts[0].trim())||Number(parts[0])!==i-1||!Object.hasOwn(colours,k))throw Error(`The CSV has an invalid row at ${i}.`);if(k!==current){if(current!==null)timeline.push([start,i-1,current]);start=i-1;current=k;}}
  const n=lines.length-1;timeline.push([start,n,current]);return validate({schema_version:2,mode:'imported_csv',title:'Imported possession analysis',model_name:'Imported labels · model not verified',n_frames:n,fps:25,teams:{H:'Home team',A:'Away team'},blocks:n===199319?[[11185,90652],[110808,190198]]:[[0,n]],timeline,tracklets:null});
 }
 return{validate,stats,csv,colours,escape};
})();
if(typeof module!=='undefined')module.exports=Lens;
if(typeof document!=='undefined'){
 const $=id=>document.getElementById(id);let data,original,playerTeam='H',period='all';
 const text=(id,value)=>{$(id).textContent=value;};
 const pct=n=>n===null?'—':`${n.toFixed(1)}%`;
 const duration=seconds=>{const s=Math.round(seconds);return s>=60?`${Math.floor(s/60)}m ${String(s%60).padStart(2,'0')}s`:`${s}s`;};
 const clock=frame=>{const s=Math.floor(frame/25);return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;};
 function render(){
  const s=Lens.stats(data,period);const demo=data.mode==='synthetic_demo';
  text('data-badge',demo?'DEMO DATA':data.mode==='licensed_local'?'LOCAL MATCH ANALYSIS':'IMPORTED LABELS');$('data-badge').classList.toggle('real',!demo);
  text('context-note',demo?'Illustrative figures, not the organiser match.':data.mode==='licensed_local'?'Model estimates from local tracking. No match-accuracy claim.':'Read locally in this browser. Model provenance is not verified.');
  $('reset-demo').hidden=data===original;text('reset-demo',original.mode==='synthetic_demo'?'Back to demo':'Back to original match');
  text('match-title',data.title||`${data.teams.H} vs ${data.teams.A}`);text('model-label',data.model_name||'Possession model estimates');
  text('estimate-label',demo?'Illustrative figures':'Model estimates');
  for(const [k,id]of [['H','home'],['A','away']]){text(`${id}-name`,data.teams[k]);text(`legend-${id}`,data.teams[k]);text(`${id}-share`,pct(s[id]));text(`${id}-time`,`${duration(s.counts[k]/25)} of team control`);$(`${id}-bar`).style.width=`${s[id]||0}%`;}
  text('share-note',s.home===null?'No live team possession in this selection.':'Share of live team control only · stopped and unknown play excluded');
  text('spells',s.spells.toLocaleString());text('average',`${s.average.toFixed(1)}s`);text('longest',duration(s.longest));text('longest-team',s.longestTeam?`${data.teams[s.longestTeam]}'s longest stretch`:'No live possession');text('analysed',duration(s.analysed));
  text('live-time',duration(s.live));text('dead-time',duration(s.counts.D/25));text('unknown-time',duration(s.counts.X/25));
  $('period').querySelector('option[value="second"]').disabled=data.blocks.length<2;
  if(s.home===null){text('takeaway','No live team control in this selection.');text('takeaway-detail','Try another match period or open a different analysis.');}
  else{const k=s.home>=s.away?'H':'A',share=Math.max(s.home,s.away);text('takeaway',Math.abs(s.home-s.away)<2?'An evenly balanced possession contest.':`${data.teams[k]} held more of the ball.`);text('takeaway-detail',`${data.teams[k]} had ${share.toFixed(1)}% of estimated live team control${period==='all'?' across the match':' in this half'}. Possession alone does not measure attacking quality.`);}
  drawTimeline(s);renderPlayers();
 }
 function drawTimeline(s){
  const a=period==='all'?0:s.blocks[0][0],b=period==='all'?data.n_frames:s.blocks.at(-1)[1];
  const ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 1000 66');svg.setAttribute('preserveAspectRatio','none');
  for(const [start,end,k] of data.timeline){const lo=Math.max(a,start),hi=Math.min(b,end);if(hi<=lo)continue;const rect=document.createElementNS(ns,'rect');rect.setAttribute('x',(lo-a)/(b-a)*1000);rect.setAttribute('width',(hi-lo)/(b-a)*1000);rect.setAttribute('height','66');rect.setAttribute('fill',Lens.colours[k]);
   const name=k==='H'||k==='A'?data.teams[k]:k==='D'?'Stopped play':'Outside analysed play / unknown';const label=`${clock(lo)}–${clock(hi)} · ${name} · ${duration((hi-lo)/25)}`;const title=document.createElementNS(ns,'title');title.textContent=label;rect.append(title);rect.addEventListener('mouseenter',()=>text('timeline-detail',label));rect.addEventListener('click',()=>text('timeline-detail',label));svg.append(rect);}
  $('timeline').replaceChildren(svg);$('timeline-ticks').replaceChildren();for(let i=0;i<=6;i++){const span=document.createElement('span');span.textContent=clock(a+(b-a)*i/6);$('timeline-ticks').append(span);}text('timeline-detail','Video time · hover or tap a stretch to explore team control.');
 }
 function renderPlayers(){
  text('players-home',data.teams.H);text('players-away',data.teams.A);$('players-home').classList.toggle('active',playerTeam==='H');$('players-away').classList.toggle('active',playerTeam==='A');text('player-team-title',`${data.teams[playerTeam]} · full match`);
  const rows=(data.tracklets?.rows||[]).filter(r=>r.team===playerTeam);const unassigned=rows.find(r=>r.track_slot==='UNASSIGNED');const assigned=rows.filter(r=>r.track_slot!=='UNASSIGNED').reduce((n,r)=>n+(r.percent_of_team_possession||0),0);
  text('associated-share',rows.some(r=>r.percent_of_team_possession!==null)?pct(assigned):'—');text('unassigned-share',unassigned?pct(unassigned.percent_of_team_possession):'—');
  const sorted=rows.filter(r=>r.track_slot!=='UNASSIGNED'&&r.associated_control_seconds>0).sort((a,b)=>b.associated_control_seconds-a.associated_control_seconds);
  if(unassigned)sorted.push(unassigned);$('player-rows').replaceChildren();
  for(const r of sorted){const unknown=r.track_slot==='UNASSIGNED';const row=document.createElement('div');row.className=`player-row ${playerTeam==='A'?'away-track':''} ${unknown?'unassigned-row':''}`;
   const icon=document.createElement('span');icon.className='track-icon';icon.textContent=unknown?'?':`T${r.track_slot}`;const label=document.createElement('span');label.className='track-name';label.textContent=unknown?'Unassigned control':`Anonymous track ${r.track_slot}`;const note=document.createElement('small');note.textContent=unknown?'No defensible individual attribution':'Identity unverified · may be a fragment';label.append(note);
   const bar=document.createElement('div');bar.className='track-bar';const fill=document.createElement('span');fill.style.width=`${r.percent_of_team_possession||0}%`;bar.append(fill);const value=document.createElement('span');value.className='track-value';value.textContent=pct(r.percent_of_team_possession);const time=document.createElement('span');time.className='track-time';time.textContent=duration(r.associated_control_seconds);row.append(icon,label,bar,value,time);$('player-rows').append(row);}
  text('player-note',!rows.length?'No track-level attribution was included in this analysis. Open an analysis JSON with anonymous-track estimates.':`${data.mode==='synthetic_demo'?'Illustrative demo tracks. ':''}Full-match figures; half filters affect the team overview only. Shares include unassigned time and use the selected team's total possession as denominator. ${data.tracklets?.ambiguity_note||''}`);
 }
 function chooseTab(button){for(const b of document.querySelectorAll('.tab')){const active=b===button;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));}for(const p of document.querySelectorAll('.panel'))p.hidden=p.id!==button.dataset.panel;$('period').disabled=button.dataset.panel==='players';}
 document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>chooseTab(b)));
 $('period').addEventListener('change',()=>{period=$('period').value;render();});
 $('players-home').addEventListener('click',()=>{playerTeam='H';renderPlayers();});$('players-away').addEventListener('click',()=>{playerTeam='A';renderPlayers();});
 $('open-analysis').addEventListener('click',()=>$('analysis-file').click());
 $('analysis-file').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;try{if(f.size>32*1024*1024)throw Error('Choose an analysis smaller than 32 MB.');const source=await f.text();const candidate=f.name.toLowerCase().endsWith('.csv')?Lens.csv(source):Lens.validate(JSON.parse(source));data=candidate;period='all';playerTeam='H';$('period').value='all';render();text('status','Match analysis opened locally. Nothing was uploaded.');}catch(error){text('status',`Could not open that analysis: ${error.message}`);}finally{e.target.value='';}});
 $('reset-demo').addEventListener('click',()=>{data=original;period='all';playerTeam='H';$('period').value='all';render();text('status','Showing the original dashboard data.');});
 $('download-tracks').addEventListener('click',()=>{const rows=(data.tracklets?.rows||[]).filter(r=>r.team===playerTeam);if(!rows.length){text('status','No anonymous-track estimates are available to save.');return;}const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';const values=[['team','track_slot','associated_control_seconds','percent_of_team_possession','identity_status'],...rows.map(r=>[data.teams[r.team],r.track_slot,r.associated_control_seconds,r.percent_of_team_possession,'anonymous unverified track; proximity is not confirmed contact'])];const blob=new Blob([values.map(r=>r.map(q).join(',')).join('\n')+'\n'],{type:'text/csv'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='possession-lens-anonymous-tracks.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
 $('download-summary').addEventListener('click',()=>{const s=Lens.stats(data,period);const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';const rows=[['team','possession_percent','control_seconds','period','status'],[data.teams.H,s.home,s.counts.H/25,period,data.mode],[data.teams.A,s.away,s.counts.A/25,period,data.mode]];const blob=new Blob([rows.map(r=>r.map(q).join(',')).join('\n')+'\n'],{type:'text/csv'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='possession-lens-summary.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
 const load=window.__POSSESSION_LENS_DATA__?Promise.resolve(window.__POSSESSION_LENS_DATA__):fetch('/api/dashboard').then(r=>{if(!r.ok)throw Error('Local analysis unavailable');return r.json();});
 load.then(d=>{data=Lens.validate(d);original=data;render();}).catch(error=>text('status',`The dashboard could not load: ${error.message}. Open dashboard.html or start the local dashboard launcher.`));
}
