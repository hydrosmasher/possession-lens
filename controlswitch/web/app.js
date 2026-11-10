'use strict';
const $=id=>document.getElementById(id);
const colours={H:'#287157',A:'#416dc7',D:'#d29a48',X:'#a9aaa8'};
let summary,timelines,selected,view=[0,1],custom=false;
const number=n=>n.toLocaleString('en-US');
function clock(frames){const seconds=Math.floor(frames/summary.fps);return `${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`;}
function labelName(k){return k==='H'?summary.teams.H:k==='A'?summary.teams.A:k==='D'?'Dead play':'X / no answer';}
function make(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function choose(){selected=summary.variants.find(v=>v.name===$('variant').value)||summary.variants[0];render();}
function fillVariants(){const select=$('variant');select.replaceChildren();for(const v of summary.variants){const option=make('option',v.name==='imported_csv'?'Imported CSV':v.name.replace('baseline_',''));option.value=v.name;select.append(option);}select.value=selected.name;}
function render(){
 const real=summary.mode==='licensed_local';
 $('mode').textContent=custom?'LOCAL CSV INSPECTION':real?'LICENSED DATA / LOCAL ONLY':'SYNTHETIC DEMO';$('mode').classList.toggle('real',real||custom);
 $('notice').textContent=custom?'Imported labels were checked in this browser. No upload occurred. Run the official validator before submission.':real?'Baseline reproduction only. All four CSVs passed format validation; accuracy has not been measured. Generated match outputs stay local.':'This walkthrough is entirely synthetic. It contains no competition match data, ground truth, or accuracy result.';
 $('run-title').textContent=custom?'Your submission timeline':summary.title;
 $('frames').textContent=number(summary.n_frames);$('duration').textContent=`${clock(summary.n_frames)} video time · ${summary.fps} fps`;
 $('spells').textContent=number(selected.possession_count);$('mean').textContent=`${selected.mean_possession_seconds.toFixed(1)} s`;
 $('quality').textContent=summary.audit?`${(100*summary.audit.ball_observed_fraction_graded).toFixed(1)}%`:'—';
 $('quality-note').textContent=summary.audit?'Within graded halves':'No tracking loaded';
 $('window-note').textContent=selected.name==='imported_csv'?'Imported labels · no smoothing applied here':selected.half_window_frames?`Centred majority · ±${selected.half_window_frames} frames · ${(2*selected.half_window_frames+1)/summary.fps} s total window`:'Nearest tracked player · no temporal smoothing';
 const shares=$('shares');shares.replaceChildren();for(const k of ['H','A','D','X']){const frac=selected.counts[k]/selected.coverage_frames;const row=make('div',undefined,'share-row');const text=make('div',undefined,'share-text');text.append(make('span',labelName(k)),make('span',`${(frac*100).toFixed(1)}% · ${number(selected.counts[k])} frames`));const track=make('div',undefined,'bar-track');const bar=make('div',undefined,'bar-fill');bar.style.backgroundColor=colours[k];bar.style.width=`${frac*100}%`;track.append(bar);row.append(text,track);shares.append(row);}
 const table=$('comparison');table.replaceChildren();for(const v of summary.variants){const tr=make('tr');tr.classList.toggle('selected',v.name===selected.name);for(const text of [v.name,v.name==='imported_csv'?'Unmodified':`±${v.half_window_frames} frames`,number(v.possession_count),number(v.short_possessions_under_1s),v.browser_valid?'Browser checked':v.format_valid?'Passed official check':'Demo only'])tr.append(make('td',text));const td=make('td');if(real&&v.name!=='imported_csv'){const a=make('a','Download CSV');a.href=`/download/${v.name}.csv`;td.append(a);}else td.textContent='—';tr.append(td);table.append(tr);}
 $('provenance').textContent=JSON.stringify({status:summary.model_status,scoring:summary.score_status,upstream_commit:summary.upstream_commit||null,versions:summary.versions||null,selected_output_sha256:selected.sha256||null},null,2);
 $('window').querySelector('[value=second]').disabled=summary.blocks.length<2;
 draw();
}
function draw(){
 if(!summary||!selected)return;
 let start=0,end=summary.n_frames;
 if($('window').value!=='all'){const b=summary.blocks[$('window').value==='first'?0:1]||summary.blocks[0];[start,end]=b;}
 const span=Number($('zoom').value)*summary.fps;
 if(span&&span<end-start){start+=Math.round((end-start-span)*Number($('position').value)/1000);end=start+span;}
 view=[start,end];$('start-time').textContent=`${clock(start)} · frame ${number(start)}`;$('end-time').textContent=`${clock(end)} · frame ${number(end-1)}`;
 const canvas=$('timeline'),dpr=window.devicePixelRatio||1,w=canvas.parentElement.clientWidth,h=126;
 canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.fillStyle='#f1f2eb';ctx.fillRect(0,0,w,h);
 for(const [a,b,k] of timelines[selected.name]){if(b<=start||a>=end)continue;ctx.fillStyle=colours[k];const x=(Math.max(a,start)-start)/(end-start)*w,ww=(Math.min(b,end)-Math.max(a,start))/(end-start)*w;ctx.fillRect(x,20,Math.max(.65,ww),85);}
 let cursor=0;const gaps=[];for(const [a,b] of summary.blocks){if(a>cursor)gaps.push([cursor,a]);cursor=b;}if(cursor<summary.n_frames)gaps.push([cursor,summary.n_frames]);
 for(const [a,b] of gaps){if(b<=start||a>=end)continue;ctx.fillStyle='#dddeda';const x=(Math.max(a,start)-start)/(end-start)*w,ww=(Math.min(b,end)-Math.max(a,start))/(end-start)*w;ctx.fillRect(x,0,ww,h);if(ww>65){ctx.fillStyle='#6b746d';ctx.font='10px Arial';ctx.fillText('UNGRADED',x+8,68);}}
 ctx.strokeStyle='#ffffff55';for(let i=1;i<8;i++){ctx.beginPath();ctx.moveTo(w*i/8,20);ctx.lineTo(w*i/8,105);ctx.stroke();}
}
function inspect(event){const canvas=$('timeline');const t=Math.min(view[1]-1,Math.max(view[0],Math.floor(view[0]+(event.clientX-canvas.getBoundingClientRect().left)/canvas.clientWidth*(view[1]-view[0]))));const segs=timelines[selected.name];let lo=0,hi=segs.length-1;while(lo<=hi){const mid=(lo+hi)>>1,[a,b,k]=segs[mid];if(t<a)hi=mid-1;else if(t>=b)lo=mid+1;else{const graded=summary.blocks.some(([x,y])=>t>=x&&t<y);$('inspection').textContent=`${clock(t)} · frame ${number(t)} · ${labelName(k)} · interval ${number(a)}–${number(b-1)} · ${((b-a)/summary.fps).toFixed(2)} s${graded?'':' · ungraded frame'}`;return;}}}
function analyse(labels,blocks){let count=0,short=0,total=0;const counts={H:0,A:0,D:0,X:0};for(const [a,b] of blocks){let s=a;for(let i=a;i<b;i++){counts[labels[i]]++;if(i+1===b||labels[i+1]!==labels[i]){if(labels[i]==='H'||labels[i]==='A'){count++;total+=i+1-s;if(i+1-s<25)short++;}s=i+1;}}}return{counts,possession_count:count,short_possessions_under_1s:short,mean_possession_seconds:count?total/25/count:0,coverage_frames:blocks.reduce((n,[a,b])=>n+b-a,0)};}
async function importCSV(event){
 const file=event.target.files[0];if(!file)return;
 try{if(file.size>8*1024*1024)throw Error('File exceeds 8 MB; expected a 199,319-row submission.');$('import-status').textContent='Checking CSV locally…';const lines=(await file.text()).replace(/^\uFEFF/,'').trimEnd().split(/\r?\n/);if(lines[0].split(',').map(s=>s.trim().toLowerCase()).join(',')!=='frame,label')throw Error('Header must be frame,label.');if(lines.length!==199320)throw Error(`Expected 199,319 rows; found ${number(lines.length-1)}.`);const labels=[];const segs=[];let start=0;for(let i=1;i<lines.length;i++){const parts=lines[i].split(',');const label=(parts[1]||'').trim().toUpperCase();if(parts.length!==2||!/^\d+$/.test(parts[0].trim())||Number(parts[0])!==i-1||!['H','A','D','X'].includes(label))throw Error(`Invalid frame or label at data row ${number(i)}.`);labels.push(label);if(i>1&&label!==labels[i-2]){segs.push([start,i-1,labels[i-2]]);start=i-1;}}segs.push([start,labels.length,labels.at(-1)]);
 const blocks=[[11185,90652],[110808,190198]];
 summary={schema_version:1,mode:'imported_csv',title:'Imported submission',model_status:'User-supplied CSV; model provenance not verified.',score_status:'Not scored.',n_frames:labels.length,fps:25,teams:{H:'Home team',A:'Away team'},blocks,audit:null,variants:[{name:'imported_csv',half_window_frames:0,browser_valid:true,format_valid:false,...analyse(labels,blocks)}]};timelines={imported_csv:segs};selected=summary.variants[0];custom=true;$('window').value='all';$('position').value=0;fillVariants();render();$('import-status').textContent='199,319 rows checked locally. Use the official Python validator before submitting.';
 }catch(error){$('import-status').textContent=`Could not load CSV: ${error.message}`;}finally{event.target.value='';}
}
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(t=>{const active=t===tab;t.classList.toggle('active',active);t.setAttribute('aria-pressed',String(active));});document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==tab.dataset.panel);draw();}));
$('variant').addEventListener('change',choose);for(const id of ['window','zoom','position'])$(id).addEventListener('input',()=>{draw();$('inspection').textContent='Move over the timeline to inspect a prediction.';});$('timeline').addEventListener('mousemove',inspect);$('timeline').addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();$('position').value=Math.max(0,Math.min(1000,Number($('position').value)+(event.key==='ArrowRight'?10:-10)));draw();}});$('csv-file').addEventListener('change',importCSV);window.addEventListener('resize',draw);
(window.__CONTROL_SWITCH_DATA__?Promise.resolve(window.__CONTROL_SWITCH_DATA__):Promise.all([fetch('/api/summary').then(r=>{if(!r.ok)throw Error('Summary unavailable');return r.json();}),fetch('/api/timeline').then(r=>{if(!r.ok)throw Error('Timeline unavailable');return r.json();})])).then(([s,t])=>{summary=s;timelines=t;selected=s.variants[1]||s.variants[0];fillVariants();render();}).catch(error=>{$('notice').textContent=`Could not load local data: ${error.message}. Restart the Python server and refresh.`;});
