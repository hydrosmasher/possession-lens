// DOM-level interaction checks; this does not replace visual browser QA.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const library=JSON.parse(fs.readFileSync(path.join(__dirname,'../examples/match-library.json'),'utf8'));
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.handlers={};this.style={};this.dataset={};this.value='';this.textContent='';this.classList={toggle(){}};}
 append(...children){for(const c of children){c.parentElement=this;this.children.push(c);}}
 replaceChildren(...children){this.children=[];this.append(...children);}
 setAttribute(k,v){this[k]=v;}
 addEventListener(event,fn){this.handlers[event]=fn;}
 querySelector(){return new Element();}
 focus(){}
}
const elements=new Map(),get=id=>{if(!elements.has(id)){const e=new Element();e.parentElement=new Element();elements.set(id,e);}return elements.get(id);};
const tabs=['overview','players','evidence'].map(panel=>{const e=new Element('button');e.dataset.panel=panel;return e;});
const document={getElementById:get,createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),querySelectorAll:s=>s==='.tab'?tabs:s==='.panel'?['overview','players','evidence'].map(get):[]};
const source=fs.readFileSync(path.join(__dirname,'../controlswitch/web/app.js'),'utf8');
const initial=library.matches.find(m=>m.example_id==='skillcorner-2841');
const context={document,window:{__POSSESSION_LENS_DATA__:initial,__POSSESSION_LENS_LIBRARY__:library},console,Promise,setTimeout,URL,Blob};
vm.runInNewContext(source,context);
(async()=>{
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(get('example-match').children.length,10);
 for(const match of library.matches){
  get('example-match').value=match.example_id;get('example-match').handlers.change();
  assert.equal(get('home-name').textContent,match.teams.H);assert.equal(get('away-name').textContent,match.teams.A);
  assert.equal(get('match-title').textContent,match.title);
  for(const team of ['H','A']){
   get(team==='H'?'players-home':'players-away').handlers.click();
   const names=new Set(match.named_players.rows.filter(r=>r.team===team&&r.person_id!=='UNASSIGNED').map(r=>r.player_name));
   assert.ok(get('player-rows').children.length>0);
   for(const row of get('player-rows').children)assert.ok(names.has(row.children[1].textContent),'only provider-mapped names may be displayed');
  }
  if(match.dead_ball_supported===false)assert.equal(get('dead-time').textContent,'Not classified');
 }
 const named=library.matches[0];get('example-match').value=named.example_id;get('example-match').handlers.change();get('player-search').value='NO_MATCH_999';get('player-search').handlers.input();assert.equal(get('player-rows').children.length,0);
 console.log('Gallery UI checks passed: all ten match switches, both teams, named-only rows, search and unsupported dead-ball display.');
})().catch(error=>{console.error(error);process.exitCode=1;});
const Lens=require('../controlswitch/web/app.js');const verified=library.matches.find(m=>m.example_id==='sportec-J03WN1');assert.equal(Lens.eventStats(verified).H.shots,10);assert.equal(Lens.eventStats(verified).A.shots,11);assert.equal(Lens.eventStats(verified).H.touches,null);assert.equal(Lens.eventStats(initial),null);assert.equal(Lens.eventStats(verified,'first').H.shots+Lens.eventStats(verified,'second').H.shots,10);
