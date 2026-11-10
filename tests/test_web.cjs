// Code-level interaction smoke test with a minimal DOM double; not a visual browser test.
const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
class Element {
 constructor(id=''){this.id=id;this.textContent='';this.children=[];this.style={};this.listeners={};this.value='';this.hidden=false;this.dataset={};this.classList={toggle(){}};this.parentElement={clientWidth:1000};this.clientWidth=1000;}
 append(...elements){this.children.push(...elements);}
 replaceChildren(...elements){this.children=elements;}
 addEventListener(event,fn){this.listeners[event]=fn;}
 setAttribute(name,value){this[name]=value;}
 querySelector(){return this.second||(this.second=new Element());}
 getContext(){return {scale(){},fillRect(){},fillText(){},beginPath(){},moveTo(){},lineTo(){},stroke(){}};}
 getBoundingClientRect(){return {left:0};}
}
async function main(){
 const html=fs.readFileSync('outputs/control-switch-demo.html','utf8');
 const data=JSON.parse(html.match(/window\.__CONTROL_SWITCH_DATA__=(.*?);<\/script>/s)[1]);
 const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
 get('window').value='all';get('zoom').value='0';get('position').value='0';
 const tabs=['explore','method'].map(panel=>{const e=new Element();e.dataset.panel=panel;return e;});
 const panels=['explore','method'].map(id=>get(id));
 const context={document:{getElementById:get,createElement:()=>new Element(),querySelectorAll:q=>q==='.tab'?tabs:panels},window:{__CONTROL_SWITCH_DATA__:data,devicePixelRatio:1,addEventListener(){}},fetch(){throw Error('Standalone demo attempted a network request');},console};
 vm.createContext(context);vm.runInContext(fs.readFileSync('controlswitch/web/app.js','utf8'),context);
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(get('mode').textContent,'SYNTHETIC DEMO');
 assert.equal(get('comparison').children.length,4);
 assert.equal(get('variant').children.length,4);
 get('variant').value='baseline_raw';get('variant').listeners.change();
 assert.match(get('window-note').textContent,/no temporal smoothing/);
 tabs[1].listeners.click();assert.equal(get('explore').hidden,true);assert.equal(get('method').hidden,false);
 const event=text=>({target:{files:[{size:text.length,text:async()=>text}],value:'csv'}});
 await get('csv-file').listeners.change(event('frame,label\n0,H\n'));
 assert.match(get('import-status').textContent,/Expected 199,319 rows/);
 const csv='frame,label\n'+Array.from({length:199319},(_,i)=>`${i},${i<100000?'H':'A'}`).join('\n')+'\n';
 await get('csv-file').listeners.change(event(csv));
 assert.equal(get('mode').textContent,'LOCAL CSV INSPECTION');
 assert.equal(get('frames').textContent,'199,319');
 assert.equal(get('comparison').children.length,1);
 assert.match(get('import-status').textContent,/checked locally/);
 console.log('Web smoke checks passed: synthetic initialization, variant selection, navigation, invalid CSV, full-length CSV import, no network requests. Visual browser QA not performed.');
}
main().catch(e=>{console.error(e);process.exit(1);});
