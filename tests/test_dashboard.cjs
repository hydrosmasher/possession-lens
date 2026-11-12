const assert=require('node:assert/strict');const Lens=require('../controlswitch/web/app.js');
const d={schema_version:2,mode:'synthetic_demo',teams:{H:'Blue',A:'Claret'},n_frames:10,fps:25,blocks:[[0,4],[6,10]],timeline:[[0,2,'H'],[2,4,'A'],[4,6,'X'],[6,8,'D'],[8,10,'H']],tracklets:null};
Lens.validate(d);assert.equal(Lens.stats(d).home,100*4/6);assert.equal(Lens.stats(d).spells,3);assert.equal(Lens.stats(d,'first').home,50);assert.equal(Lens.stats(d,'second').home,100);
assert.throws(()=>Lens.validate({...d,timeline:[[0,11,'H']]}));assert.throws(()=>Lens.validate({...d,blocks:[[3,6],[5,9]]}));
const stopped={...d,timeline:[[0,10,'D']]};assert.equal(Lens.stats(stopped).home,null);
assert.equal(Lens.csv('frame,label\n0,H\n1,D\n2,A\n').n_frames,3);assert.throws(()=>Lens.csv('frame,label\n0,H\n2,A\n'));
assert.equal(Lens.escape('<script>'), '&lt;script&gt;');console.log('Dashboard checks passed: period denominators, stops/unknowns, timeline validation, CSV import and escaping.');
const named={...d,mode:'external_named',named_players:{source_kind:'external_provider_stable_ids',rows:[{team:'H',person_id:'a',player_name:'Ada Example',associated_control_seconds:.08,percent_of_team_possession:50}]}};
Lens.validate(named);assert.throws(()=>Lens.validate({...named,mode:'licensed_local'}));assert.throws(()=>Lens.validate({...named,named_players:{...named.named_players,rows:[...named.named_players.rows,...named.named_players.rows]}}));
const contributions={...named,named_players:{...named.named_players,rows:[{team:'H',person_id:'a',player_name:'Ada Example',associated_control_seconds:3,percent_of_team_possession:30},{team:'H',person_id:'b',player_name:'Ben Example',associated_control_seconds:5,percent_of_team_possession:50},{team:'H',person_id:'UNASSIGNED',player_name:'Unassigned control',associated_control_seconds:2,percent_of_team_possession:20},{team:'A',person_id:'c',player_name:'Ada Away',associated_control_seconds:4,percent_of_team_possession:100}]}};
assert.deepEqual(Lens.playerRows(contributions,'H','ADA').map(r=>r.person_id),['a','UNASSIGNED']);assert.deepEqual(Lens.playerRows(contributions,'H','missing').map(r=>r.person_id),['UNASSIGNED']);assert.deepEqual(Lens.playerRows(contributions,'H').map(r=>r.person_id),['b','a','UNASSIGNED']);
console.log('Player search passed: names, team separation, sorting and unassigned coverage.');
const fs=require('node:fs');const path=require('node:path');
const lib=JSON.parse(fs.readFileSync(path.join(__dirname,'../examples/match-library.json'),'utf8'));
assert.equal(lib.matches.length,10);for(const match of lib.matches){Lens.validate(match);assert.equal(match.mode,'external_named');assert.ok(match.named_players.rows.filter(p=>p.person_id!=='UNASSIGNED').every(p=>p.player_name.trim()));}
console.log('All ten match-library payloads passed timeline and provider-name validation.');
// Clipping a possession at a window/half boundary must not count time outside it.
const clip={fps:25,blocks:[[0,500],[750,1000]],timeline:[[0,250,'H'],[250,500,'A'],[500,750,'X'],[750,1000,'H']]};
assert.equal(Lens.windowStats(clip,125,375).counts.H,125);
assert.equal(Lens.windowStats(clip,125,375).counts.A,125);
assert.equal(Lens.controlWindows(clip,'all',10).length,3);
assert.equal(Lens.controlWindows(clip,'second',10)[0].start,750);
assert.equal(Lens.windowStats(clip,500,750).home,null);
assert.equal(Lens.spellProfile(clip,[[125,375]])[1].H,1);
assert.equal(Lens.spellProfile(clip,[[125,375]])[1].A,1);
console.log('Control explorer checks passed: clipped windows, half boundaries, unknown-only intervals and spell buckets.');
