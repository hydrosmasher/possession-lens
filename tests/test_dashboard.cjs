const assert=require('node:assert/strict');const Lens=require('../controlswitch/web/app.js');
const d={schema_version:2,mode:'synthetic_demo',teams:{H:'Blue',A:'Claret'},n_frames:10,fps:25,blocks:[[0,4],[6,10]],timeline:[[0,2,'H'],[2,4,'A'],[4,6,'X'],[6,8,'D'],[8,10,'H']],tracklets:null};
Lens.validate(d);assert.equal(Lens.stats(d).home,100*4/6);assert.equal(Lens.stats(d).spells,3);assert.equal(Lens.stats(d,'first').home,50);assert.equal(Lens.stats(d,'second').home,100);
assert.throws(()=>Lens.validate({...d,timeline:[[0,11,'H']]}));assert.throws(()=>Lens.validate({...d,blocks:[[3,6],[5,9]]}));
const stopped={...d,timeline:[[0,10,'D']]};assert.equal(Lens.stats(stopped).home,null);
assert.equal(Lens.csv('frame,label\n0,H\n1,D\n2,A\n').n_frames,3);assert.throws(()=>Lens.csv('frame,label\n0,H\n2,A\n'));
assert.equal(Lens.escape('<script>'), '&lt;script&gt;');console.log('Dashboard checks passed: period denominators, stops/unknowns, timeline validation, CSV import and escaping.');
const named={...d,mode:'external_named',named_players:{source_kind:'external_provider_stable_ids',rows:[{team:'H',person_id:'a',player_name:'Ada Example',associated_control_seconds:.08,percent_of_team_possession:50}]}};
Lens.validate(named);assert.throws(()=>Lens.validate({...named,mode:'licensed_local'}));assert.throws(()=>Lens.validate({...named,named_players:{...named.named_players,rows:[...named.named_players.rows,...named.named_players.rows]}}));
