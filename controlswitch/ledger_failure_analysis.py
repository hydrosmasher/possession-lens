from pathlib import Path
import json,numpy as np,torch
from controlswitch.ownership_ledger import load_actor,association
from controlswitch.player_set_state import player_nodes
from controlswitch.external import period_arrays
from controlswitch.tracker_robustness import CONDITIONS,condition
root=Path.cwd();out=root/'local/ownership-ledger-062';actor_report=json.loads((root/'local/actor-061/report.json').read_text());actor=load_actor(root/'local/actor-061/actor-encoder.npz',actor_report['encoder_sha256']);torch.set_num_threads(1);groups={}
for fi,mid in enumerate(('J03WN1','J03WMX')):
 with np.load(next(Path('/Users/hydrosmasher/Desktop/Possession Lens/data/external/cache').glob(mid+'-*.npz')),allow_pickle=False) as a:record={k:a[k] for k in a.files}
 for half in (0,1):
  seq=period_arrays(record,half)
  for ci,c in enumerate(CONDITIONS):
   s,mask=condition(seq,c,half,False);nodes=player_nodes(s,record['pitch'],mask);_,r=association(nodes,actor);seen=nodes[...,10].max((1,2))>.5
   with np.load(out/f'case-{fi}-{half*4+ci}.npz',allow_pickle=False) as a:y=a['reference'];p=a['actor_ledger'];old=a['current']
   masks={'all_reference_live':y<2,'new_wrong_team':(y<2)&(p<2)&(p!=y)&(old==y),'new_recovered_team':(y<2)&(p==y)&(old<2)&(old!=y)}
   for group,m in masks.items():
    for observed in (False,True):
     take=m&(seen==observed);key=group+('/visible_ball' if observed else '/missing_ball');g=groups.setdefault(key,{'frames':0,'max_support_sum':0.,'correct_team_support_sum':0.,'wrong_team_support_sum':0.});ix=np.flatnonzero(take);g['frames']+=len(ix)
     if len(ix):g['max_support_sum']+=float(r[ix].max(1).sum());g['correct_team_support_sum']+=float(r[ix,y[ix]].sum());g['wrong_team_support_sum']+=float(r[ix,1-y[ix]].sum())
   print('diagnosed',fi,half,c,flush=True)
for g in groups.values():
 for k in list(g):
  if k.endswith('_sum'):g[k[:-4]+'_mean']=g.pop(k)/max(g['frames'],1)
report={'scope':'External frame-level diagnostic of rejected062. Reference labels used only for failure analysis, not runtime inputs. Repeated corruption draws on two reused games; counts are not unique-match totals.','groups':groups,'promoted':False}
(out/'failure-analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
