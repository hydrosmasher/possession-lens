"""External-only withheld-observation audits, not human identity/contact truth."""
from pathlib import Path
import json,copy
import numpy as np
from .control_recovery import recover_ids,ownership,bridge_control
from .skillcorner_europe import load_match,digest
from .rnn_runtime import load_frozen,infer_frozen
TRAIN=['4039','3749','3518','2068','2440'];VALID=['2269']

def gap_mask(n,seed,fps=25):
 rng=np.random.default_rng(seed);mask=np.zeros(n,bool)
 while mask.mean()<.25:
  a=int(rng.integers(0,n));mask[a:min(n,a+int(rng.integers(max(2,fps//5),fps+1)))]=True
 return mask

def audit_ids(path,limits,seed):
 m=json.loads((path/'match_data.json').read_text());frames=json.loads((path/'structured_data.json').read_text());teams={p['trackable_object']:0 if p['team_id']==m['home_team']['id'] else 1 for p in m['players'] if p.get('trackable_object') is not None};mask=gap_mask(len(frames),seed,10);truth={}
 for i,frame in enumerate(frames):
  if mask[i]:
   for j,obj in enumerate(frame['data']):
    tid=obj.get('trackable_object')
    if tid in teams:truth[(i,j)]=tid;obj.pop('trackable_object');obj['group_name']='home team' if teams[tid]==0 else 'away team'
 result={}
 for limit in limits:
  trial=copy.deepcopy(frames);recover_ids(trial,teams,limit);found=[(trial[i]['data'][j].get('trackable_object'),tid) for (i,j),tid in truth.items()];recovered=[(p,t) for p,t in found if p is not None];correct=sum(p==t for p,t in recovered)
  result[limit]={'recovered':len(recovered),'correct':correct,'precision':correct/len(recovered) if recovered else None,'withheld':len(truth)}
 return result

def run(raw,model,out):
 raw=Path(raw);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 protocol={'train':TRAIN,'validation':VALID,'id_gaps_native_frames':[5,10,20],'ball_gaps_25hz_frames':[10,20,30],'minimum_proxy_precision':.995,'minimum_validation_recovered_frames':500,'selection':'highest recovered count on training with >=99.5% proxy precision; independent validation gate, no validation retuning','identity_rule':'same contiguous provider track_id and same roster ID on both sides; no motion identity guesses','control_rule':'same named owner anchors, same team, continuous player observations, uncontested <=2m interpolated ball corridor','competition_data_used':False,'viewed_test_matches_used_for_selection':False,'model_sha256':digest(model),'limitations':['Withheld provider IDs and proximity associations are surrogate references, not independent human truth.','Automatic provider identity mistakes can persist.','Synthetic missingness may differ from natural missingness.']}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');id_results={};ball_results={};state=load_frozen(model)
 for k,mid in enumerate(TRAIN+VALID):
  print('Auditing development match',mid,flush=True);path=raw/mid;id_results[mid]=audit_ids(path,protocol['id_gaps_native_frames'],42+k if mid in TRAIN else 202)
  m=json.loads((path/'match_data.json').read_text());n_named=sum(p.get('trackable_object') is not None for p in m['players']);parts,_=load_match(raw,mid);summary={v:{'correct':0,'recovered':0} for v in protocol['ball_gaps_25hz_frames']}
  for half,(s,pitch) in enumerate(parts):
   pred,_=infer_frozen(state,s,pitch);labels=np.array(['H','A','D'])[pred];truth=ownership(labels,s['xy'],s['ball'],s['teams']);mask=gap_mask(len(labels),900+k*2+half);ball=s['ball'].copy();ball[mask]=np.nan;gradable=mask&np.isfinite(s['ball']).all(1)
   for limit in summary:
    owners,added=bridge_control(labels,s['xy'],ball,s['teams'],n_named,limit);ix=added&gradable;summary[limit]['recovered']+=int(ix.sum());summary[limit]['correct']+=int(np.sum(ix&(owners==truth)&(truth>=0)&(truth<n_named)))
  ball_results[mid]=summary
  (out/'progress.json').write_text(json.dumps({'identity':id_results,'ball':ball_results},indent=2)+'\n')
 def select(results,limits):
  trials=[]
  for limit in limits:
   r=sum(results[mid][limit]['recovered'] for mid in TRAIN);c=sum(results[mid][limit]['correct'] for mid in TRAIN);trials.append({'gap':limit,'recovered':r,'correct':c,'precision':c/r if r else 0})
  eligible=[t for t in trials if t['recovered']>=500 and t['precision']>=.995];chosen=max(eligible,key=lambda x:x['recovered']) if eligible else None
  validation=None;enabled=False
  if chosen:
   v=[results[mid][chosen['gap']] for mid in VALID];r=sum(x['recovered'] for x in v);c=sum(x['correct'] for x in v);validation={'recovered':r,'correct':c,'precision':c/r if r else 0};enabled=r>=500 and validation['precision']>=.995
  return {'training_trials':trials,'chosen':chosen,'validation':validation,'enabled':enabled}
 report={'protocol':protocol,'identity':select(id_results,protocol['id_gaps_native_frames']),'ball':select(ball_results,protocol['ball_gaps_25hz_frames']),'details':{'identity':id_results,'ball':ball_results},'fresh_test_results':None}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('identity','ball')},indent=2));return report
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('raw','model','out'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();run(a.raw,a.model,a.out)
