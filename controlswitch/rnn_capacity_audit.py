"""Retrospective regression on already-viewed external tests. Never a fresh test."""
from pathlib import Path
import json,argparse
import numpy as np
from .external import period_arrays
from .rnn_benchmark import conditions,evaluate
from .rnn_runtime import load_frozen,infer_frozen
from .rnn_boundaries import spell_diagnostics
from .experiment import baseline_prediction
from .pipeline import digest

def run(cache_root,candidate,reference,out):
 out=Path(out)
 if out.exists():raise ValueError('Choose a new audit folder')
 out.mkdir(parents=True);paths={'capacity_candidate':Path(candidate),'current_default':Path(reference)};frozen={k:load_frozen(v) for k,v in paths.items()}
 protocol={'status':'retrospective_previously_viewed_match_regression','matches':['J03WMX','J03WN1','J03WQQ'],'checkpoint_sha256':{k:digest(v) for k,v in paths.items()},'selection_on_audit_matches':False,'competition_data_used':False,'warning':'These matches were viewed during earlier possession development. This is not independent generalisation evidence. Candidate remains experimental regardless of this audit.'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results={}
 for mid in protocol['matches']:
  cache=sorted(Path(cache_root).glob(mid+'-*.npz'))
  if len(cache)!=1:raise ValueError('Expected exactly one external regression cache')
  with np.load(cache[0],allow_pickle=False) as d:data={k:d[k] for k in d.files}
  if str(data['match_id'])!=mid:raise ValueError('Regression cache identity mismatch')
  rows=conditions([(period_arrays(data,i),data['pitch']) for i in (0,1)],300);scores={};diagnostics={}
  for name,state in frozen.items():
   pred=[infer_frozen(state,s,p,mask)[0] for s,p,c,mask in rows];scores[name]=evaluate(rows,pred);diagnostics[name]=spell_diagnostics(rows,pred)
  scores['baseline_smooth50']=evaluate(rows,[baseline_prediction(s,50,mask) for s,p,c,mask in rows])
  for row in scores.values():row['mean_clean_masked_spell_f1']=row.pop('selection_spell_f1')
  results[mid]={'scores':scores,'diagnostics':diagnostics,'cache_sha256':digest(cache[0])};(out/'report.json').write_text(json.dumps({'protocol':protocol,'results':results},indent=2)+'\n');print('Retrospective external match completed:',mid,flush=True)
 return {'protocol':protocol,'results':results}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('cache-root','candidate','reference','out'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();run(a.cache_root,a.candidate,a.reference,a.out)
if __name__=='__main__':main()
