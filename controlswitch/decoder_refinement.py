"""Validation-only decoder refinement. Never opens viewed test matches."""
from pathlib import Path
import json,argparse
import numpy as np
from .skillcorner_europe import load_match,digest
from .rnn_runtime import load_frozen,infer_frozen
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import spell_diagnostics
from .europe_recovery import accepted,totals
from .model import decode

def run(raw,model,anchor,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if (out/'report.json').exists():raise ValueError('Use a new experiment folder')
 protocol={'validation_only':['2269'],'new_test_match':None,'viewed_tests_loaded':False,'penalties':[18,24,30,36,42,48,54],'selection':'max validation mean spell F1, pooled misses and merge overlaps <= context reference, false positives and short pulses constrained against original European anchor','model_sha256':digest(model),'anchor_sha256':digest(anchor),'fresh_evaluation_status':'blocked_pending_new_licensed_top_five_tracking_and_labels','competition_data_used':False}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 parts,_=load_match(raw,'2269');rows=conditions(parts,100);state=load_frozen(model);ref=[];probs=[]
 for s,p,c,mask in rows:
  y,pb=infer_frozen(state,s,p,mask);ref.append(y);probs.append(pb)
 reference=evaluate(rows,ref);rd=spell_diagnostics(rows,ref);a=load_frozen(anchor);ad=spell_diagnostics(rows,[infer_frozen(a,s,p,mask)[0] for s,p,c,mask in rows]);trials=[]
 for penalty in protocol['penalties']:
  pred=[decode(p,penalty) for p in probs];score=evaluate(rows,pred);d=spell_diagnostics(rows,pred);current=totals(d);old=totals(rd)
  eligible=accepted(d,ad) and current['missed_reference_spells_at_iou_0_5']<=old['missed_reference_spells_at_iou_0_5'] and current['predictions_overlapping_multiple_reference_spells']<=old['predictions_overlapping_multiple_reference_spells']
  trials.append({'penalty':penalty,**score,'diagnostics':d,'eligible':eligible})
 eligible=[t for t in trials if t['eligible']];best=max(eligible,key=lambda x:x['selection_spell_f1']) if eligible else None;improved=best is not None and best['selection_spell_f1']>reference['selection_spell_f1']
 if improved:
  with np.load(model,allow_pickle=False) as d:weights={k:d[k] for k in d.files}
  weights['transition_penalty']=np.array(float(best['penalty']));np.savez_compressed(out/'validation-only-candidate.npz',**weights)
 report={'protocol':protocol,'reference_validation':reference,'reference_diagnostics':rd,'trials':trials,'selected':best,'improved_validation':improved,'promoted_to_default':False,'fresh_test_results':None,'limitations':['Repeatedly reused validation match; results can overfit it.','No fresh licensed holdout is available; no independent improvement claim.','Partial automatic H/A labels cannot establish complete-match or official challenge accuracy.']}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'improved_validation':improved,'reference_F1':reference['selection_spell_f1'],'selected_F1':None if best is None else best['selection_spell_f1'],'selected_penalty':None if best is None else best['penalty']}));return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['raw','model','anchor','out']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();run(a.raw,a.model,a.anchor,a.out)
