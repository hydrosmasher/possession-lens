"""Validation-only recurrent interaction experiment, no viewed test loading."""
from pathlib import Path
import json,argparse
import numpy as np
from .model import features,interaction_features,decode
from . import rnn_deep as net
from .rnn_runtime import load_frozen,infer_frozen,INTERACTION_CONTEXT_VERSION
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import spell_diagnostics
from .europe_recovery import accepted,totals
from .skillcorner_europe import load_match,digest
from .experiment import missing_mask
TRAIN=['4039','3749','3518','2068','2440']

def observations(s,pitch,mask=None,swap=False):return np.column_stack([features(s,pitch,swap=swap,ball_missing=mask),interaction_features(s,swap=swap,ball_missing=mask)]).astype(np.float32)
def observation_probabilities(model,s,pitch,mask=None):
 a=net.predict(model,observations(s,pitch,mask));b=net.predict(model,observations(s,pitch,mask,True));return (a+b[:,[1,0,2]])*.5

def run(raw,reference,anchor,out,iterations=12):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if (out/'report.json').exists():raise ValueError('Use a new experiment folder')
 protocol={'train':TRAIN,'validation':['2269'],'viewed_tests_loaded':False,'new_test_results':None,'features':'120 offline multiscale geometry + 10 causal team-pooled proximity/trajectory/movement interactions','architecture':'one 24-unit tanh recurrent layer, from scratch','max_sampled_iterations':iterations,'windows_per_iteration':1024,'seed':42,'dropout':.15,'l2':.0001,'penalties':[24,36,48,60],'selection':'highest validation clean/dropout spell F1 with misses and geometric merges <= context reference, false-switch guard anchored to original European RNN','reference_sha256':digest(reference),'anchor_sha256':digest(anchor),'competition_data_used':False}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 train=[]
 for mid in TRAIN:parts,_=load_match(raw,mid);train.extend(parts)
 validation,_=load_match(raw,'2269');rows=conditions(validation,100);ref=load_frozen(reference);refpred=[infer_frozen(ref,s,p,mask)[0] for s,p,c,mask in rows];rd=spell_diagnostics(rows,refpred);rs=evaluate(rows,refpred);a=load_frozen(anchor);ad=spell_diagnostics(rows,[infer_frozen(a,s,p,mask)[0] for s,p,c,mask in rows]);seq=[]
 for i,(s,pitch) in enumerate(train):
  for swap in (False,True):
   y=s['y'].copy()
   if swap:y=np.where(y==0,1,np.where(y==1,0,y))
   y=np.where(y<2,y,-1)
   for masked in (False,True):seq.append((observations(s,pitch,missing_mask(len(y),42+i) if masked else None,swap),y.copy()))
 val=[(observations(s,p),np.where(s['y']<2,s['y'],-1)) for s,p in validation];trials=[];history=[];best=None
 def progress(row,model):
  nonlocal best
  history.append(row);probs=[observation_probabilities(model,s,p,mask) for s,p,c,mask in rows]
  for penalty in protocol['penalties']:
   pred=[decode(p,penalty) for p in probs];score=evaluate(rows,pred);diag=spell_diagnostics(rows,pred);t=totals(diag);r=totals(rd);ok=accepted(diag,ad) and t['missed_reference_spells_at_iou_0_5']<=r['missed_reference_spells_at_iou_0_5'] and t['predictions_overlapping_multiple_reference_spells']<=r['predictions_overlapping_multiple_reference_spells']
   trial={'iteration':row['iteration'],'penalty':penalty,'eligible':ok,**score,'diagnostics':diag};trials.append(trial)
   if ok and (best is None or score['selection_spell_f1']>best['selection_spell_f1']):
    best=trial;np.savez_compressed(out/'candidate.npz',**model,version=np.array(INTERACTION_CONTEXT_VERSION),transition_penalty=np.array(float(penalty)),symmetric=np.array(True))
  (out/'learning.json').write_text(json.dumps(history,indent=2)+'\n');(out/'validation-trials.json').write_text(json.dumps(trials,indent=2)+'\n');print('Best eligible validation:',None if best is None else best['selection_spell_f1'],flush=True)
 net.train(seq,val,(24,),epochs=iterations,patience=iterations,on_epoch_model=progress)
 report={'protocol':protocol,'reference_validation':rs,'reference_diagnostics':rd,'selected':best,'improved_validation':best is not None and best['selection_spell_f1']>rs['selection_spell_f1'],'fresh_test_results':None,'promoted_to_default':False,'limitations':['Repeated validation can overfit PSG–Marseille.','No new independent test or 0.8 score claim.','Provider partial H/A labels do not validate dead ball.']}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'improved_validation':report['improved_validation'],'reference_F1':rs['selection_spell_f1'],'selected_F1':None if best is None else best['selection_spell_f1']}),flush=True);return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('raw','reference','anchor','out'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--iterations',type=int,default=12);a=p.parse_args();run(a.raw,a.reference,a.anchor,a.out,a.iterations)
