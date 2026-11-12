"""Expanded-data instantaneous vs multiscale-context RNN; external data only."""
from pathlib import Path
import argparse,json,shutil,gc
import numpy as np
from . import rnn_deep as net
from .model import instantaneous_features,features,decode
from .skillcorner_europe import load_match,digest
from .context_data import SPLIT
from .rnn_runtime import load_frozen,infer_frozen,CONTEXT_VERSION
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import spell_diagnostics
from .europe_recovery import accepted
from .experiment import missing_mask,baseline_prediction

PENALTIES=(3,12,24,48)

def observation_features(s,pitch,mask=None,swap=False,context=False):
 return features(s,pitch,swap=swap,ball_missing=mask) if context else instantaneous_features(s,pitch,swap=swap,ball_missing=mask)

def observation_probabilities(model,s,pitch,mask=None,context=False):
 a=net.predict(model,observation_features(s,pitch,mask,False,context));b=net.predict(model,observation_features(s,pitch,mask,True,context))
 return (a+b[:,[1,0,2]])*.5

def save_model(path,model,penalty,context):
 np.savez_compressed(path,**model,version=np.array(CONTEXT_VERSION if context else net.VERSION),transition_penalty=np.array(float(penalty)),symmetric=np.array(True))

def run(root,out,reference,production,iterations=16):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if (out/'freeze-record.json').exists():raise ValueError('Frozen experiment: never refit against a viewed test')
 protocol=json.loads((out/'preregistered-protocol.json').read_text())
 if protocol['train']!=SPLIT['train'] or protocol['validation']!=SPLIT['validation'] or protocol['sealed_test']!=SPLIT['test'][0]:raise ValueError('Incorrect fixed split')
 protocol.update(maximum_iterations=iterations,reference_sha256=digest(reference),production_sha256=digest(production),test_ablations='Evaluate both validation-selected frozen families and comparators; never choose using test scores',training_augmentation='home/away swap × clean/fixed contiguous ball dropout; four sequences per half')
 (out/'run-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 train=[];audits={}
 for mid in SPLIT['train']:
  parts,audits[mid]=load_match(root,mid);train.extend(parts)
 validation,audits[SPLIT['validation'][0]]=load_match(root,SPLIT['validation'][0])
 (out/'development-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
 rows=conditions(validation,100);ref=load_frozen(reference);refpred=[infer_frozen(ref,s,p,mask)[0] for s,p,c,mask in rows];refscore=evaluate(rows,refpred);refdiag=spell_diagnostics(rows,refpred)
 families={};trials=[]
 for context in (False,True):
  family='context' if context else 'instantaneous';sequences=[];history=[];best_any=None;best_eligible=None
  print('Training expanded-data family:',family,flush=True)
  for i,(s,pitch) in enumerate(train):
   for swap in (False,True):
    y=s['y'].copy()
    if swap:y=np.where(y==0,1,np.where(y==1,0,y))
    y=np.where(y<2,y,-1)
    for masked in (False,True):sequences.append((observation_features(s,pitch,missing_mask(len(y),42+i) if masked else None,swap,context),y.copy()))
  val=[(observation_features(s,p,None,False,context),np.where(s['y']<2,s['y'],-1)) for s,p in validation]
  def progress(row,model):
   nonlocal best_any,best_eligible
   history.append(row);probs=[observation_probabilities(model,s,p,mask,context) for s,p,c,mask in rows]
   for penalty in PENALTIES:
    preds=[decode(p,penalty) for p in probs];score=evaluate(rows,preds);diag=spell_diagnostics(rows,preds);eligible=accepted(diag,refdiag)
    trial={'family':family,'context':context,'iteration':row['iteration'],'penalty':penalty,'passes_false_positive_guard':eligible,**score,'diagnostics':diag};trials.append(trial)
    if best_any is None or score['selection_spell_f1']>best_any['selection_spell_f1']:
     best_any=trial;save_model(out/f'{family}-unconstrained.npz',model,penalty,context)
    if eligible and (best_eligible is None or score['selection_spell_f1']>best_eligible['selection_spell_f1']):
     best_eligible=trial;save_model(out/f'{family}-eligible.npz',model,penalty,context)
   (out/f'{family}-learning.json').write_text(json.dumps(history,indent=2)+'\n');(out/'validation-trials.json').write_text(json.dumps(trials,indent=2)+'\n')
   print('Best validation',family,json.dumps({'any':best_any['selection_spell_f1'],'eligible':None if best_eligible is None else best_eligible['selection_spell_f1']}),flush=True)
  net.train(sequences,val,(24,),epochs=iterations,patience=iterations,on_epoch_model=progress)
  selected=best_eligible if best_eligible is not None else best_any
  shutil.copy2(out/(f'{family}-'+('eligible.npz' if best_eligible is not None else 'unconstrained.npz')),out/f'{family}-selected.npz')
  families[family]=selected;del sequences,val;gc.collect()
 # Prefer an eligible candidate; otherwise freeze an explicitly rejected diagnostic candidate.
 eligible=[r for r in families.values() if r['passes_false_positive_guard']]
 chosen=max(eligible or list(families.values()),key=lambda r:r['selection_spell_f1']);shutil.copy2(out/f'{chosen["family"]}-selected.npz',out/'candidate-model.npz')
 baselines=[{'window':w,**evaluate(rows,[baseline_prediction(s,w,mask) for s,p,c,mask in rows])} for w in (0,25,50,125)];baseline=max(baselines,key=lambda x:x['selection_spell_f1'])
 sources=[Path(__file__),Path(__file__).with_name('rnn_deep.py'),Path(__file__).with_name('rnn_runtime.py'),Path(__file__).with_name('model.py'),Path(__file__).with_name('skillcorner_europe.py')]
 frozen={'family':chosen['family'],'iteration':chosen['iteration'],'penalty':chosen['penalty'],'passes_false_positive_guard':chosen['passes_false_positive_guard'],'validation_improved':chosen['selection_spell_f1']>refscore['selection_spell_f1'],'candidate_sha256':digest(out/'candidate-model.npz'),'families':{k:{'sha256':digest(out/f'{k}-selected.npz'),'iteration':v['iteration'],'penalty':v['penalty'],'eligible':v['passes_false_positive_guard']} for k,v in families.items()},'reference_sha256':digest(reference),'production_sha256':digest(production),'baseline_window':baseline['window'],'source_sha256':{p.name:digest(p) for p in sources},'test_labels_read_at_freeze':False}
 (out/'freeze-record.json').write_text(json.dumps(frozen,indent=2)+'\n')
 print('Frozen: opening reserved Barcelona–Real Madrid 2841 once.',flush=True)
 test,test_audit=load_match(root,SPLIT['test'][0]);rows=conditions(test,500);results={};diagnostics={}
 paths={'candidate':out/'candidate-model.npz','expanded_instantaneous':out/'instantaneous-selected.npz','expanded_context':out/'context-selected.npz','previous_european':reference,'production_current':production}
 for name,path in paths.items():
  state=load_frozen(path);pred=[infer_frozen(state,s,p,mask)[0] for s,p,c,mask in rows];results[name]=evaluate(rows,pred);diagnostics[name]=spell_diagnostics(rows,pred)
 pred=[baseline_prediction(s,baseline['window'],mask) for s,p,c,mask in rows];results['baseline']=evaluate(rows,pred);diagnostics['baseline']=spell_diagnostics(rows,pred)
 report={'protocol':protocol,'frozen':frozen,'validation_reference':refscore,'validation_reference_diagnostics':refdiag,'validation_selected':chosen,'validation_families':families,'test_audit':test_audit,'test_results':results,'test_diagnostics':diagnostics,'candidate_beats_previous_european':results['candidate']['selection_spell_f1']>results['previous_european']['selection_spell_f1'],'promoted_to_default':False,'test_status':'viewed_once_after_freeze','limitations':protocol['limitations']+['Unknown labels split/censor spells; no dead-ball supervision.','A different match score is not a cross-experiment F1 gain; compare models only on this same test.','Test ablations are diagnostic and must not be used to change the validation-selected candidate.','One fresh match is limited evidence; no statistical-significance or 0.8–0.9 guarantee.']}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('Test F1:',json.dumps({k:v['selection_spell_f1'] for k,v in results.items()}),flush=True);return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('root','out','reference','production'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--iterations',type=int,default=16);a=p.parse_args();run(a.root,a.out,a.reference,a.production,a.iterations)
