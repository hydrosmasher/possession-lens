"""External-only bounded depth/training comparison. Never reads challenge data."""
from pathlib import Path
import json,argparse
import numpy as np
from . import rnn_deep as net
from .rnn_refine import read_cache
from .rnn_benchmark import augmented_sequences,conditions,evaluate
from .model import instantaneous_features,decode
from .rnn_runtime import load_frozen,infer_frozen
from .experiment import baseline_prediction
from .pipeline import digest

def observation_probabilities(model,s,pitch,mask=None):
 a=net.predict(model,instantaneous_features(s,pitch,ball_missing=mask));b=net.predict(model,instantaneous_features(s,pitch,swap=True,ball_missing=mask))
 return (a+b[:,[1,0,2]])*.5

def run(training_cache,validation_cache,reference,out,max_iterations=32,probe_mode="clean"):
 out=Path(out)
 if out.exists():raise ValueError('Choose a new experiment folder')
 out.mkdir(parents=True)
 protocol={'training_match':'J03WOH','validation_match':'J03WPY','depths':[1,2,3],'hidden_units_per_layer':24,'maximum_sampled_iterations':max_iterations,'windows_per_iteration':1024,'window_frames':128,'batch_size':16,'seed':42,'dropout':.15,'l2':.0001,'gradient_clip_norm':5,'early_stop_patience':3,'minimum_cross_entropy_improvement':.0001,'checkpoint_selection':'minimum fixed validation-window cross entropy per architecture','architecture_selection':'mean clean/masked full validation spell F1 with fixed decoder penalty 24; tie prefers fewer layers','dropout_probe_seed':101,'train_probe_windows':64,'validation_probe_windows':128,'test_data_loaded':False,'competition_data_used':False,'training_windows_reset_state':True,'loss_probes':probe_mode+' training vs clean validation; use clean mode for comparable diagnostics','inference_carries_state':True,'note':'An iteration samples up to 1024 windows, not a full epoch. J03WPY is repeatedly reused validation; independent testing is still required.','source_sha256':{'training_cache':digest(training_cache),'validation_cache':digest(validation_cache),'reference_checkpoint':digest(reference)}}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 training=read_cache(training_cache,'J03WOH');validation=read_cache(validation_cache,'J03WPY');sequences=augmented_sequences(training);val=[(instantaneous_features(s,p),s['y']) for s,p in validation];rows=conditions(validation,100)
 trials=[];histories={};states={}
 for depth in protocol['depths']:
  name=f'{depth}-layer';log=[]
  def progress(row):
   log.append(row);(out/f'{name}-learning.json').write_text(json.dumps(log,indent=2)+'\n')
  model,history,best_epoch=net.train(sequences,val,(24,)*depth,epochs=max_iterations,on_epoch=progress,train_probe_sequences=[(instantaneous_features(s,p),s['y']) for s,p in training] if probe_mode=='clean' else None)
  histories[name]=history;states[name]=model
  np.savez_compressed(out/f'{name}-model.npz',**model,version=np.array(net.VERSION),transition_penalty=np.array(24.),symmetric=np.array(True))
  probs=[observation_probabilities(model,s,p,mask) for s,p,c,mask in rows]
  trial={'model':name,'layers':depth,'parameter_count':sum(model[k].size for k in net.parameters(model)),'selected_iteration':best_epoch,'iterations_run':len(history),'early_stopping_triggered':len(history)<max_iterations,'best_probe_validation_ce':history[best_epoch-1]['validation_probe_cross_entropy'],'best_probe_train_ce':history[best_epoch-1]['train_probe_cross_entropy'],**evaluate(rows,[decode(p,24) for p in probs])};trials.append(trial)
  (out/'validation-trials.json').write_text(json.dumps(trials,indent=2)+'\n')
 reference_state=load_frozen(reference);ref=evaluate(rows,[infer_frozen(reference_state,s,p,mask)[0] for s,p,c,mask in rows]);baseline=evaluate(rows,[baseline_prediction(s,50,mask) for s,p,c,mask in rows]);chosen=max(trials,key=lambda r:(r['selection_spell_f1'],-r['layers']));model=states[chosen['model']]
 np.savez_compressed(out/'candidate-model.npz',**model,version=np.array(net.VERSION),transition_penalty=np.array(24.),symmetric=np.array(True))
 report={'protocol':protocol,'validation_trials':trials,'selected_candidate':chosen,'reference_validation':ref,'baseline_validation':baseline,'learning_curves':histories,'candidate_sha256':digest(out/'candidate-model.npz'),'promoted_to_default':False,'limitations':['Repeated validation selection can overfit J03WPY even with early stopping.','Probe loss is measured on fixed sampled reset-state windows, not full-match accuracy.','Full inference carries state beyond the training window; full validation segmentation is separately reported.','A sampled iteration is not a full pass over the training match.','No unseen match is currently available; do not claim independent test improvement.','Extra layers are not guaranteed to help; keep the default frozen model pending evidence.']}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('training-cache','validation-cache','reference','out'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--max-iterations',type=int,default=32)
 p.add_argument('--probe-mode',choices=['clean','augmented'],default='clean')
 a=p.parse_args();r=run(a.training_cache,a.validation_cache,a.reference,a.out,a.max_iterations,a.probe_mode);print(json.dumps({'selected_candidate':r['selected_candidate'],'reference_validation':r['reference_validation'],'baseline_validation':r['baseline_validation']},indent=2))
if __name__=='__main__':main()
