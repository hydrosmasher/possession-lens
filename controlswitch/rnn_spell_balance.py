"""Train with bounded inverse-possession-duration weights on external training data.

Each valid frame is still supervised by its actual class. There is no injection
of labels at inference. Longer spells cannot dominate solely by frame count.
"""
from pathlib import Path
import argparse,json
import numpy as np
from .rnn_refine import read_cache
from .rnn_boundaries import dwell_decode,spell_diagnostics
from .rnn import train,observation_probabilities,RNN_DWELL_VERSION
from .rnn_benchmark import conditions,evaluate
from .model import instantaneous_features,decode
from .pipeline import digest,pm

def spell_weights(y,anchor,alpha):
    w=np.zeros(len(y),np.float32)
    for a,b,k in pm.segments(y,pm.valid_runs(y)):
        w[a:b]=float(np.clip((anchor/(b-a))**alpha,.25,4)) if k<2 else .5
    return w

def run(training_cache,validation_cache,reference,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new run folder')
    training=read_cache(training_cache,'J03WOH');validation=read_cache(validation_cache,'J03WPY');out.mkdir(parents=True)
    lengths=[b-a for s,p in training for a,b,k in pm.possession_spells(s['y'],pm.valid_runs(s['y']))]
    anchor=float(np.median(lengths))
    protocol={'training_match':'J03WOH','validation_match':'J03WPY','viewed_tests_loaded':False,
              'competition_data_used':False,'epochs':5,'alphas':[.5,1.0],'clip':[.25,4],
              'training_median_live_spell_frames':anchor,'dead_frame_weight':.5,
              'normalisation':'one global training mean; weighted loss divided by valid frame count per chunk',
              'penalties':[6,12,24],'minimum_frames':[1,5,10],
              'selection':'mean clean/masked validation spell F1, no spell-count matching',
              'reference_sha256':digest(reference)}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with np.load(reference,allow_pickle=False) as d:
        old={k:d[k] for k in ('mean','scale','wx','wh','bh','wy','by')};oldpenalty=float(d['transition_penalty'])
    rows=conditions(validation,100);oldp=[observation_probabilities(old,s,p,True,mask) for s,p,c,mask in rows]
    oldpred=[decode(p,oldpenalty) for p in oldp];reference_metrics=evaluate(rows,oldpred)
    trials=[];models={};histories={}
    sequences=[(instantaneous_features(s,p),s['y']) for s,p in training]
    for alpha in protocol['alphas']:
        print('Training spell balance alpha',alpha,flush=True)
        weights=[spell_weights(y,anchor,alpha) for x,y in sequences]
        model,loss=train(sequences,epochs=5,sample_weights=weights,weight_normalisation='global')
        models[alpha]=model;histories[str(alpha)]=loss
        probs=[observation_probabilities(model,s,p,True,mask) for s,p,c,mask in rows]
        for penalty in protocol['penalties']:
            for minimum in protocol['minimum_frames']:
                print('Validation',alpha,penalty,minimum,flush=True)
                preds=[dwell_decode(p,penalty,np.full(len(p),.5),0,minimum) for p in probs]
                trials.append({'alpha':alpha,'penalty':penalty,'min_frames':minimum,**evaluate(rows,preds)})
    chosen=max(trials,key=lambda t:t['selection_spell_f1']);improved=chosen['selection_spell_f1']>reference_metrics['selection_spell_f1']
    if improved:
        model=models[chosen['alpha']]
        np.savez_compressed(out/'candidate-model.npz',**model,version=np.array(RNN_DWELL_VERSION),symmetric=np.array(True),
                            transition_penalty=np.array(chosen['penalty']),min_frames=np.array(chosen['min_frames']))
        probs=[observation_probabilities(model,s,p,True,mask) for s,p,c,mask in rows]
        preds=[dwell_decode(p,chosen['penalty'],np.full(len(p),.5),0,chosen['min_frames']) for p in probs]
    else:
        import shutil
        shutil.copy2(reference,out/'candidate-model.npz');preds=oldpred
    report={'protocol':protocol,'reference_validation':reference_metrics,'best_trial':chosen,
            'candidate_improved_validation':improved,'validation_trials':trials,'training_loss':histories,
            'reference_diagnostics':spell_diagnostics(rows,oldpred),'candidate_diagnostics':spell_diagnostics(rows,preds),
            'candidate_sha256':digest(out/'candidate-model.npz'),'test_results':None,
            'limitations':['Validation reused for development; a fresh test is required.',
                           'Weighted training may trade frame accuracy for recovering shorter spells.',
                           'Offline decoder uses future evidence.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('training-cache','validation-cache','reference','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=run(a.training_cache,a.validation_cache,a.reference,a.out)
    print(json.dumps({k:r[k] for k in ('candidate_improved_validation','reference_validation','best_trial','candidate_diagnostics')},indent=2))
if __name__=='__main__':main()
