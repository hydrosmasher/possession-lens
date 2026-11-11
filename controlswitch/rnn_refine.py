"""Validation-only RNN refinement after the J03WMX test was viewed.

This command accepts exactly J03WOH training and J03WPY validation caches. It has
no test input or test-scoring option. No competition inputs are supported.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from .external import period_arrays
from .rnn import train,predict,RNN_VERSION,RNN_REFINED_VERSION
from .model import instantaneous_features,decode
from .rnn_benchmark import conditions,evaluate
from .pipeline import digest

SOURCES={'J03WOH':'ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f',
         'J03WPY':'845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1'}
TEAM_PENALTIES=(6,12,24)
STOP_PENALTIES=(3,6,12,24)
BOOSTS=(3,6)

def transition_weights(y,boost=3,radius=12):
    """Supervision weights from external TRAINING labels only (not model inputs)."""
    y=np.asarray(y);valid=(y>=0)&(y<3)
    if boost<1 or radius<0:raise ValueError('Invalid transition weighting settings')
    changes=np.r_[False,(y[1:]!=y[:-1])&valid[1:]&valid[:-1]]
    index=np.arange(len(y));prefix=np.r_[0,np.cumsum(changes)]
    nearby=prefix[np.minimum(len(y),index+radius+1)]-prefix[np.maximum(0,index-radius)]>0
    return np.where(valid,np.where(nearby,boost,1),0).astype(np.float32)

def read_cache(path,expected):
    if expected not in SOURCES:raise ValueError('Refinement accepts training/validation identities only')
    with np.load(path,allow_pickle=False) as d:
        data={k:d[k] for k in d.files}
    if str(data['match_id'])!=expected:raise ValueError('Wrong external cache for declared role')
    if str(data['source_sha256'])!=SOURCES[expected]:raise ValueError('Unexpected external source hash')
    return [(period_arrays(data,i),data['pitch']) for i in (0,1)]

def run(training_cache,validation_cache,reference_checkpoint,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new refinement folder')
    training=read_cache(training_cache,'J03WOH');validation=read_cache(validation_cache,'J03WPY')
    out.mkdir(parents=True)
    protocol={'status':'validation_only_development','epochs':5,'seed':42,'hidden_units':24,
              'training_match':'J03WOH','validation_match':'J03WPY','training_boosts':BOOSTS,
              'transition_weight_radius_frames':12,'team_penalties':TEAM_PENALTIES,'stop_penalties':STOP_PENALTIES,
              'selection':'mean clean/masked validation spell F1; never match a target spell count',
              'validation_dropout_seed':100,'competition_data_used':False,'test_match_loaded':False,
              'previously_viewed_J03WMX_not_scored':True,'fresh_test_required':True,
              'reference_model_sha256':digest(reference_checkpoint),
              'cache_sha256':{'training':digest(training_cache),'validation':digest(validation_cache)},
              'original_source_sha256':SOURCES}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with np.load(reference_checkpoint,allow_pickle=False) as d:
        if str(d['version'])!=RNN_VERSION:raise ValueError('Unexpected reference model version')
        reference={k:d[k] for k in ('mean','scale','wx','wh','bh','wy','by')}
        original_penalty=float(d['transition_penalty'])
    rows=conditions(validation,100);trials=[];models={'reference':reference}
    sequences=[(instantaneous_features(s,p),s['y']) for s,p in training]
    histories={}
    for boost in BOOSTS:
        name=f'boundary_weight_{boost}'
        print('Training',name,flush=True)
        weights=[transition_weights(y,boost) for x,y in sequences]
        models[name],histories[name]=train(sequences,epochs=5,sample_weights=weights)
        np.savez_compressed(out/f'{name}.npz',**models[name],version=np.array(RNN_VERSION))
    for name,model in models.items():
        print('Selecting decoder on validation only:',name,flush=True)
        probs=[predict(model,instantaneous_features(s,pitch,ball_missing=mask)) for s,pitch,c,mask in rows]
        if name=='reference':
            reference_metrics=evaluate(rows,[decode(p,original_penalty) for p in probs])
        for team in TEAM_PENALTIES:
            for stop in STOP_PENALTIES:
                trial={'model':name,'team_penalty':team,'stop_penalty':stop,
                       **evaluate(rows,[decode(p,team,stop_penalty=stop) for p in probs])}
                trials.append(trial)
    chosen=max(trials,key=lambda t:t['selection_spell_f1'])
    # Keep the reference if the refinement search fails to improve the fixed comparator.
    if chosen['selection_spell_f1']<=reference_metrics['selection_spell_f1']:
        chosen={'model':'reference','team_penalty':original_penalty,'stop_penalty':original_penalty,**reference_metrics}
    model=models[chosen['model']]
    np.savez_compressed(out/'candidate-model.npz',**model,version=np.array(RNN_REFINED_VERSION),decoder=np.array('viterbi'),
                        transition_penalty=np.array(chosen['team_penalty']),stop_penalty=np.array(chosen['stop_penalty']))
    report={'protocol':protocol,'reference_validation':reference_metrics,'selected_validation':chosen,
            'validation_trials':trials,'training_losses':histories,
            'candidate_sha256':digest(out/'candidate-model.npz'),'test_results':None,
            'limitations':['J03WPY has been reused for development; selection may overfit this match.',
                           'No new independent test has been performed. Do not replace the tested checkpoint based on validation alone.',
                           'Weighted training labels are supervision only; inference needs observations, never labels.',
                           'Viterbi uses future evidence and is for offline analysis.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-cache','validation-cache','reference-checkpoint','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=run(a.training_cache,a.validation_cache,a.reference_checkpoint,a.out)
    print(json.dumps({k:r[k] for k in ('reference_validation','selected_validation','test_results')},indent=2))
if __name__=='__main__':main()
