"""External-trained boundary evidence plus minimum-duration offline decoding.

The RNN weights remain frozen. Boundary fitting uses J03WOH only; settings are
selected on J03WPY. No previously viewed test is loaded by this module.
"""
from pathlib import Path
import json,argparse
import numpy as np
from .rnn import observation_probabilities,RNN_SYMMETRIC_VERSION
from .model import (instantaneous_features,interaction_features,boundary_features,
                    boundary_targets,fit_classifier,probabilities,decode)
from .rnn_refine import read_cache
from .rnn_benchmark import conditions,evaluate
from .experiment import missing_mask
from .pipeline import digest,pm

BOUNDARY_VERSION='numpy-tanh-rnn-v1-boundary-dwell-v1'

def invariant_boundary_features(sequence,pitch,p,mask=None):
    x=instantaneous_features(sequence,pitch,ball_missing=mask);a=x[:,6:18];b=x[:,18:30]
    geometry=np.column_stack([x[:,:6],np.minimum(a,b),np.maximum(a,b),np.abs(a-b)])
    interaction=interaction_features(sequence,ball_missing=mask)
    geometry=np.column_stack([geometry,np.minimum(interaction[:,:5],interaction[:,5:]),np.maximum(interaction[:,:5],interaction[:,5:])])
    team_free=np.column_stack([np.minimum(p[:,0],p[:,1]),np.maximum(p[:,0],p[:,1]),p[:,2]])
    return boundary_features(geometry,team_free)

def dwell_decode(p,penalty,boundary,strength=1,min_frames=5):
    """Viterbi with transition evidence and a hard dwell floor for interior runs.

    First/last runs can be truncated by the sequence boundary. The floor applies
    to every state, including D, preventing a short stop pulse from splitting a
    possession. Cost reductions cannot force a switch without emission evidence.
    """
    p=np.asarray(p)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p):raise ValueError('Expected nonempty N x 3 probabilities')
    if not isinstance(min_frames,(int,np.integer)) or not 1<=min_frames<=100:raise ValueError('Invalid dwell floor')
    if penalty<0 or not np.isfinite(penalty) or strength<0 or not np.isfinite(strength):raise ValueError('Invalid transition settings')
    boundary=np.asarray(boundary)
    if boundary.shape!=(len(p),) or not np.isfinite(boundary).all() or ((boundary<0)|(boundary>1)).any():raise ValueError('Invalid boundary probabilities')
    if min_frames==1:return decode(p,penalty,boundary,strength)
    m=min_frames;n=len(p);logp=np.log(np.maximum(p,1e-8))
    costs=penalty*np.clip(1+strength*(1-2*boundary),.15,3)
    score=np.full((3,m),-np.inf);score[:,-1]=logp[0]
    switch_from=np.zeros((n,3),np.int8);mature_from=np.zeros((n,3),np.int16)
    diagonal=np.eye(3,dtype=bool);labels=np.arange(3)
    for t in range(1,n):
        new=np.full_like(score,-np.inf);new[:,1:]=score[:,:-1]
        mature=score[:,-1]>=score[:,-2]
        mature_from[t]=np.where(mature,m-1,m-2)
        new[:,-1]=np.maximum(new[:,-1],score[:,-1])
        change=score[:,-1,None]-costs[t];change=np.where(diagonal,-np.inf,change)
        origins=np.argmax(change,axis=0);switch_from[t]=origins
        new[:,0]=change[origins,labels]
        score=new+logp[t,:,None]
    label,age=np.unravel_index(np.argmax(score),score.shape);out=np.empty(n,np.int8);out[-1]=label
    for t in range(n-1,0,-1):
        if age==0:label=int(switch_from[t,label]);age=m-1
        elif age==m-1:age=int(mature_from[t,label])
        else:age-=1
        out[t-1]=label
    return out

def load_base(checkpoint):
    with np.load(checkpoint,allow_pickle=False) as d:
        if str(d['version'])!=RNN_SYMMETRIC_VERSION:raise ValueError('Expected frozen symmetric RNN')
        return {k:d[k] for k in ('mean','scale','wx','wh','bh','wy','by')},float(d['transition_penalty'])

def spell_diagnostics(rows,preds):
    result={}
    for condition in ('clean','masked'):
        durations=[];hits=[];pred_lengths=[];merges=0
        for (s,_,c,_),pred in zip(rows,preds):
            if c!=condition:continue
            runs=pm.valid_runs(s['y']);gt=pm.possession_spells(s['y'],runs);ps=pm.possession_spells(pred,runs)
            overlap=pm.spell_overlaps(gt,ps);matched=pm.matched_gt_flags(overlap,len(gt),.5)
            durations.extend([(b-a)/25 for a,b,_ in gt]);hits.extend(matched);pred_lengths.extend([(b-a)/25 for a,b,_ in ps])
            # Geometric merge diagnostic: a prediction overlaps >1 same-team reference spell.
            merges+=sum(sum(iou>0 and j==pi for iou,gi,j in overlap)>1 for pi in range(len(ps)))
        durations=np.array(durations);hits=np.array(hits,dtype=bool)
        bins=[]
        for lo,hi in [(0,1),(1,3),(3,10),(10,float('inf'))]:
            mask=(durations>=lo)&(durations<hi);total=int(mask.sum())
            bins.append({'duration_seconds':[lo,hi if np.isfinite(hi) else None],'reference_spells':total,
                         'matched_at_iou_0_5':int(hits[mask].sum()),'recall':float(hits[mask].mean()) if total else None})
        result[condition]={'recall_by_duration':bins,'matched_reference_spells_at_iou_0_5':int(hits.sum()),
                           'missed_reference_spells_at_iou_0_5':int(len(hits)-hits.sum()),
                           'false_positive_spells_at_iou_0_5':int(len(pred_lengths)-hits.sum()),
                           'predicted_spells_under_0_4_seconds':int((np.array(pred_lengths)<.4).sum()),
                           'predictions_overlapping_multiple_reference_spells':merges,
                           'note':'Merge count is descriptive overlap, not a separately validated event metric.'}
    return result

def run(training_cache,validation_cache,checkpoint,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output folder')
    training=read_cache(training_cache,'J03WOH');validation=read_cache(validation_cache,'J03WPY')
    model,reference_penalty=load_base(checkpoint);out.mkdir(parents=True)
    protocol={'training_match':'J03WOH','validation_match':'J03WPY','viewed_tests_loaded':False,
              'competition_data_used':False,'base_sha256':digest(checkpoint),
              'penalties':[12,24],'strengths':[.5,1,2],'minimum_frames':[1,5,10],
              'selection':'mean clean/masked validation spell F1; no spell-count matching',
              'head':'team-invariant logistic boundary verifier; 35 epochs; external training only',
              'reference_cache_sha256':{'training':digest(training_cache),'validation':digest(validation_cache)}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    xs=[];ys=[]
    for i,(s,pitch) in enumerate(training):
        for masked in (False,True):
            mask=missing_mask(len(s['y']),42+i) if masked else None
            p=observation_probabilities(model,s,pitch,True,mask)
            x=invariant_boundary_features(s,pitch,p,mask);y=boundary_targets(s['y'])
            ix=np.flatnonzero((s['y']>=0)&(s['y']<3))[::5]
            xs.append(x[ix]);ys.append(y[ix])
    print('Fitting external-only boundary verifier',flush=True)
    head=fit_classifier(np.concatenate(xs),np.concatenate(ys),classes=2,epochs=35)
    rows=conditions(validation,100);probs=[];events=[]
    for s,pitch,c,mask in rows:
        p=observation_probabilities(model,s,pitch,True,mask);probs.append(p)
        events.append(probabilities(head,invariant_boundary_features(s,pitch,p,mask))[:,1])
    baseline_predictions=[decode(p,reference_penalty) for p in probs]
    baseline=evaluate(rows,baseline_predictions);trials=[]
    for penalty in protocol['penalties']:
        for strength in protocol['strengths']:
            for minimum in protocol['minimum_frames']:
                print(f'Validation: penalty={penalty} strength={strength} minimum={minimum}',flush=True)
                preds=[dwell_decode(p,penalty,b,strength,minimum) for p,b in zip(probs,events)]
                trials.append({'penalty':penalty,'strength':strength,'min_frames':minimum,**evaluate(rows,preds)})
    chosen=max(trials,key=lambda t:t['selection_spell_f1'])
    improved=chosen['selection_spell_f1']>baseline['selection_spell_f1']
    if improved:
        np.savez_compressed(out/'candidate-model.npz',**model,version=np.array(BOUNDARY_VERSION),
                            **{f'event_{k}':v for k,v in head.items()},symmetric=np.array(True),
                            transition_penalty=np.array(chosen['penalty']),boundary_strength=np.array(chosen['strength']),
                            min_frames=np.array(chosen['min_frames']))
        selected_predictions=[dwell_decode(p,chosen['penalty'],b,chosen['strength'],chosen['min_frames']) for p,b in zip(probs,events)]
    else:
        import shutil
        shutil.copy2(checkpoint,out/'candidate-model.npz');selected_predictions=baseline_predictions
    report={'protocol':protocol,'reference_validation':baseline,'best_trial':chosen,'candidate_improved_validation':improved,
            'validation_trials':trials,'reference_diagnostics':spell_diagnostics(rows,baseline_predictions),
            'candidate_diagnostics':spell_diagnostics(rows,selected_predictions),
            'candidate_sha256':digest(out/'candidate-model.npz'),'test_results':None,
            'limitations':['Validation has been reused; a new independent test is required.',
                           'Boundary verifier is trained on training-match RNN predictions; generalisation is uncertain.',
                           'Boundary context and decoding are offline, not live forecasts.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('training-cache','validation-cache','checkpoint','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=run(a.training_cache,a.validation_cache,a.checkpoint,a.out)
    print(json.dumps({k:r[k] for k in ('candidate_improved_validation','reference_validation','best_trial','candidate_diagnostics')},indent=2))
if __name__=='__main__':main()
