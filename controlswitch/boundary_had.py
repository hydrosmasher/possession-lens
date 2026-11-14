"""External-only H/A turnover verification020 on frozen complete019 emissions."""
from pathlib import Path
import argparse,json
import numpy as np
from .tracker_robustness import CONDITIONS,condition,evaluate
from .external import period_arrays
from .rnn_runtime import load_frozen,infer_frozen
from .control_correction import corrected_probabilities,blend_control
from .deadball import observation_features,fuse
from .rnn_boundaries import invariant_boundary_features
from .model import shifted,rolling,fit_classifier,probabilities,decode
from .pipeline import digest
from . import tree_model

VERSION='had-boundary-020'
VARIANTS=[{'name':'balanced_half','strength':.5,'mode':'balanced'},{'name':'balanced_full','strength':1.,'mode':'balanced'},{'name':'recovery_half','strength':.5,'mode':'recovery'}]

def adaptive_decode(p,score,strength=.5,mode='balanced',penalty=12.,stop_penalty=3.):
    p=np.asarray(p);score=np.asarray(score)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)==0 or score.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(score).all() or (p<0).any() or (p.sum(1)<=0).any() or ((score<0)|(score>1)).any():raise ValueError('Invalid emission or event-score arrays')
    if mode not in ('balanced','recovery') or not np.isfinite(strength) or not 0<=strength<=1 or not np.isfinite(penalty) or penalty<0 or not np.isfinite(stop_penalty) or stop_penalty<0:raise ValueError('Invalid adaptive decoder settings')
    factor=np.clip(1+strength*(1-2*score),.25,2) if mode=='balanced' else np.maximum(1-strength*score,.25)
    costs=penalty*factor;n=len(p);logp=np.log(np.maximum(p,1e-8));back=np.zeros((n,3),np.int8);values=logp[0].copy()
    for i in range(1,n):
        matrix=np.array([[0,costs[i],stop_penalty],[costs[i],0,stop_penalty],[stop_penalty,stop_penalty,0]])
        choices=values[:,None]-matrix;back[i]=choices.argmax(0);values=choices[back[i],np.arange(3)]+logp[i]
    labels=np.empty(n,np.int8);labels[-1]=values.argmax()
    for i in range(n-1,0,-1):labels[i-1]=back[i,labels[i]]
    return labels

def boundary_observations(s,pitch,p,mask=None):
    # Sorted state evidence alone loses a confident H/A reversal; pooled absolute
    # probability changes retain it while remaining invariant to team order.
    changes=[np.abs(shifted(p,k)-shifted(p,-k))[:,:2].sum(1) for k in (12,25,50)]
    return np.column_stack([invariant_boundary_features(s,pitch,p,mask),*changes]).astype(np.float32)

def targets(y):
    y=np.asarray(y);direct=np.r_[False,(y[1:]<2)&(y[:-1]<2)&(y[1:]!=y[:-1])]
    event=(rolling(direct[:,None].astype(float),12)[:,0]>0).astype(np.int8)
    known_live=rolling((y<2)[:,None].astype(float),12)[:,0]>.999
    return event,known_live

def frozen_emissions(backbone,control,dead,s,pitch,mask):
    _,rnn=infer_frozen(backbone,s,pitch,mask)
    live=blend_control(rnn,corrected_probabilities(control,s,pitch,mask),.75)
    d=tree_model.predict(dead,observation_features(s,pitch,mask))[:,1]
    return fuse(live,d,.75)

def run(cache_root,control_root,dead_root,team_model,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new run folder')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'history':'Two extensively reused Bundesliga matches; all020 follows019 development; no independent test.','folds':[{'train':'J03WMX','validation':'J03WN1'},{'train':'J03WN1','validation':'J03WMX'}],'competition_data_used':False,'sources':'Existing first-division IDSSE/Sportec CC BY4.0; provider BallStatus/Possession supervision only','emissions':'Frozen019 in each opposite-match training fold; RNN, H/A correction and dead experts unchanged','conditions':list(CONDITIONS),'training_conditions':'same as018/019; unchanged masks and tracker-noise seeds','head':'L2 multinomial logistic2-class,25epochs,seed120,l2=.001; train-match-only standardisation and fit','sampling':'every5th valid live-neighbourhood frame; randomly downsample negatives to positives per half/condition; event scores not naturally calibrated probabilities','target':'Direct known H/A turnover within+-12frames; only entirely known-live neighbourhoods valid; D changes excluded','features':'team-invariant existing geometry/interaction/boundary features plus pooled absolute H/A probability changes at+-12/25/50frames; no labels or timestamps','variants':VARIANTS,'decoder':'Only H/A transition cost adapts; stop/restart cost3 fixed; no hard duration floor','selection':'Best four-condition meanF1 passing no aggregate misses/merges/FP increase; each clean/ball-gap fold no F1 loss>.01; each tracker-noise fold no loss; deadF1 loss<=.005 each fold/condition. No post-result retuning.','hashes':{'team':digest(team_model),'control':[digest(Path(control_root)/f'fold-{i}/control-estimator.npz') for i in (0,1)],'dead':[digest(Path(dead_root)/f'fold-{i}/dead-blend-0.5.npz') for i in (0,1)]}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');backbone=load_frozen(team_model);folds=[];all_rows=[];all_current=[];all_preds={v['name']:[] for v in VARIANTS}
    for fold,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('Fold',fold,train,'->',val,flush=True);fd=out/f'fold-{fold}';fd.mkdir();records={}
        for mid in (train,val):
            paths=list(Path(cache_root).glob(mid+'-*.npz'))
            if len(paths)!=1:raise ValueError('Ambiguous cache')
            with np.load(paths[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
            if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
        control=tree_model.load(Path(control_root)/f'fold-{fold}/control-estimator.npz');dead=tree_model.load(Path(dead_root)/f'fold-{fold}/dead-blend-0.5.npz');xs=[];ys=[];audit=[]
        for half in (0,1):
            original=period_arrays(records[train],half);pitch=records[train]['pitch'];t,valid=targets(original['y'])
            for ci,c in enumerate(CONDITIONS):
                print('Training features',fold,half,c,flush=True);s,mask=condition(original,c,half,True);p=frozen_emissions(backbone,control,dead,s,pitch,mask);x=boundary_observations(s,pitch,p,mask)
                ix=np.flatnonzero(valid)[::5];pos=ix[t[ix]==1];neg=ix[t[ix]==0];rng=np.random.default_rng(120+half*4+ci)
                if len(pos)==0:continue
                neg=rng.choice(neg,size=min(len(neg),len(pos)),replace=False);chosen=np.r_[pos,neg];xs.append(x[chosen]);ys.append(t[chosen]);audit.append({'half':half+1,'condition':c,'positive':len(pos),'negative':len(neg)})
        print('Fitting event head',fold,flush=True);head=fit_classifier(np.concatenate(xs),np.concatenate(ys),classes=2,epochs=25,seed=120,l2=.001);np.savez_compressed(fd/'boundary-head.npz',**head)
        rows=[];current=[];preds={v['name']:[] for v in VARIANTS}
        for half in (0,1):
            original=period_arrays(records[val],half);pitch=records[val]['pitch']
            for c in CONDITIONS:
                print('Validation',fold,half,c,flush=True);s,mask=condition(original,c,half,False);rows.append({'y':original['y'],'condition':c});p=frozen_emissions(backbone,control,dead,s,pitch,mask);q=probabilities(head,boundary_observations(s,pitch,p,mask))[:,1];current.append(decode(p,12,stop_penalty=3))
                for v in VARIANTS:preds[v['name']].append(adaptive_decode(p,q,v['strength'],v['mode']))
                np.savez_compressed(fd/f'validation-{half*4+CONDITIONS.index(c)}.npz',reference=original['y'],emissions=p,boundary_score=q,current=current[-1],**{k:v[-1] for k,v in preds.items()})
        result={'train':train,'validation':val,'training_audit':audit,'head_sha256':digest(fd/'boundary-head.npz'),'current':evaluate(rows,current),'variants':{k:evaluate(rows,p,current) for k,p in preds.items()}}
        (fd/'report.json').write_text(json.dumps(result,indent=2)+'\n');folds.append(result);all_rows+=rows;all_current+=current
        for k in all_preds:all_preds[k]+=preds[k]
    current=evaluate(all_rows,all_current);trials=[]
    for v in VARIANTS:
        name=v['name'];score=evaluate(all_rows,all_preds[name],all_current);a=score['totals'];b=current['totals'];guards={'mean_f1':score['mean_spell_f1']>current['mean_spell_f1'],'misses':a['missed']<=b['missed'],'merges':a['merges']<=b['merges'],'false_positives':a['false_positive']<=b['false_positive'],'prior_conditions':all(f['variants'][name][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'tracker_noise':all(f['variants'][name]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),'dead_f1':all(f['variants'][name][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS)}
        trials.append({**v,'scores':score,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None,'best_unrestricted':max(trials,key=lambda t:t['scores']['mean_spell_f1']),'fresh_test_results':None,'promoted_to_default':False,'limitations':['Repeatedly reused two-match development with in-fold emissions used to fit the boundary head; no independent test.','Balanced scores are event evidence, not calibrated turnover likelihood.','Noise simulation does not establish real tracker robustness.','Counts repeat possessions across conditions; merges are geometric diagnostics.','No provided-game labels, video, tracking or public score used.','All context and decoding are offline.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','control-root','dead-root','team-model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,a.control_root,a.dead_root,a.team_model,a.out)
if __name__=='__main__':main()
