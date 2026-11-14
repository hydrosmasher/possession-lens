"""External-only tracker-error augmentation018. No challenge assets are accepted."""
from pathlib import Path
import argparse,json
import numpy as np
from .control_correction import control_features,corrected_probabilities,blend_control
from .deadball import observation_features,fuse
from .gap_possession import long_mask,diagnostics
from .experiment import missing_mask,aggregate
from .external import period_arrays
from .rnn_runtime import load_frozen,infer_frozen
from .model import decode
from .pipeline import digest
from . import tree_model

CONDITIONS=('clean','masked','long_gap','tracker_noise')
VERSION='tracker-robustness-018'

def interval_mask(n,rng,fraction,low,high):
    if n<1 or not 0<=fraction<=1 or not 1<=low<=high:raise ValueError('Invalid mask specification')
    mask=np.zeros(n,bool)
    while mask.mean()<fraction:
        a=int(rng.integers(n));mask[a:min(n,a+int(rng.integers(low,high+1)))]=True
    return mask

def noisy_tracking(sequence,seed):
    """Synthetic observation degradation independent of possession references."""
    rng=np.random.default_rng(seed)
    s={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in sequence.items() if k!='y'}
    n=len(s['ball']);mask=interval_mask(n,rng,.4,50,300)
    s['ball']+=rng.normal(0,.25,s['ball'].shape)
    s['ball_speed']=np.where(np.isfinite(s['ball_speed']),np.clip(s['ball_speed']+rng.normal(0,.5,n),0,50),np.nan)
    s['ball'][mask]=np.nan;s['ball_speed'][mask]=np.nan
    s['xy']+=rng.normal(0,.35,s['xy'].shape)
    s['speed']=np.where(np.isfinite(s['speed']),np.clip(s['speed']+rng.normal(0,.3,s['speed'].shape),0,15),np.nan)
    for j in range(len(s['teams'])):
        missing=interval_mask(n,rng,.1,25,100);s['xy'][missing,j]=np.nan;s['speed'][missing,j]=np.nan
    duplicates=[int(np.flatnonzero(s['teams']==team)[0]) for team in (0,1)]
    s['xy']=np.concatenate([s['xy'],s['xy'][:,duplicates]+rng.normal(0,.1,(n,2,2))],axis=1)
    s['speed']=np.column_stack([s['speed'],s['speed'][:,duplicates]])
    s['teams']=np.r_[s['teams'],s['teams'][duplicates]]
    return s

def condition(sequence,kind,half,training):
    if kind=='tracker_noise':return noisy_tracking(sequence,(618 if training else 818)+half),None
    mask=None if kind=='clean' else missing_mask(len(sequence['ball']),(114 if training else 214)+half) if kind=='masked' else long_mask(len(sequence['ball']),(314 if training else 414)+half)
    return {k:v for k,v in sequence.items() if k!='y'},mask

def evaluate(rows,preds,current=None):
    result={}
    for c in CONDITIONS:
        ix=[i for i,r in enumerate(rows) if r['condition']==c];score=aggregate([(rows[i]['y'],preds[i]) for i in ix]);ds=[diagnostics(rows[i]['y'],preds[i],current[i] if current is not None else None) for i in ix]
        score['diagnostics']={k:sum(d[k] for d in ds) for k in ds[0]}
        y=np.concatenate([rows[i]['y'] for i in ix]);p=np.concatenate([preds[i] for i in ix]);tp=int(((y==2)&(p==2)).sum());score['dead_f1']=2*tp/max(int((y==2).sum()+(p==2).sum()),1)
        result[c]=score
    result['mean_spell_f1']=float(np.mean([result[c]['spell_f1_mean'] for c in CONDITIONS]));result['totals']={k:sum(result[c]['diagnostics'][k] for c in CONDITIONS) for k in result['clean']['diagnostics']}
    return result

def fit_head(records,train_id,kind,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    xs=[];ys=[]
    for half in (0,1):
        original=period_arrays(records[train_id],half);y=original['y'];pitch=records[train_id]['pitch'];ix=np.flatnonzero(y<(2 if kind=='control' else 3))[::5]
        for c in CONDITIONS:
            s,mask=condition(original,c,half,True)
            for swap in ((False,True) if kind=='control' else (False,)):
                x=control_features(s,pitch,mask,swap) if kind=='control' else observation_features(s,pitch,mask)
                xs.append(x[ix])
                target=(1-y[ix] if swap else y[ix]) if kind=='control' else (y[ix]==2).astype(int)
                ys.append(target)
    x=np.concatenate(xs);y=np.concatenate(ys);head=HistGradientBoostingClassifier(max_iter=100,learning_rate=.08,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=118)
    with threadpool_limits(limits=1):head.fit(x,y)
    tree_model.save(out,head,'hist_boost');export=tree_model.load(out)
    if not np.allclose(tree_model.predict(export,x[:10000])[:,:2],head.predict_proba(x[:10000]),atol=1e-9):raise AssertionError('Export mismatch')
    return head,export,{'samples':len(y),'classes':np.bincount(y).tolist(),'sha256':digest(out)}

def run(cache_root,reference_root,forward_dead,reverse_dead,team_model,out):
    from threadpoolctl import threadpool_limits
    out=Path(out)
    if out.exists():raise ValueError('Choose new experiment folder')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'folds':[{'train':'J03WMX','validation':'J03WN1'},{'train':'J03WN1','validation':'J03WMX'}],'history':'Both previously viewed; exploratory reused validation after017. No independent test.','competition_data_used':False,'sources':'Existing first-division Bundesliga IDSSE/Sportec, CC BY4.0, Bassek et al2025; provider BallStatus/BallPossession labels separate from observations','conditions':list(CONDITIONS),'noise':{'ball_gap_target':.4,'ball_gap_draw_frames':[50,300],'ball_jitter_std_m':.25,'ball_speed_jitter_std_mps':.5,'player_gap_target_per_slot':.1,'player_gap_draw_frames':[25,100],'player_jitter_std_m':.35,'player_speed_jitter_std_mps':.3,'duplicates':2,'duplicate_jitter_std_m':.1,'train_seed':618,'validation_seed':818},'noise_selection':'Fixed before any018 fit/evaluation, independent of challenge and labels; no estimation from provided game. Overlapping masks can form longer gaps.','other_conditions':'Same short/long masks and train/validation seeds as017','training':'one fixed estimator each for H/A and alive/dead, train clean/short/long/tracker_noise, live-only H/A with team swaps; same100 boosting iterations/15leaves/lr.08/minleaf40/l2=10/no earlystop; seed118; stride5','variants':['new_control','new_dead','both'],'frozen_decoder':{'mix':.75,'penalty':12,'stop_penalty':3,'dead_weight':.75},'selection':'Best four-condition mean spellF1 among variants with no aggregate increase in misses/merges/FP; each fold original clean/masked/long F1 no >.01 decline; each fold tracker_noise F1 no decline; each fold deadF1 no >.005 decline in every condition. No post-result sweep.','reference_control_sha256':[digest(Path(reference_root)/f'fold-{i}/control-estimator.npz') for i in (0,1)],'reference_dead_sha256':[digest(forward_dead),digest(reverse_dead)],'team_sha256':digest(team_model)}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');backbone=load_frozen(team_model);folds=[];all_rows=[];all_current=[];all_preds={k:[] for k in protocol['variants']}
    for fold,(train,val,dead_path) in enumerate([('J03WMX','J03WN1',forward_dead),('J03WN1','J03WMX',reverse_dead)]):
        print('Fold',fold,train,'->',val,flush=True);fd=out/f'fold-{fold}';fd.mkdir();records={}
        for mid in (train,val):
            paths=list(Path(cache_root).glob(mid+'-*.npz'))
            if len(paths)!=1:raise ValueError('Ambiguous cache')
            with np.load(paths[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
            if str(records[mid]['match_id'])!=mid:raise ValueError('Match mismatch')
        print('Training control',flush=True);control_native,new_control,control_info=fit_head(records,train,'control',fd/'control-estimator.npz')
        print('Training dead',flush=True);dead_native,new_dead,dead_info=fit_head(records,train,'dead',fd/'dead-estimator.npz')
        old_control=tree_model.load(Path(reference_root)/f'fold-{fold}/control-estimator.npz');old_dead=tree_model.load(dead_path);rows=[];current=[];preds={k:[] for k in protocol['variants']}
        for half in (0,1):
            original=period_arrays(records[val],half);pitch=records[val]['pitch']
            for c in CONDITIONS:
                print('Validation',fold,half,c,flush=True);s,mask=condition(original,c,half,False);rows.append({'y':original['y'],'condition':c})
                _,rnn=infer_frozen(backbone,s,pitch,mask);old_h=corrected_probabilities(old_control,s,pitch,mask);new_h=corrected_probabilities(new_control,s,pitch,mask)
                x=observation_features(s,pitch,mask);old_d=tree_model.predict(old_dead,x)[:,1];new_d=tree_model.predict(new_dead,x)[:,1]
                with threadpool_limits(limits=1):
                    native_d=dead_native.predict_proba(x)[:,1]
                    native_h=(control_native.predict_proba(control_features(s,pitch,mask))+control_native.predict_proba(control_features(s,pitch,mask,True))[:,[1,0]])*.5
                if not np.allclose(native_d,new_d,atol=1e-9) or not np.allclose(native_h,new_h[:,:2],atol=1e-9):raise AssertionError('Validation export mismatch')
                current.append(decode(fuse(blend_control(rnn,old_h,.75),old_d,.75),12,stop_penalty=3))
                for name,h,d in [('new_control',new_h,old_d),('new_dead',old_h,new_d),('both',new_h,new_d)]:preds[name].append(decode(fuse(blend_control(rnn,h,.75),d,.75),12,stop_penalty=3))
        result={'train':train,'validation':val,'control_training':control_info,'dead_training':dead_info,'current':evaluate(rows,current),'variants':{k:evaluate(rows,p,current) for k,p in preds.items()},'all_native_export_predictions_match':True};folds.append(result);(fd/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        for i,row in enumerate(rows):np.savez_compressed(fd/f'validation-{i}.npz',reference=row['y'],current=current[i],**{k:p[i] for k,p in preds.items()})
        all_rows+=rows;all_current+=current
        for k in all_preds:all_preds[k]+=preds[k]
    current=evaluate(all_rows,all_current);trials=[]
    for name in protocol['variants']:
        score=evaluate(all_rows,all_preds[name],all_current);a=score['totals'];b=current['totals'];guards={'mean_f1':score['mean_spell_f1']>current['mean_spell_f1'],'misses':a['missed']<=b['missed'],'merges':a['merges']<=b['merges'],'false_positives':a['false_positive']<=b['false_positive'],'prior_conditions':all(f['variants'][name][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'tracker_noise':all(f['variants'][name]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),'dead_f1':all(f['variants'][name][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS)}
        trials.append({'name':name,'scores':score,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None,'best_unrestricted':max(trials,key=lambda t:t['scores']['mean_spell_f1']),'fresh_test_results':None,'promoted_to_default':False,'limitations':['Two repeatedly used provider-labelled matches; no independent test or challenge score.','Noise parameters are synthetic examples, not a model of actual challenge error distribution.','Aggregated diagnostic counts repeat the same possessions across stress conditions.','No use of provided-game public score, video, labels or tracking in fitting, configuration or model choice.','Offline processing uses future tracking within each half.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','reference-root','forward-dead','reverse-dead','team-model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,a.reference_root,a.forward_dead,a.reverse_dead,a.team_model,a.out)
if __name__=='__main__':main()
