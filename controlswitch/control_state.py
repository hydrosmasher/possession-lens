"""028 interaction and control-memory features for external-only team correction."""
from pathlib import Path
import argparse,json
import numpy as np
from .control_correction import control_features,swap_control_columns,corrected_probabilities,blend_control
from .model import interaction_features,rolling,shifted,trailing,decode
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,evaluate
from .rnn_runtime import load_frozen,infer_frozen
from .deadball import observation_features,fuse
from .dead_blend import combine,save_combined
from .pipeline import digest
from . import tree_model

VERSION='control-interaction-memory-028'

def state_features(s,pitch,mask=None):
    base=control_features(s,pitch,mask)
    inter=interaction_features(s,ball_missing=mask)
    short=rolling(inter,12);context=np.column_stack([inter,short,rolling(inter,50),shifted(short,25)-shifted(short,-25)])
    anchors=inter[:,[0,5]]
    # Soft proximity+trajectory+player-motion association, not verified touches.
    # Fixed temporal supports:0.48s/2s/8s, independently within a single half.
    memory=np.column_stack([trailing(anchors,w) for w in (12,50,200)]+[trailing(anchors[::-1],w)[::-1] for w in (12,50,200)])
    return np.column_stack([base,context,memory]).astype(np.float32)

def swap_state_columns(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=296:raise ValueError('Expected296 state features')
    inter_order=np.r_[np.arange(5,10),np.arange(5)]
    order=np.r_[*(inter_order+10*i for i in range(4))]
    memory_order=np.arange(12).reshape(6,2)[:,::-1].ravel()
    return np.column_stack([swap_control_columns(x[:,:244]),x[:,244:284][:,order],x[:,284:][:,memory_order]])

def state_probabilities(model,s,pitch,mask=None):
    x=state_features(s,pitch,mask);a=tree_model.predict(model,x);b=tree_model.predict(model,swap_state_columns(x));return (a+b[:,[1,0,2]])*.5

def evaluate_accuracy(rows,preds,current=None):
    r=evaluate(rows,preds,current)
    for c in CONDITIONS:
        ix=[i for i,x in enumerate(rows) if x['condition']==c];total=sum(len(rows[i]['y']) for i in ix);correct=sum(int((rows[i]['y']==preds[i]).sum()) for i in ix);r[c]['frame_accuracy']=correct/total
    return r

def run(cache_root,control_root,dead_root,team_model,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output folder')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'history':'Adaptive development after027 audit; same two extensively reused Bundesliga matches, not independent testing','competition_data_used':False,'sources':'Existing first-division IDSSE/Sportec,CC BY4.0; provider BallStatus/Possession for supervision only','folds':[{'train':'J03WMX','validation':'J03WN1'},{'train':'J03WN1','validation':'J03WMX'}],'change':'Add pooled proximity/ball-trajectory/player-motion interactions and fixed soft-control temporal supports to017 correction features','features':'244 old features +40 interaction/context +12 past/future soft-association averages =296; no identities or labels in inputs','memory_support_frames':[12,50,200],'training':'Original017 clean/short/long input augmentation and team swaps; known live labels every5th frame;100 boosting iterations/15leaves/lr.08/minleaf40/L2=10/no earlystop; same seed117 as017','fixed':'RNN,019 dead estimator,correction mix.75,teamcost12,stopcost3,deadweight.75','variants':['state','half_blend'],'half_blend':'50/50 old/new control-expert log-odds blend; old tree indices stay within first244 columns; represented as200 numeric trees over296 features','selection':'Highest four-condition spellF1 passing no total misses/merges/FP increase; prior clean/ball-gap fold F1 decline<=.01; each tracker_noise fold no decline; deadF1 decline<=.005 each condition/fold; pooled clean frame accuracy no decline.95% is reported as a target, never fabricated.','hashes':{'team':digest(team_model),'old_control':[digest(Path(control_root)/f'fold-{i}/control-estimator.npz') for i in (0,1)],'dead':[digest(Path(dead_root)/f'fold-{i}/dead-blend-0.5.npz') for i in (0,1)]}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');backbone=load_frozen(team_model);folds=[];all_rows=[];all_current=[];all_preds={k:[] for k in protocol['variants']}
    for fold,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('Fold',fold,train,'->',val,flush=True);fd=out/f'fold-{fold}';fd.mkdir();records={}
        for mid in (train,val):
            f=list(Path(cache_root).glob(mid+'-*.npz'))
            if len(f)!=1:raise ValueError('Ambiguous cache')
            with np.load(f[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
            if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong external match')
        xs=[];ys=[]
        for half in (0,1):
            original=period_arrays(records[train],half);ix=np.flatnonzero(original['y']<2)[::5]
            for c in CONDITIONS[:3]:
                s,mask=condition(original,c,half,True);x=state_features(s,records[train]['pitch'],mask);xs.extend([x[ix],swap_state_columns(x)[ix]]);ys.extend([original['y'][ix],1-original['y'][ix]])
        x=np.concatenate(xs);y=np.concatenate(ys);head=HistGradientBoostingClassifier(max_iter=100,learning_rate=.08,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=117)
        print('Fitting296-feature control',x.shape,flush=True)
        with threadpool_limits(limits=1):head.fit(x,y)
        tree_model.save(fd/'control-state.npz',head,'hist_boost');state=tree_model.load(fd/'control-state.npz')
        if not np.allclose(head.predict_proba(x[:10000]),tree_model.predict(state,x[:10000])[:,:2],atol=1e-9):raise AssertionError('Export mismatch')
        n=len(y);del x,y,xs,ys;old=tree_model.load(Path(control_root)/f'fold-{fold}/control-estimator.npz');padded={'meta':{**old['meta'],'features':296},'trees':old['trees']};save_combined(fd/'control-half-blend.npz',combine(padded,state,.5));blended=tree_model.load(fd/'control-half-blend.npz');dead=tree_model.load(Path(dead_root)/f'fold-{fold}/dead-blend-0.5.npz');rows=[];current=[];preds={k:[] for k in protocol['variants']}
        for half in (0,1):
            original=period_arrays(records[val],half);pitch=records[val]['pitch']
            for c in CONDITIONS:
                print('Validation',fold,half,c,flush=True);s,mask=condition(original,c,half,False);rows.append({'y':original['y'],'condition':c});_,rnn=infer_frozen(backbone,s,pitch,mask);d=tree_model.predict(dead,observation_features(s,pitch,mask))[:,1];old_h=corrected_probabilities(old,s,pitch,mask);current.append(decode(fuse(blend_control(rnn,old_h,.75),d,.75),12,stop_penalty=3))
                x=state_features(s,pitch,mask)
                # Prefix equivalence validates padding the old expert into the new schema.
                np.testing.assert_array_equal(tree_model.predict(padded,x),tree_model.predict(old,x[:,:244]))
                for name,model in [('state',state),('half_blend',blended)]:
                    a=tree_model.predict(model,x);b=tree_model.predict(model,swap_state_columns(x));h=(a+b[:,[1,0,2]])*.5;preds[name].append(decode(fuse(blend_control(rnn,h,.75),d,.75),12,stop_penalty=3))
                with threadpool_limits(limits=1):native=(head.predict_proba(x)+head.predict_proba(swap_state_columns(x))[:,[1,0]])*.5
                if not np.allclose(state_probabilities(state,s,pitch,mask)[:,:2],native,atol=1e-9):raise AssertionError('Validation export mismatch')
        result={'train':train,'validation':val,'training_examples':n,'state_sha256':digest(fd/'control-state.npz'),'current':evaluate_accuracy(rows,current),'variants':{k:evaluate_accuracy(rows,p,current) for k,p in preds.items()},'export_matches_native':True};folds.append(result);(fd/'report.json').write_text(json.dumps(result,indent=2)+'\n');all_rows+=rows;all_current+=current
        for k in preds:all_preds[k]+=preds[k]
        for i,r in enumerate(rows):np.savez_compressed(fd/f'validation-{i}.npz',reference=r['y'],current=current[i],**{k:p[i] for k,p in preds.items()})
    current=evaluate_accuracy(all_rows,all_current);trials=[]
    for k in protocol['variants']:
        score=evaluate_accuracy(all_rows,all_preds[k],all_current);a=score['totals'];b=current['totals'];guards={'spell_f1':score['mean_spell_f1']>current['mean_spell_f1'],'misses':a['missed']<=b['missed'],'merges':a['merges']<=b['merges'],'false_positives':a['false_positive']<=b['false_positive'],'prior_conditions':all(f['variants'][k][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'tracker_noise':all(f['variants'][k]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),'dead_f1':all(f['variants'][k][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS),'clean_frame_accuracy':score['clean']['frame_accuracy']>=current['clean']['frame_accuracy']}
        trials.append({'variant':k,'scores':score,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None,'fresh_test_results':None,'promoted_to_default':False,'limitations':['Two heavily reused provider-labelled matches; not independent accuracy or a competition percentile.','Feature windows and future memories are offline soft association, not confirmed touches.','95% under clean tracking does not imply95% under missing/noisy data.','No provided-game data, video, labels or public score used for this experiment.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['clean']['frame_accuracy'],current['totals'])
    for t in trials:print(t['variant'],t['scores']['mean_spell_f1'],t['scores']['clean']['frame_accuracy'],t['scores']['totals'],t['guards'])
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','control-root','dead-root','team-model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,a.control_root,a.dead_root,a.team_model,a.out)
if __name__=='__main__':main()
