"""Missing-ball robustness experiment016. External data only; no challenge input."""
from pathlib import Path
import argparse,json
import numpy as np
from .deadball import observation_features,fuse
from .external import period_arrays
from .rnn_runtime import load_frozen,infer_frozen
from .model import decode
from .experiment import missing_mask,baseline_prediction,aggregate
from .pipeline import pm,digest
from . import tree_model

VERSION='gap-aware-had-016'
CONDITIONS=('clean','masked','long_gap')

def long_mask(n,seed):
    if n<1:raise ValueError('Empty sequence')
    rng=np.random.default_rng(seed);mask=np.zeros(n,bool)
    while mask.mean()<.4:
        a=int(rng.integers(n));mask[a:min(n,a+int(rng.integers(50,301)))]=True
    return mask

def repair_ball(sequence,mask=None,max_gap=50):
    """Short, bracketted linear interpolation, no extrapolation or label inputs.

    Original observation missingness is returned, so the dead-state estimator can
    distinguish observed and reconstructed coordinates. Ends are never repaired.
    """
    if not isinstance(max_gap,(int,np.integer)) or not 0<=max_gap<=50:raise ValueError('Invalid gap limit')
    out={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in sequence.items() if k!='y'}
    ball=out['ball'];speed=out['ball_speed'];n=len(ball)
    if mask is not None:
        if np.asarray(mask).shape!=(n,):raise ValueError('Invalid observation mask')
        ball[mask]=np.nan;speed[mask]=np.nan
    seen=np.isfinite(ball).all(1);filled=np.zeros(n,bool)
    bounds=np.flatnonzero(np.r_[False,~seen,False][1:]!=np.r_[False,~seen,False][:-1])
    for a,b in zip(bounds[::2],bounds[1::2]):
        if a==0 or b==n or b-a>max_gap:continue
        velocity=np.linalg.norm(ball[b]-ball[a-1])*25/(b-a+1)
        if not np.isfinite(velocity) or velocity>50:continue
        fraction=np.arange(1,b-a+1)/(b-a+1)
        ball[a:b]=ball[a-1]+fraction[:,None]*(ball[b]-ball[a-1]);speed[a:b]=velocity;filled[a:b]=True
    return out,seen,filled

def gap_features(sequence,pitch,mask=None):
    original,seen,_=repair_ball(sequence,mask,0)
    repaired,_,filled=repair_ball(sequence,mask,50)
    n=len(seen);idx=np.arange(n);previous=np.maximum.accumulate(np.where(seen,idx,-625));following=np.minimum.accumulate(np.where(seen,idx,n+625)[::-1])[::-1]
    reliability=np.column_stack([seen,filled,np.minimum(idx-previous,625)/625,np.minimum(following-idx,625)/625])
    return np.column_stack([observation_features(original,pitch),observation_features(repaired,pitch),reliability]).astype(np.float32)

def diagnostics(y,pred,reference=None):
    runs=pm.valid_runs(y);gt=pm.possession_spells(y,runs);ps=pm.possession_spells(pred,runs);overlap=pm.spell_overlaps(gt,ps);hits=pm.matched_gt_flags(overlap,len(gt),.5)
    result={'missed':int((~hits).sum()),'false_positive':int(len(ps)-hits.sum()),'merges':sum(sum(j==pi and iou>0 for iou,gi,j in overlap)>1 for pi in range(len(ps))),'short_under_0_4s':sum(b-a<10 for a,b,k in ps)}
    if reference is not None:
        old=pm.matched_gt_flags(pm.spell_overlaps(gt,pm.possession_spells(reference,runs)),len(gt),.5)
        result.update({'newly_recovered':int((hits&~old).sum()),'lost_existing':int((old&~hits).sum())})
    return result

def evaluate(rows,preds,reference=None):
    result={}
    for condition in CONDITIONS:
        indices=[i for i,r in enumerate(rows) if r['condition']==condition]
        score=aggregate([(rows[i]['s']['y'],preds[i]) for i in indices])
        ds=[diagnostics(rows[i]['s']['y'],preds[i],reference[i] if reference is not None else None) for i in indices]
        score['diagnostics']={k:sum(d[k] for d in ds) for k in ds[0]}
        y=np.concatenate([rows[i]['s']['y'] for i in indices]);p=np.concatenate([preds[i] for i in indices]);tp=int(((y==2)&(p==2)).sum());score['dead_f1']=2*tp/max(int((y==2).sum()+(p==2).sum()),1)
        result[condition]=score
    result['mean_spell_f1']=float(np.mean([result[c]['spell_f1_mean'] for c in CONDITIONS]))
    result['totals']={k:sum(result[c]['diagnostics'][k] for c in CONDITIONS) for k in result['clean']['diagnostics']}
    return result

def run(cache_root,forward_reference,reverse_reference,team_model,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    out=Path(out)
    if out.exists():raise ValueError('Choose new experiment folder')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'folds':[{'train':'J03WMX','validation':'J03WN1'},{'train':'J03WN1','validation':'J03WMX'}],'history':'Both matches reused development; not independent testing or unbiased cross-validation','league':'Bundesliga first division','source':'IDSSE/Sportec Bassek et al2025, CC BY4.0; original XML/cache hashes in014/015 audits','train_conditions':['clean','20-100-frame random ball gaps to40%','50-300-frame random ball gaps to40%'],'train_seed_short':114,'train_seed_long':314,'validation_seed_short':214,'validation_seed_long':414,'training_stride':5,'estimator':{'max_iter':100,'learning_rate':.08,'max_leaf_nodes':15,'min_samples_leaf':40,'l2_regularization':10,'early_stopping':False},'decoder_frozen':{'penalty':12,'stop_penalty':3,'dead_weight':.75},'variants':[{'name':'gap_head','team_repair_frames':0},{'name':'gap_head_repair25','team_repair_frames':25},{'name':'gap_head_repair50','team_repair_frames':50}],'repair':'Only bracketed gaps<=limit within half, anchor-implied ball speed<=50m/s. No extrapolation; original visibility retained in dead features. No reference labels used in repairs.','features':'original240 + repaired240 invariant geometry/context, plus observed/filled masks and capped previous/next observation ages;484 total','selection':'Highest mean three-condition spellF1 across both folds among guarded variants. Mean gain; clean each fold no >0.01 regression; aggregate misses, false positives and merges no increase; ball-gap deadF1 each fold no >0.005 regression. No retuning after outcome.','team_checkpoint_sha256':digest(team_model),'reference_dead_sha256':[digest(forward_reference),digest(reverse_reference)],'competition_data_used':False,'fresh_test_results':None}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    backbone=load_frozen(team_model);fold_results=[];all_rows=[];all_current=[];all_preds={v['name']:[] for v in protocol['variants']}
    for fold,(train_id,val_id,refpath) in enumerate([('J03WMX','J03WN1',forward_reference),('J03WN1','J03WMX',reverse_reference)]):
        print('Fold',fold,train_id,'->',val_id,flush=True);fold_out=out/f'fold-{fold}';fold_out.mkdir();records={}
        for mid in (train_id,val_id):
            paths=list(Path(cache_root).glob(mid+'-*.npz'))
            if len(paths)!=1:raise ValueError('Ambiguous/missing audited external cache')
            with np.load(paths[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
            if str(records[mid]['match_id'])!=mid:raise ValueError('Match identity mismatch')
        xs=[];ys=[]
        for half in (0,1):
            s=period_arrays(records[train_id],half);pitch=records[train_id]['pitch']
            for condition in CONDITIONS:
                mask=None if condition=='clean' else missing_mask(len(s['y']),114+half) if condition=='masked' else long_mask(len(s['y']),314+half)
                x=gap_features(s,pitch,mask);ix=np.flatnonzero(s['y']<3)[::5];xs.append(x[ix]);ys.append((s['y'][ix]==2).astype(int))
        x=np.concatenate(xs);y=np.concatenate(ys);head=HistGradientBoostingClassifier(**protocol['estimator'],random_state=116)
        with threadpool_limits(limits=1):head.fit(x,y)
        tree_model.save(fold_out/'dead-estimator.npz',head,'hist_boost');exported=tree_model.load(fold_out/'dead-estimator.npz')
        if not np.allclose(head.predict_proba(x[:10000]),tree_model.predict(exported,x[:10000])[:,:2],atol=1e-9):raise AssertionError('Training export mismatch')
        current_head=tree_model.load(refpath);rows=[];current=[];baseline=[];preds={v['name']:[] for v in protocol['variants']}
        for half in (0,1):
            s=period_arrays(records[val_id],half);pitch=records[val_id]['pitch']
            for condition in CONDITIONS:
                print('Validation',fold,half,condition,flush=True)
                mask=None if condition=='clean' else missing_mask(len(s['y']),214+half) if condition=='masked' else long_mask(len(s['y']),414+half)
                rows.append({'s':s,'pitch':pitch,'condition':condition,'mask':mask});_,team=infer_frozen(backbone,s,pitch,mask)
                d=tree_model.predict(current_head,observation_features(s,pitch,mask))[:,1];current.append(decode(fuse(team,d,.75),12,stop_penalty=3));baseline.append(baseline_prediction(s,50,mask))
                x=gap_features(s,pitch,mask)
                with threadpool_limits(limits=1):q=head.predict_proba(x)[:,1]
                if not np.allclose(q,tree_model.predict(exported,x)[:,1],atol=1e-9):raise AssertionError('Validation export mismatch')
                for v in protocol['variants']:
                    p=team
                    if v['team_repair_frames']:
                        repaired,_,_=repair_ball(s,mask,v['team_repair_frames']);_,p=infer_frozen(backbone,repaired,pitch)
                    preds[v['name']].append(decode(fuse(p,q,.75),12,stop_penalty=3))
        result={'train':train_id,'validation':val_id,'training_samples':len(y),'model_sha256':digest(fold_out/'dead-estimator.npz'),'baseline':evaluate(rows,baseline),'current':evaluate(rows,current),'variants':{name:evaluate(rows,p,current) for name,p in preds.items()},'export_matches_native':True}
        fold_results.append(result);(fold_out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        all_rows+=rows;all_current+=current
        for name in all_preds:all_preds[name]+=preds[name]
        for i,r in enumerate(rows):np.savez_compressed(fold_out/f'validation-{i}.npz',reference=r['s']['y'],current=current[i],**{name:p[i] for name,p in preds.items()})
    current=evaluate(all_rows,all_current);trials=[]
    for variant in protocol['variants']:
        name=variant['name'];score=evaluate(all_rows,all_preds[name],all_current);t=score['totals'];r=current['totals']
        guards={'mean_f1':score['mean_spell_f1']>current['mean_spell_f1'],'misses':t['missed']<=r['missed'],'merges':t['merges']<=r['merges'],'false_positives':t['false_positive']<=r['false_positive'],'clean_folds':all(f['variants'][name]['clean']['spell_f1_mean']>=f['current']['clean']['spell_f1_mean']-.01 for f in fold_results),'dead_gap_folds':all(f['variants'][name][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in fold_results for c in ('masked','long_gap'))}
        trials.append({**variant,'scores':score,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];selected=max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None
    report={'protocol':protocol,'folds':fold_results,'current':current,'trials':trials,'selected':selected,'best_unrestricted':max(trials,key=lambda t:t['scores']['mean_spell_f1']),'fresh_test_results':None,'promoted_to_default':False,'limitations':['Repeated two-match external development; shared clubs/provider and decoder ancestry limit inference.','Random missingness stress cannot represent all real ball errors or biased occlusions.','Short interpolation can smooth real airborne/deflected trajectories; observed mask is retained.','Merge diagnostic counts geometric overlap, not independently verified events.','No challenge data, video, labels or output used.','Offline features use future observations within each half.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','forward-reference','reverse-reference','team-model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,a.forward_reference,a.reverse_reference,a.team_model,a.out)
if __name__=='__main__':main()
