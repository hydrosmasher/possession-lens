"""External-only team-control correction017; frozen dead-ball model and decoder."""
from pathlib import Path
import argparse,json
import numpy as np
from .gap_possession import CONDITIONS,long_mask,repair_ball,evaluate
from .deadball import observation_features,fuse
from .model import features,decode
from .external import period_arrays
from .rnn_runtime import load_frozen,infer_frozen
from .experiment import missing_mask
from .pipeline import digest,pm
from . import tree_model

VERSION='hybrid-control-had-017'

def control_features(sequence,pitch,mask=None,swap=False):
    original,seen,_=repair_ball(sequence,mask,0);repaired,_,filled=repair_ball(sequence,mask,50)
    n=len(seen);idx=np.arange(n);prev=np.maximum.accumulate(np.where(seen,idx,-625));nxt=np.minimum.accumulate(np.where(seen,idx,n+625)[::-1])[::-1]
    return np.column_stack([features(original,pitch,swap=swap),features(repaired,pitch,swap=swap),seen,filled,np.minimum(idx-prev,625)/625,np.minimum(nxt-idx,625)/625]).astype(np.float32)

def blend_control(rnn,correction,mix):
    a=np.asarray(rnn,float);b=np.asarray(correction,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(mix) or not 0<=mix<=1 or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():raise ValueError('Invalid control probability blend')
    a=a[:,:2]/np.maximum(a[:,:2].sum(1,keepdims=True),1e-12);b=b[:,:2]/np.maximum(b[:,:2].sum(1,keepdims=True),1e-12)
    out=(1-mix)*a+mix*b
    return np.column_stack([out,np.zeros(len(out))])

def swap_control_columns(x):
    """Exact team permutation of244 features, avoiding repeated geometry work."""
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=244:raise ValueError('Expected244 control features')
    team_order=np.r_[np.arange(6),np.arange(18,30),np.arange(6,18)]
    columns=np.r_[*(team_order+30*i for i in range(8)),np.arange(240,244)]
    return x[:,columns]

def corrected_probabilities(estimator,sequence,pitch,mask=None):
    x=control_features(sequence,pitch,mask)
    a=tree_model.predict(estimator,x)
    b=tree_model.predict(estimator,swap_control_columns(x))
    return (a+b[:,[1,0,2]])*.5

def load_candidate(settings,team_model,dead_model,control_model):
    settings=json.loads(Path(settings).read_text())
    if settings['version']!=VERSION or digest(team_model)!=settings['team_sha256'] or digest(dead_model)!=settings['dead_sha256'] or digest(control_model)!=settings['control_sha256']:raise ValueError('Hybrid asset/version mismatch')
    if settings['mix'] not in (.25,.5,.75):raise ValueError('Invalid frozen correction mix')
    return {'settings':settings,'team':load_frozen(team_model),'dead':tree_model.load(dead_model),'correction':tree_model.load(control_model)}

def infer_candidate(candidate,sequence,pitch,mask=None):
    if candidate['team'].get('context') and not candidate['team'].get('interaction_context'):
        from .europe_context import observation_probabilities as context_emissions
        team=context_emissions(candidate['team']['model'],sequence,pitch,mask,context=True)
    else:
        _,team=infer_frozen(candidate['team'],sequence,pitch,mask)
    correction=corrected_probabilities(candidate['correction'],sequence,pitch,mask)
    dead=tree_model.predict(candidate['dead'],observation_features(sequence,pitch,mask))[:,1]
    config=candidate['settings'];p=fuse(blend_control(team,correction,config['mix']),dead,config['dead_weight'])
    return decode(p,config['penalty'],stop_penalty=config['stop_penalty']),p

def run(cache_root,forward_reference,reverse_reference,team_model,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    out=Path(out)
    if out.exists():raise ValueError('Choose a new folder')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'folds':[{'train':'J03WMX','validation':'J03WN1'},{'train':'J03WN1','validation':'J03WMX'}],'history':'Both matches reused extensively; follow-up after viewing016. This is exploratory development, not an independent test.','competition_data_used':False,'league':'Bundesliga first division','source':'IDSSE/Sportec CC BY4.0, Bassek et al2025; labels provider BallStatus/BallPossession; no labels in features','seed':117,'train_conditions':list(CONDITIONS),'train_seeds':{'masked':114,'long_gap':314},'validation_seeds':{'masked':214,'long_gap':414},'training_stride':5,'augmentation':'swap H/A features and target; clean, short-gap and long-gap tracking; fit only live provider labels','estimator':{'max_iter':100,'learning_rate':.08,'max_leaf_nodes':15,'min_samples_leaf':40,'l2_regularization':10,'early_stopping':False},'features':'original120+bracketed50-frame-repaired120+4 observation reliability features,244 columns; identity-free','mixes':[.25,.5,.75],'dead_estimator':'unchanged014/015 for each opposite training fold','decoder':{'penalty':12,'stop_penalty':3,'dead_weight':.75},'selection':'best pooled clean/short/long meanF1 subject to no aggregate increase in misses/merges/FP, clean fold F1 no >.01 regression; no new decoder tuning','team_sha256':digest(team_model),'dead_sha256':[digest(forward_reference),digest(reverse_reference)]}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');backbone=load_frozen(team_model);all_rows=[];all_current=[];all_preds={mix:[] for mix in protocol['mixes']};folds=[]
    for fold,(train_id,val_id,dead_path) in enumerate([('J03WMX','J03WN1',forward_reference),('J03WN1','J03WMX',reverse_reference)]):
        print('Fold',fold,train_id,'->',val_id,flush=True);fd=out/f'fold-{fold}';fd.mkdir();records={}
        for mid in (train_id,val_id):
            paths=list(Path(cache_root).glob(mid+'-*.npz'))
            if len(paths)!=1:raise ValueError('Ambiguous/missing external cache')
            with np.load(paths[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
            if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
        xs=[];ys=[]
        for half in (0,1):
            s=period_arrays(records[train_id],half);pitch=records[train_id]['pitch'];ix=np.flatnonzero(s['y']<2)[::5]
            for condition in CONDITIONS:
                mask=None if condition=='clean' else missing_mask(len(s['y']),114+half) if condition=='masked' else long_mask(len(s['y']),314+half)
                for swap in (False,True):
                    xs.append(control_features(s,pitch,mask,swap)[ix]);ys.append(1-s['y'][ix] if swap else s['y'][ix])
        x=np.concatenate(xs);y=np.concatenate(ys);head=HistGradientBoostingClassifier(**protocol['estimator'],random_state=117)
        with threadpool_limits(limits=1):head.fit(x,y)
        tree_model.save(fd/'control-estimator.npz',head,'hist_boost');model=tree_model.load(fd/'control-estimator.npz')
        if not np.allclose(tree_model.predict(model,x[:10000])[:,:2],head.predict_proba(x[:10000]),atol=1e-9):raise AssertionError('Export mismatch')
        del xs,ys,x
        rows=[];current=[];preds={mix:[] for mix in protocol['mixes']};dead_model=tree_model.load(dead_path)
        for half in (0,1):
            s=period_arrays(records[val_id],half);pitch=records[val_id]['pitch']
            for condition in CONDITIONS:
                print('Validation',fold,half,condition,flush=True)
                mask=None if condition=='clean' else missing_mask(len(s['y']),214+half) if condition=='masked' else long_mask(len(s['y']),414+half)
                rows.append({'s':s,'pitch':pitch,'mask':mask,'condition':condition});_,p=infer_frozen(backbone,s,pitch,mask)
                d=tree_model.predict(dead_model,observation_features(s,pitch,mask))[:,1];current.append(decode(fuse(p,d,.75),12,stop_penalty=3))
                correction=corrected_probabilities(model,s,pitch,mask)
                with threadpool_limits(limits=1):native=(head.predict_proba(control_features(s,pitch,mask))+head.predict_proba(control_features(s,pitch,mask,True))[:,[1,0]])*.5
                if not np.allclose(correction[:,:2],native,atol=1e-9):raise AssertionError('Validation export mismatch')
                for mix in protocol['mixes']:preds[mix].append(decode(fuse(blend_control(p,correction,mix),d,.75),12,stop_penalty=3))
        result={'train':train_id,'validation':val_id,'training_samples':len(y),'control_sha256':digest(fd/'control-estimator.npz'),'current':evaluate(rows,current),'variants':{str(mix):evaluate(rows,p,current) for mix,p in preds.items()},'export_matches_native':True}
        (fd/'report.json').write_text(json.dumps(result,indent=2)+'\n');folds.append(result);all_rows+=rows;all_current+=current
        for mix in protocol['mixes']:all_preds[mix]+=preds[mix]
        for i,r in enumerate(rows):np.savez_compressed(fd/f'validation-{i}.npz',reference=r['s']['y'],current=current[i],**{'mix'+str(mix):p[i] for mix,p in preds.items()})
    current=evaluate(all_rows,all_current);trials=[]
    for mix in protocol['mixes']:
        score=evaluate(all_rows,all_preds[mix],all_current);t=score['totals'];r=current['totals'];guards={'mean_f1':score['mean_spell_f1']>current['mean_spell_f1'],'misses':t['missed']<=r['missed'],'merges':t['merges']<=r['merges'],'false_positives':t['false_positive']<=r['false_positive'],'clean_folds':all(f['variants'][str(mix)]['clean']['spell_f1_mean']>=f['current']['clean']['spell_f1_mean']-.01 for f in folds)}
        trials.append({'mix':mix,'scores':score,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];selected=max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'best_unrestricted':max(trials,key=lambda t:t['scores']['mean_spell_f1']),'fresh_test_results':None,'promoted_to_default':False,'limitations':['Reused two-match development with all labels viewed; not unbiased generalisation evidence.','Tracking and supervision share provider; complete ball observations artificially masked.','New team model is trained only on live labels; current dead model remains fixed.','Partial SkillCorner scores and full H/A/D scores are not interchangeable.','No challenge data or video used.','Offline context, interpolation and decoding use future tracking.']}
    report['missing_ball_audit']=audit_missing(out,selected['mix']) if selected else None
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print('MIX',t['mix'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if selected:
        config={'version':VERSION,'team_sha256':digest(team_model),'dead_sha256':digest(forward_reference),'control_sha256':digest(out/'fold-0/control-estimator.npz'),'mix':selected['mix'],**protocol['decoder']}
        (out/'settings.json').write_text(json.dumps(config,indent=2)+'\n')
    return report

def audit_missing(run_dir,mix):
    """Post-selection diagnostics only, never a tuning target."""
    rows=[]
    for fold in (0,1):
        for half in (0,1):
            for k,condition in enumerate(CONDITIONS):
                if condition=='clean':continue
                with np.load(Path(run_dir)/f'fold-{fold}/validation-{half*3+k}.npz',allow_pickle=False) as a:
                    y=a['reference'];current=a['current'];candidate=a['mix'+str(mix)]
                mask=missing_mask(len(y),214+half) if condition=='masked' else long_mask(len(y),414+half)
                gt=pm.possession_spells(y,pm.valid_runs(y));touch=np.array([mask[a:b].any() for a,b,c in gt],bool)
                row={'fold':fold,'half':half+1,'condition':condition,'masked_frames':int(mask.sum()),'reference_spells_touching_gaps':int(touch.sum())}
                for name,pred in [('current',current),('candidate',candidate)]:
                    matched=pm.matched_gt_flags(pm.spell_overlaps(gt,pm.possession_spells(pred,pm.valid_runs(y))),len(gt),.5)
                    row[name]={'missed_spells_touching_gaps':int((~matched&touch).sum()),'masked_frame_accuracy':float(np.mean(y[mask]==pred[mask]))}
                rows.append(row)
    return {'purpose':'Post-selection description; full-sequence spell matches, grouped by whether reference spells touch masked observations. Accuracy is masked-frame-only, not spellF1.','rows':rows,'missed_touching_gaps':{name:sum(r[name]['missed_spells_touching_gaps'] for r in rows) for name in ('current','candidate')}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','forward-reference','reverse-reference','team-model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,a.forward_reference,a.reverse_reference,a.team_model,a.out)
if __name__=='__main__':main()
