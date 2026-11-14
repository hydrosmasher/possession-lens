"""024: expand control-correction training using existing licensed European pool."""
from pathlib import Path
import argparse,json,gc
import numpy as np
from .control_correction import control_features,corrected_probabilities,blend_control
from .tracker_robustness import CONDITIONS,condition,evaluate
from .skillcorner_europe import load_match
from .external import period_arrays
from .rnn_runtime import load_frozen,infer_frozen
from .deadball import observation_features,fuse
from .model import decode
from .experiment import missing_mask
from .gap_possession import long_mask
from .dead_blend import combine,save_combined
from .pipeline import digest
from . import tree_model

VERSION='multisource-control-024'
TRAIN_SKILL=('4039','3749','3518','2068','2440')

def skill_training(root):
    xs=[];ys=[];audit=[]
    for mid in TRAIN_SKILL:
        print('Training pool',mid,flush=True);parts,_=load_match(root,mid)
        for half,(s,pitch) in enumerate(parts):
            known=np.flatnonzero(s['y']<2);n=min(1500,len(known));ix=known[np.linspace(0,len(known)-1,n,dtype=int)]
            for c in ('clean','masked','long_gap'):
                mask=None if c=='clean' else missing_mask(len(s['y']),114+half) if c=='masked' else long_mask(len(s['y']),314+half)
                for swap in (False,True):
                    xs.append(control_features(s,pitch,mask,swap)[ix]);ys.append(1-s['y'][ix] if swap else s['y'][ix])
            audit.append({'match':mid,'half':half+1,'known_25hz_frames':len(known),'sampled_unique_frames':n,'augmented_examples':n*6})
        del parts;gc.collect()
    return np.concatenate(xs),np.concatenate(ys),audit

def run(skill_root,cache_root,reference_root,dead_root,team_model,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    out=Path(out)
    if out.exists():raise ValueError('Choose a new folder')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'history':'Adaptive research after020–023 on reused two-match Bundesliga validation; not fresh or independent testing','competition_data_used':False,'added_training_matches':list(TRAIN_SKILL),'source':'Existing SkillCorner historical MIT release revision1ed50b775f93075c2a6d00d0d795f13b7b89709e; no2269 validation or viewed SkillCorner test matches added. Sportec first-division Bundesliga CC BY4.0.','added_sampling':'Up to1500 evenly spaced known25Hz H/A frames per half; clean/short/long observations, both team orders. Unknown excluded, never D. No new downloads.','sportec_training':'Original017 natural live every5th frame, both halves, clean/short/long and team swap; opposite match stays validation','estimator':'Same100 boosting iterations,15leaves,learning rate.08,minleaf40,l2=10,no earlystop; seed124','frozen':'Context RNN,019 dead estimator and decoder unchanged; correction mix.75, teamcost12,stopcost3,deadweight.75','variants':['expanded','half_blend'],'half_blend':'50/50 frozen017/new-control log-odds blend, then exact team-order averaging; exported200-tree numeric NPZ','selection':'Highest four-condition meanF1 with no total misses/merges/FP increase; each fold clean/ball-gap loss<=.01, each tracker-noise fold no loss, every deadF1 loss<=.005; no post-result sweep','hashes':{'team':digest(team_model),'reference_control':[digest(Path(reference_root)/f'fold-{i}/control-estimator.npz') for i in (0,1)],'dead':[digest(Path(dead_root)/f'fold-{i}/dead-blend-0.5.npz') for i in (0,1)]}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');sx,sy,sa=skill_training(skill_root);(out/'added-training-audit.json').write_text(json.dumps(sa,indent=2)+'\n');backbone=load_frozen(team_model);folds=[];all_rows=[];all_current=[];all_preds={k:[] for k in protocol['variants']}
    for fold,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('Fold',fold,train,'->',val,flush=True);fd=out/f'fold-{fold}';fd.mkdir();records={}
        for mid in (train,val):
            files=list(Path(cache_root).glob(mid+'-*.npz'))
            if len(files)!=1:raise ValueError('Ambiguous cache')
            with np.load(files[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
            if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
        xs=[sx];ys=[sy]
        for half in (0,1):
            original=period_arrays(records[train],half);ix=np.flatnonzero(original['y']<2)[::5]
            for c in ('clean','masked','long_gap'):
                s,mask=condition(original,c,half,True)
                for swap in (False,True):xs.append(control_features(s,records[train]['pitch'],mask,swap)[ix]);ys.append(1-original['y'][ix] if swap else original['y'][ix])
        x=np.concatenate(xs);y=np.concatenate(ys);head=HistGradientBoostingClassifier(max_iter=100,learning_rate=.08,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=124)
        print('Fitting expanded control',x.shape,flush=True)
        with threadpool_limits(limits=1):head.fit(x,y)
        tree_model.save(fd/'control-expanded.npz',head,'hist_boost');expanded=tree_model.load(fd/'control-expanded.npz')
        if not np.allclose(head.predict_proba(x[:10000]),tree_model.predict(expanded,x[:10000])[:,:2],atol=1e-9):raise AssertionError('Export mismatch')
        count=len(y);del x,y,xs,ys;gc.collect();old=tree_model.load(Path(reference_root)/f'fold-{fold}/control-estimator.npz');dead=tree_model.load(Path(dead_root)/f'fold-{fold}/dead-blend-0.5.npz');save_combined(fd/'control-half-blend.npz',combine(old,expanded,.5));models={'expanded':expanded,'half_blend':tree_model.load(fd/'control-half-blend.npz')};rows=[];current=[];preds={k:[] for k in models}
        for half in (0,1):
            original=period_arrays(records[val],half);pitch=records[val]['pitch']
            for c in CONDITIONS:
                print('Validation',fold,half,c,flush=True);s,mask=condition(original,c,half,False);rows.append({'y':original['y'],'condition':c});_,rnn=infer_frozen(backbone,s,pitch,mask);d=tree_model.predict(dead,observation_features(s,pitch,mask))[:,1];h=corrected_probabilities(old,s,pitch,mask);current.append(decode(fuse(blend_control(rnn,h,.75),d,.75),12,stop_penalty=3))
                for k,model in models.items():preds[k].append(decode(fuse(blend_control(rnn,corrected_probabilities(model,s,pitch,mask),.75),d,.75),12,stop_penalty=3))
                with threadpool_limits(limits=1):native=(head.predict_proba(control_features(s,pitch,mask))+head.predict_proba(control_features(s,pitch,mask,True))[:,[1,0]])*.5
                if not np.allclose(corrected_probabilities(expanded,s,pitch,mask)[:,:2],native,atol=1e-9):raise AssertionError('Validation export mismatch')
        r={'train':train,'validation':val,'training_examples':count,'added_skillcorner_examples':len(sy),'current':evaluate(rows,current),'variants':{k:evaluate(rows,p,current) for k,p in preds.items()},'export_matches_native':True};folds.append(r);(fd/'report.json').write_text(json.dumps(r,indent=2)+'\n');all_rows+=rows;all_current+=current
        for k in models:all_preds[k]+=preds[k]
        for i,row in enumerate(rows):np.savez_compressed(fd/f'validation-{i}.npz',reference=row['y'],current=current[i],**{k:p[i] for k,p in preds.items()})
    current=evaluate(all_rows,all_current);trials=[]
    for k in protocol['variants']:
        score=evaluate(all_rows,all_preds[k],all_current);a=score['totals'];b=current['totals'];guards={'mean_f1':score['mean_spell_f1']>current['mean_spell_f1'],'misses':a['missed']<=b['missed'],'merges':a['merges']<=b['merges'],'false_positives':a['false_positive']<=b['false_positive'],'prior_conditions':all(f['variants'][k][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'tracker_noise':all(f['variants'][k]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),'dead_f1':all(f['variants'][k][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS)}
        trials.append({'variant':k,'scores':score,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];report={'protocol':protocol,'added_training_audit':sa,'folds':folds,'current':current,'trials':trials,'selected':max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None,'best_unrestricted':max(trials,key=lambda t:t['scores']['mean_spell_f1']),'fresh_test_results':None,'promoted_to_default':False,'limitations':['Mixed providers have different possession semantics and tracking errors.','Same two Bundesliga matches reused extensively; no fresh independent test.','Added SkillCorner data are existing training games, not new matches or test data.','Synthetic noise may not reflect challenge tracking.','No challenge data, video, labels or public score used.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'])
    for t in trials:print(t['variant'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'])
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('skill-root','cache-root','reference-root','dead-root','team-model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.skill_root,a.cache_root,a.reference_root,a.dead_root,a.team_model,a.out)
if __name__=='__main__':main()
