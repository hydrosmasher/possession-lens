"""039: add weak known-team external examples to live/dead fitting."""
from pathlib import Path
import argparse,json
import numpy as np
from .ranked_pipeline import make_bundle,infer
from .ranking_experiment import asset_paths
from .skillcorner_europe import load_match
from .deadball import observation_features
from .boundary_had import boundary_observations,adaptive_decode
from .boundary_alignment import eligible
from .control_state import evaluate_accuracy
from .model import probabilities
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest
from . import tree_model

TRAIN_IDS=('4039','3749','3518','2068','2440')
SPEC={'name':'joint_relative_quality1_boundary_full','control':'relative','dead':'quality','cap':1.,'boundary':'balanced','strength':1.}
MIXES=(.25,.5)
def bundle037(fi,roots,control_root):
    paths=asset_paths(fi,SPEC,roots);paths['control']=Path(control_root)/f'fold-{fi}/relative_motion.npz';paths['boundary']=Path(roots['aligned_boundary'])/f'fold-{fi}/boundary-aligned.npz';return make_bundle(SPEC,paths),paths

def scaled_dead(q):
    q=np.clip(q,1e-6,1-1e-6);return 1/(1+np.exp(-.75*np.log(q/(1-q))))

def blend_dead(a,b,mix):
    if not 0<=mix<=1:raise ValueError('Invalid dead blend')
    a=np.clip(a,1e-6,1-1e-6);b=np.clip(b,1e-6,1-1e-6);z=(1-mix)*np.log(a/(1-a))+mix*np.log(b/(1-b));return 1/(1+np.exp(-z))

def predict_set(bundle,head,s,pitch,mask=None,mixes=MIXES):
    base,p=infer(bundle,s,pitch,mask);new=scaled_dead(tree_model.predict(head,observation_features(s,pitch,mask))[:,1]);h=p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12);out={}
    for mix in mixes:
        d=blend_dead(p[:,2],new,mix);q=np.column_stack([h*(1-d[:,None]),d]);event=probabilities(bundle['models']['boundary'],boundary_observations(s,pitch,q,mask))[:,1];out[f'multidead{mix:g}']=adaptive_decode(q,event,1.)
    return base,out

def run(cache_root,raw,roots,control_root,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    out=Path(out)
    if out.exists():raise ValueError('Choose new output folder')
    prior=json.loads((Path(control_root)/'report.json').read_text())
    if not prior['recommended_for_freeze']:raise ValueError('Expected frozen037 reference')
    out.mkdir(parents=True);protocol={'version':'multisource-dead-039','reference':'frozen037','hypothesis':'034 reveals cross-provider false-dead predictions. Add partial known-team examples from the existing five SkillCorner training matches as weak live negatives, never unknown-to-dead conversion.',
     'sources':'First-division Sportec/IDSSE WMX/WN1 CC BY4.0 strong BallStatus; historical MIT SkillCorner4039/3749/3518/2068/2440 partial possession groups. PSG2269 excluded. No new or fresh match claim.',
     'weak_label_caveat':'A team-possession group is not verified live/dead truth. Treat known-team frames as weak negatives with aggregate25% training weight; Sportec remains75%. No SkillCorner positive dead labels invented.',
     'training':'Existing240 team-invariant observation features. Four fixed conditions. Sportec every5th known frame; SkillCorner every25th known-team frame.100trees/15leaves/lr.08/minleaf40/L2=10/seed139/no earlystop. Weight normalisation preserves total example weight and source75/25mass.',
     'variants':'Frozen037 dead odds blended with new head scaled at.75: logodds mixture .25 or.5. Control/RNN/boundary unchanged.',
     'selection':'Official mean four-condition spellF1 gain>=.003 over037; clean/gap per-fold decline<=.01, noise fold no decline, deadF1 decline<=.005. Highest eligible mean; report FP trade-off.',
     'audit':'Freeze one winner and hashes before new-draw audit seeds2114/2314/2718. Only winner/037 decoded. Audit mean must improve, otherwise retain037 without runner-up selection.',
     'history':'Adaptive after viewed034; all validation games heavily reused. No independent-test or competition-percentile claim. No challenge inputs/labels/video/public score used.',
     'competition_data_used':False,'competition_percentile_known':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');weakx=[];weak_audit=[]
    for mid in TRAIN_IDS:
        print('Weak source',mid,flush=True);parts,audit=load_match(raw,mid);weak_audit.append({'match_id':mid,'audit':audit})
        for half,(seq,pitch) in enumerate(parts):
            ix=np.flatnonzero(seq['y']<2)[::25]
            for c in CONDITIONS:
                print('Weak features',mid,half,c,flush=True);s,mask=condition(seq,c,half,True);weakx.append(observation_features(s,pitch,mask)[ix])
    weakx=np.concatenate(weakx);np.savez_compressed(out/'weak-training-features.npz',x=weakx);records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong external match')
    names=[f'multidead{m:g}' for m in MIXES];folds=[];all_rows=[];all_base=[];all_preds={k:[] for k in names}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();bundle,paths=bundle037(fi,roots,control_root);xs=[];ys=[]
        for half in (0,1):
            seq=period_arrays(records[train],half);ix=np.flatnonzero(seq['y']<3)[::5]
            for c in CONDITIONS:
                s,mask=condition(seq,c,half,True);xs.append(observation_features(s,records[train]['pitch'],mask)[ix]);ys.append((seq['y'][ix]==2).astype(np.int8))
        strongx=np.concatenate(xs);strongy=np.concatenate(ys);n=len(strongy);m=len(weakx);x=np.concatenate([strongx,weakx]);y=np.r_[strongy,np.zeros(m,np.int8)];total=n+m;weights=np.r_[np.full(n,total*.75/n),np.full(m,total*.25/m)];head=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.08,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=139)
        print('Fit multi-source dead',fi,x.shape,flush=True)
        with threadpool_limits(limits=1):head.fit(x,y,sample_weight=weights)
        tree_model.save(fd/'multisource-dead.npz',head,'hist_boost');model=tree_model.load(fd/'multisource-dead.npz');np.testing.assert_allclose(head.predict_proba(x[:10000]),tree_model.predict(model,x[:10000])[:,:2],atol=1e-9);del x,y,weights,xs,ys,strongx,strongy;rows=[];base=[];preds={k:[] for k in names}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);a,b=predict_set(bundle,model,s,records[val]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});base.append(a)
                for k in names:preds[k].append(b[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=a,**b)
        f={'train':train,'validation':val,'training_audit':{'strong_examples':n,'weak_negative_examples':m,'strong_dead_examples':int(sum((period_arrays(records[train],h)['y']==2).sum() for h in (0,1))),'source_weight_mass':[.75,.25]},'reference_hashes':{k:digest(p) for k,p in paths.items()},'new_head_hash':digest(fd/'multisource-dead.npz'),'current':evaluate_accuracy(rows,base),'candidates':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base
        for k in names:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for mix,name in zip(MIXES,names):
        score=evaluate_accuracy(all_rows,all_preds[name],all_base);guards=eligible(folds,name,current,score);trials.append({'name':name,'mix':mix,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'head_hashes':[digest(out/f'fold-{i}/multisource-dead.npz') for i in (0,1)],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            bundle,_=bundle037(fi,roots,control_root);model=tree_model.load(out/f'fold-{fi}/multisource-dead.npz')
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),2114+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),2314+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,2718+half)
                    a,b=predict_set(bundle,model,s,records[mid]['pitch'],mask,mixes=(selected['mix'],));rows.append({'y':seq['y'],'condition':c});base.append(a);preds.append(b[selected['name']])
        audit={'current':evaluate_accuracy(rows,base),'candidate':evaluate_accuracy(rows,preds,base),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'weak_source_audit':weak_audit,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','raw','roots','control-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,a.raw,json.loads(a.roots.read_text()),a.control_root,a.out)
if __name__=='__main__':main()
