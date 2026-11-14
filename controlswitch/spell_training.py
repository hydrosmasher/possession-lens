"""032: external spell-balanced H/A fitting with tracker-noise augmentation."""
from pathlib import Path
import argparse,json
import numpy as np
from .control_state import state_features,swap_state_columns,evaluate_accuracy
from .ranked_pipeline import make_bundle,infer
from .ranking_experiment import asset_paths
from .boundary_alignment import eligible
from .boundary_had import boundary_observations,adaptive_decode
from .model import probabilities
from .control_correction import blend_control
from .europe_context import observation_probabilities
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .external import period_arrays
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest
from . import tree_model

SPEC={'name':'joint_state_quality1_boundary_full','control':'state','dead':'quality','cap':1.,'boundary':'balanced','strength':1.}
VARIANTS=('noise_uniform','noise_spell_balanced')

def spell_weights(y):
    """Train labels only: bounded inverse-sqrt live-run duration, unit live mean."""
    y=np.asarray(y)
    if y.ndim!=1 or not len(y) or not np.isin(y,[0,1,2,3]).all():raise ValueError('Expected H/A/D/X training labels')
    edges=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)];lengths=[b-a for a,b in zip(edges[:-1],edges[1:]) if y[a]<2];w=np.zeros(len(y),float)
    if not lengths:return w
    median=float(np.median(lengths))
    for a,b in zip(edges[:-1],edges[1:]):
        if y[a]<2:w[a:b]=np.clip(np.sqrt(median/(b-a)),.5,3)
    w[y<2]/=w[y<2].mean();return w

def bundle031(fold,roots):
    paths=asset_paths(fold,SPEC,roots);paths['boundary']=Path(roots['aligned_boundary'])/f'fold-{fold}/boundary-aligned.npz';return make_bundle(SPEC,paths),paths

def variant_predictions(bundle,models,s,pitch,mask=None):
    base,p=infer(bundle,s,pitch,mask);team=observation_probabilities(bundle['models']['team']['model'],s,pitch,mask,context=True);x=state_features(s,pitch,mask);sx=swap_state_columns(x);d=p[:,2];out={}
    for name,head in models.items():
        a=tree_model.predict(head,x);b=tree_model.predict(head,sx);h=blend_control(team,(a+b[:,[1,0,2]])*.5,.75);q=np.column_stack([h[:,:2]*(1-d[:,None]),d]);event=probabilities(bundle['models']['boundary'],boundary_observations(s,pitch,q,mask))[:,1];out[name]=adaptive_decode(q,event,1.)
    return base,out

def run(cache_root,roots,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    out=Path(out)
    if out.exists():raise ValueError('Choose a new run folder')
    out.mkdir(parents=True)
    protocol={'version':'spell-training-032','reference':'frozen031','hypothesis':'Uniform frame loss overweights long possessions. Add tracker-noise control examples and compare uniform versus inverse-sqrt spell-duration weights.',
     'training':'Existing296features, original100trees/15leaves/lr.08/minleaf40/L2=10/no earlystop,seed132. Four fixed training conditions, known-live every5th frame, exact team swaps. Weight median computed within each training half only; clip raw inverse-sqrt weights[.5,3], then unit live mean.',
     'variants':list(VARIANTS),'frozen':'RNN,014/018/029 dead/quality,031 boundary head and full-strength decoder; correction blend.75; no new boundary refit',
     'history':'Two heavily reused first-division Bundesliga matches, opposite-match development. Not independent tests. No challenge inputs accepted.',
     'selection':'Gain>=.003 mean official four-condition spellF1; every clean/gap fold decline<=.01, noise fold no decline, deadF1 decline<=.005. Highest eligible mean; report FP trade-off. No post-result search.',
     'audit':'Freeze winner and hashes before different-draw audit, seeds1514/1714/2118. Only winner/031 evaluated; retain031 if audit mean does not improve. No runner-up selection.',
     'competition_data_used':False,'competition_percentile_known':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous external cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
    folds=[];all_rows=[];all_base=[];all_preds={k:[] for k in VARIANTS}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();bundle,paths=bundle031(fi,roots);xs=[];ys=[];ws=[]
        for half in (0,1):
            seq=period_arrays(records[train],half);ix=np.flatnonzero(seq['y']<2)[::5];w=spell_weights(seq['y'])
            for c in CONDITIONS:
                print('Training features',fi,half,c,flush=True);s,mask=condition(seq,c,half,True);x=state_features(s,records[train]['pitch'],mask);xs.extend([x[ix],swap_state_columns(x)[ix]]);ys.extend([seq['y'][ix],1-seq['y'][ix]]);ws.extend([w[ix],w[ix]])
        x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);models={};train_audit={'examples':len(y),'features':x.shape[1],'sample_weight_min':float(w.min()),'sample_weight_max':float(w.max()),'class_counts':np.bincount(y,minlength=2).tolist()}
        for name in VARIANTS:
            print('Fit',fi,name,x.shape,flush=True);head=HistGradientBoostingClassifier(max_iter=100,learning_rate=.08,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=132)
            with threadpool_limits(limits=1):head.fit(x,y,sample_weight=w if name=='noise_spell_balanced' else None)
            tree_model.save(fd/(name+'.npz'),head,'hist_boost');models[name]=tree_model.load(fd/(name+'.npz'));np.testing.assert_allclose(head.predict_proba(x[:10000]),tree_model.predict(models[name],x[:10000])[:,:2],atol=1e-9)
        del x,y,w,xs,ys,ws;rows=[];base=[];preds={k:[] for k in VARIANTS}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);a,b=variant_predictions(bundle,models,s,records[val]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});base.append(a)
                for k in VARIANTS:preds[k].append(b[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=a,**b)
        f={'train':train,'validation':val,'training_audit':train_audit,'frozen_component_hashes':{k:digest(p) for k,p in paths.items()},'current':evaluate_accuracy(rows,base),'candidates':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base
        for k in VARIANTS:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for name in VARIANTS:
        score=evaluate_accuracy(all_rows,all_preds[name],all_base);guards=eligible(folds,name,current,score);trials.append({'name':name,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;freeze={'selected':selected,'model_hashes':[{name:digest(out/f'fold-{fi}'/(name+'.npz')) for name in VARIANTS} for fi in (0,1)],'before_new_draw_audit':True};(out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            bundle,_=bundle031(fi,roots);models={selected['name']:tree_model.load(out/f'fold-{fi}'/(selected['name']+'.npz'))}
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),1514+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),1714+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,2118+half)
                    a,b=variant_predictions(bundle,models,s,records[mid]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});base.append(a);preds.append(b[selected['name']])
        audit={'current':evaluate_accuracy(rows,base),'candidate':evaluate_accuracy(rows,preds,base),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-root',required=True,type=Path);p.add_argument('--roots',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.out)
if __name__=='__main__':main()
