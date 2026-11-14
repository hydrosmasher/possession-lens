"""033: nonlinear external turnover evidence for short possession recovery."""
from pathlib import Path
import argparse,json
import numpy as np
from .spell_training import bundle031
from .ranked_pipeline import infer
from .boundary_had import boundary_observations,targets,adaptive_decode
from .boundary_alignment import eligible
from .model import probabilities
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest
from . import tree_model

VARIANTS=[{'name':'linear_soft','kind':'linear','floor':.05},{'name':'tree_default','kind':'tree','floor':.25},{'name':'tree_soft','kind':'tree','floor':.05}]

def floor_decode(p,score,floor=.25,penalty=12.,stop_penalty=3.):
    # Same balanced full-strength Viterbi as031, with an explicit event floor.
    p=np.asarray(p);score=np.asarray(score)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or score.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(score).all() or (p<0).any() or (p.sum(1)<=0).any() or ((score<0)|(score>1)).any():raise ValueError('Invalid emissions/events')
    if not np.isfinite(floor) or not .01<=floor<=1 or not np.isfinite(penalty) or penalty<0 or not np.isfinite(stop_penalty) or stop_penalty<0:raise ValueError('Invalid transition controls')
    costs=penalty*np.clip(2-2*score,floor,2);logp=np.log(np.maximum(p,1e-8));back=np.zeros((len(p),3),np.int8);values=logp[0].copy()
    for i in range(1,len(p)):
        matrix=np.array([[0,costs[i],stop_penalty],[costs[i],0,stop_penalty],[stop_penalty,stop_penalty,0]]);choices=values[:,None]-matrix;back[i]=choices.argmax(0);values=choices[back[i],np.arange(3)]+logp[i]
    labels=np.empty(len(p),np.int8);labels[-1]=values.argmax()
    for i in range(len(p)-1,0,-1):labels[i-1]=back[i,labels[i]]
    return labels

def predict_variants(bundle,tree,s,pitch,mask=None):
    base,p=infer(bundle,s,pitch,mask);x=boundary_observations(s,pitch,p,mask);scores={'linear':probabilities(bundle['models']['boundary'],x)[:,1],'tree':tree_model.predict(tree,x)[:,1]};return base,{v['name']:floor_decode(p,scores[v['kind']],v['floor']) for v in VARIANTS}

def run(cache_root,roots,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output folder')
    out.mkdir(parents=True)
    protocol={'version':'nonlinear-boundary-033','reference':'frozen031','hypothesis':'Duration audit shows low short-spell recovery. Test nonlinear existing116-feature event interactions and a smaller event-evidence transition floor, with no hard duration minimum.',
      'variants':VARIANTS,'training':'Same balanced031 examples/indices from external training match only;100boosting iterations/7leaves/lr.05/minleaf80/L2=20/seed133/no earlystop; no new features, no validation-based earlystop.',
      'selection':'Mean official spellF1 gain>=.003 over031; each clean/gap fold decline<=.01; noise fold no decline; deadF1 decline<=.005. Highest eligible mean. Report false-positive trade-off.',
      'audit':'Freeze/hash one winner before new-draw audit; seeds1614/1814/2218, winner/031 only; audit mean must improve, otherwise retain031 without runner-up selection.',
      'history':'Two heavily reused external first-division Bundesliga matches, not fresh or independent games. No challenge inputs/labels/video/public scores used. Studies032 and033 both compare031 independently.',
      'competition_data_used':False,'competition_percentile_known':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong external match')
    folds=[];all_rows=[];all_base=[];all_preds={v['name']:[] for v in VARIANTS}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();bundle,paths=bundle031(fi,roots);xs=[];ys=[];audit=[]
        for half in (0,1):
            seq=period_arrays(records[train],half);t,valid=targets(seq['y'])
            for ci,c in enumerate(CONDITIONS):
                print('Training event features',fi,half,c,flush=True);s,mask=condition(seq,c,half,True);_,p=infer(bundle,s,records[train]['pitch'],mask);x=boundary_observations(s,records[train]['pitch'],p,mask);ix=np.flatnonzero(valid)[::5];pos=ix[t[ix]==1];neg=ix[t[ix]==0]
                if not len(pos):continue
                neg=np.random.default_rng(131+half*4+ci).choice(neg,min(len(neg),len(pos)),replace=False);chosen=np.r_[pos,neg];xs.append(x[chosen]);ys.append(t[chosen]);audit.append({'half':half+1,'condition':c,'positive':len(pos),'negative':len(neg)})
        x=np.concatenate(xs);y=np.concatenate(ys);np.savez_compressed(fd/'training-features.npz',x=x,y=y);head=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.05,min_samples_leaf=80,l2_regularization=20,early_stopping=False,random_state=133)
        with threadpool_limits(limits=1):head.fit(x,y)
        tree_model.save(fd/'boundary-tree.npz',head,'hist_boost');tree=tree_model.load(fd/'boundary-tree.npz');np.testing.assert_allclose(head.predict_proba(x),tree_model.predict(tree,x)[:,:2],atol=1e-9);del x,y,xs,ys
        rows=[];base=[];preds={v['name']:[] for v in VARIANTS}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);a,b=predict_variants(bundle,tree,s,records[val]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});base.append(a)
                for k in preds:preds[k].append(b[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=a,**b)
        f={'train':train,'validation':val,'training_audit':audit,'frozen_component_hashes':{k:digest(p) for k,p in paths.items()},'boundary_tree_sha256':digest(fd/'boundary-tree.npz'),'current':evaluate_accuracy(rows,base),'candidates':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base
        for k in preds:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for v in VARIANTS:
        score=evaluate_accuracy(all_rows,all_preds[v['name']],all_base);guards=eligible(folds,v['name'],current,score);trials.append({**v,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'head_hashes':[digest(out/f'fold-{i}/boundary-tree.npz') for i in (0,1)],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            bundle,_=bundle031(fi,roots);tree=tree_model.load(out/f'fold-{fi}/boundary-tree.npz')
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),1614+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),1814+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,2218+half)
                    # Only the frozen winner is decoded on audit draws.
                    a,p=infer(bundle,s,records[mid]['pitch'],mask);x=boundary_observations(s,records[mid]['pitch'],p,mask);q=probabilities(bundle['models']['boundary'],x)[:,1] if selected['kind']=='linear' else tree_model.predict(tree,x)[:,1];rows.append({'y':seq['y'],'condition':c});base.append(a);preds.append(floor_decode(p,q,selected['floor']))
        audit={'current':evaluate_accuracy(rows,base),'candidate':evaluate_accuracy(rows,preds,base),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.out)
if __name__=='__main__':main()
