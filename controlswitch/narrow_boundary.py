"""041: tighter external turnover localisation on frozen040 emissions."""
from pathlib import Path
import argparse,json
import numpy as np
from .ranking_experiment import asset_paths
from .ranked_pipeline import make_bundle,infer,load_head
from .boundary_had import boundary_observations
from .boundary_alignment import eligible
from .nonlinear_boundary import floor_decode
from .control_state import evaluate_accuracy
from .model import rolling,fit_classifier,probabilities
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest

SPEC={'name':'joint_motion_quality1_boundary_full','control':'joint','dead':'quality','cap':1.,'boundary':'balanced','strength':1.}
VARIANTS=[{'name':f'r{r}_floor{f:g}','radius':r,'floor':f} for r in (2,5) for f in (.25,.05)]

def bundle040(fi,roots,control_root):
    paths=asset_paths(fi,SPEC,roots);paths['control']=Path(control_root)/f'fold-{fi}/joint200.npz';paths['boundary']=Path(roots['aligned_boundary'])/f'fold-{fi}/boundary-aligned.npz';return make_bundle(SPEC,paths),paths

def narrow_targets(y,radius):
    y=np.asarray(y)
    if radius not in (2,5) or y.ndim!=1 or not len(y):raise ValueError('Invalid narrow-boundary target')
    event=np.r_[False,(y[1:]<2)&(y[:-1]<2)&(y[1:]!=y[:-1])];target=(rolling(event[:,None].astype(float),radius)[:,0]>0).astype(np.int8);valid=rolling((y<2)[:,None].astype(float),12)[:,0]>.999;return target,valid

def run(cache_root,roots,control_root,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output folder')
    out.mkdir(parents=True);protocol={'version':'narrow-boundary-041','reference':'frozen040','hypothesis':'The previous +-12frame positive event window can blur short-possession boundary timing. Test +-2 and+-5frame targets on the same116 invariant features and frozen040 emissions.',
     'training':'Each radius: all valid positive frames, balanced random negative subset from every5th valid frame; only entirely known-live +-12 neighbourhoods. Logistic25epochs/seed141/L2.001,train-only standardisation. Same four fixed external training conditions. No new input features or epoch search.',
     'variants':VARIANTS,'decoder':'Full-strength balanced switch adaptation; H/A cost12,stop3 fixed. Event floor.25(default) or.05. No hard possession duration floor.',
     'selection':'Mean official four-condition spellF1 gain>=.003 over040; clean/gap fold decline<=.01, noise fold no decline, deadF1 decline<=.005. Highest eligible mean, report FP trade-off.',
     'audit':'Freeze one winner and hashes before different-draw audit seeds2314/2514/2918. Only winner/040 decoded; audit mean must improve or retain040, no runner-up selection.',
     'history':'Two heavily reused external first-division Bundesliga matches, not independent tests. In-fold emission backbones saw the same training match; validation is opposite match. No challenge inputs, video, possession labels or public score used.',
     'competition_data_used':False,'competition_percentile_known':False};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
    folds=[];all_rows=[];all_base=[];all_preds={v['name']:[] for v in VARIANTS}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();bundle,paths=bundle040(fi,roots,control_root);xs={r:[] for r in (2,5)};ys={r:[] for r in (2,5)};train_audit=[]
        for half in (0,1):
            seq=period_arrays(records[train],half)
            for ci,c in enumerate(CONDITIONS):
                print('Training features',fi,half,c,flush=True);s,mask=condition(seq,c,half,True);_,p=infer(bundle,s,records[train]['pitch'],mask);x=boundary_observations(s,records[train]['pitch'],p,mask)
                for radius in (2,5):
                    t,valid=narrow_targets(seq['y'],radius);pos=np.flatnonzero(valid&(t==1));neg=np.flatnonzero(valid&(t==0))[::5]
                    if not len(pos):continue
                    neg=np.random.default_rng(141+radius*100+half*4+ci).choice(neg,min(len(neg),len(pos)),replace=False);chosen=np.r_[pos,neg];xs[radius].append(x[chosen]);ys[radius].append(t[chosen]);train_audit.append({'radius':radius,'half':half+1,'condition':c,'positive':len(pos),'negative':len(neg)})
        heads={}
        for radius in (2,5):
            x=np.concatenate(xs[radius]);y=np.concatenate(ys[radius]);np.savez_compressed(fd/f'training-r{radius}.npz',x=x,y=y);head=fit_classifier(x,y,classes=2,epochs=25,seed=141,l2=.001);np.savez_compressed(fd/f'head-r{radius}.npz',**head);heads[radius]=load_head(fd/f'head-r{radius}.npz')
        del xs,ys,x,y;rows=[];base=[];preds={v['name']:[] for v in VARIANTS}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);a,p=infer(bundle,s,records[val]['pitch'],mask);x=boundary_observations(s,records[val]['pitch'],p,mask);scores={r:probabilities(h,x)[:,1] for r,h in heads.items()};rows.append({'y':seq['y'],'condition':c});base.append(a)
                for v in VARIANTS:preds[v['name']].append(floor_decode(p,scores[v['radius']],v['floor']))
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=a,**{k:p[-1] for k,p in preds.items()})
        f={'train':train,'validation':val,'training_audit':train_audit,'reference_hashes':{k:digest(p) for k,p in paths.items()},'current':evaluate_accuracy(rows,base),'candidates':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base
        for k in preds:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for v in VARIANTS:
        score=evaluate_accuracy(all_rows,all_preds[v['name']],all_base);guards=eligible(folds,v['name'],current,score);trials.append({**v,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'head_hashes':[{r:digest(out/f'fold-{i}'/f'head-r{r}.npz') for r in (2,5)} for i in (0,1)],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            bundle,_=bundle040(fi,roots,control_root);head=load_head(out/f'fold-{fi}'/f'head-r{selected["radius"]}.npz')
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),2314+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),2514+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,2918+half)
                    a,p=infer(bundle,s,records[mid]['pitch'],mask);q=probabilities(head,boundary_observations(s,records[mid]['pitch'],p,mask))[:,1];rows.append({'y':seq['y'],'condition':c});base.append(a);preds.append(floor_decode(p,q,selected['floor']))
        audit={'current':evaluate_accuracy(rows,base),'candidate':evaluate_accuracy(rows,preds,base),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','control-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.control_root,a.out)
if __name__=='__main__':main()
