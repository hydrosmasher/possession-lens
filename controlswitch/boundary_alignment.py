"""External-only031: align turnover head to the frozen030 observation model."""
from pathlib import Path
import argparse,json
import numpy as np
from .ranked_pipeline import make_bundle,infer,load_head
from .ranking_experiment import asset_paths
from .boundary_had import boundary_observations,targets,adaptive_decode
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .model import fit_classifier,probabilities
from .pipeline import digest

SPEC={'name':'joint_state_quality1_boundary','control':'state','dead':'quality','cap':1.,'boundary':'balanced','strength':.5}
VARIANTS=[{'name':'old_full','head':'old','strength':1.},{'name':'aligned_half','head':'aligned','strength':.5},{'name':'aligned_full','head':'aligned','strength':1.}]

def eligible(folds,name,current,score):
    return {'mean_gain_at_least_003':score['mean_spell_f1']>=current['mean_spell_f1']+.003,
            'prior_conditions':all(f['candidates'][name][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),
            'tracker_noise':all(f['candidates'][name]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),
            'dead_f1':all(f['candidates'][name][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS)}

def run(cache_root,roots,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new run directory')
    out.mkdir(parents=True)
    protocol={'version':'boundary-alignment-031','reference':'frozen030','history':'Two extensively reused top-five-league Bundesliga matches; adaptive development, not independent testing. No challenge inputs accepted.',
      'hypothesis':'020 turnover head saw019 emissions; refit same116 features on030 emissions to reduce feature-distribution mismatch. Contrast fixed stronger old head.',
      'variants':VARIANTS,'training':'Same020 target +-12frame direct H/A turnover, fully known-live neighbourhoods; balanced positives/negatives every5th frame per half/condition; logistic25epochs/seed131/L2.001. Train-only normalisation; no new feature or target search.',
      'selection':'Highest four-condition official spellF1, gain>=.003 over030; each clean/gap fold loss<=.01, each noise fold no decline, each deadF1 loss<=.005. FP reported, no absolute veto. No post-result search.',
      'freeze':'Select and hash before new-draw audit; only winner and030 audited. Audit must improve meanF1; otherwise retain030 without runner-up selection.',
      'audit_seeds':{'masked':1414,'long_gap':1614,'tracker_noise':2018},'new_match_test':False,'competition_data_used':False,'limitations':'Training emission backbones have seen the same training match; head validation is opposite match, both repeatedly reused. Balanced scores are event evidence, not calibrated probabilities.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records={}
    for mid in ('J03WMX','J03WN1'):
        files=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(files)!=1:raise ValueError('Ambiguous external cache')
        with np.load(files[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong external match')
    folds=[];all_rows=[];all_base=[];all_preds={v['name']:[] for v in VARIANTS}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();paths=asset_paths(fi,SPEC,roots);bundle=make_bundle(SPEC,paths);old=bundle['models']['boundary'];xs=[];ys=[];audit=[]
        for half in (0,1):
            seq=period_arrays(records[train],half);target,valid=targets(seq['y'])
            for ci,c in enumerate(CONDITIONS):
                print('Train features',fi,half,c,flush=True);s,mask=condition(seq,c,half,True);_,p=infer(bundle,s,records[train]['pitch'],mask);x=boundary_observations(s,records[train]['pitch'],p,mask)
                ix=np.flatnonzero(valid)[::5];pos=ix[target[ix]==1];neg=ix[target[ix]==0]
                if not len(pos):continue
                neg=np.random.default_rng(131+half*4+ci).choice(neg,min(len(neg),len(pos)),replace=False);chosen=np.r_[pos,neg];xs.append(x[chosen]);ys.append(target[chosen]);audit.append({'half':half+1,'condition':c,'positive':len(pos),'negative':len(neg)})
        head=fit_classifier(np.concatenate(xs),np.concatenate(ys),classes=2,epochs=25,seed=131,l2=.001);np.savez_compressed(fd/'boundary-aligned.npz',**head);del xs,ys
        rows=[];base=[];preds={v['name']:[] for v in VARIANTS}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validate',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);labels,p=infer(bundle,s,records[val]['pitch'],mask);x=boundary_observations(s,records[val]['pitch'],p,mask);q={'old':probabilities(old,x)[:,1],'aligned':probabilities(head,x)[:,1]};rows.append({'y':seq['y'],'condition':c});base.append(labels)
                for v in VARIANTS:preds[v['name']].append(adaptive_decode(p,q[v['head']],v['strength']))
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=labels,emissions=p,old_score=q['old'],aligned_score=q['aligned'],**{k:v[-1] for k,v in preds.items()})
        f={'train':train,'validation':val,'training_audit':audit,'input_model_sha256':{k:digest(p) for k,p in paths.items()},'head_sha256':digest(fd/'boundary-aligned.npz'),'current':evaluate_accuracy(rows,base),'candidates':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base
        for k in all_preds:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for v in VARIANTS:
        score=evaluate_accuracy(all_rows,all_preds[v['name']],all_base);guards=eligible(folds,v['name'],current,score);trials.append({**v,'scores':score,'guards':guards,'eligible':all(guards.values())})
    candidates=[t for t in trials if t['eligible']];selected=max(candidates,key=lambda t:t['scores']['mean_spell_f1']) if candidates else None
    freeze={'selected':selected,'head_hashes':[digest(out/f'fold-{i}/boundary-aligned.npz') for i in (0,1)],'audit_not_yet_run':True};(out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print('FROZEN',selected,flush=True)
    audit=None
    if selected:
        rows=[];base=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            bundle=make_bundle(SPEC,asset_paths(fi,SPEC,roots));head=bundle['models']['boundary'] if selected['head']=='old' else load_head(out/f'fold-{fi}/boundary-aligned.npz')
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),1414+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),1614+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,2018+half)
                    a,p=infer(bundle,s,records[mid]['pitch'],mask);q=probabilities(head,boundary_observations(s,records[mid]['pitch'],p,mask))[:,1];rows.append({'y':seq['y'],'condition':c});base.append(a);preds.append(adaptive_decode(p,q,selected['strength']))
        audit={'current':evaluate_accuracy(rows,base),'candidate':evaluate_accuracy(rows,preds,base),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'competition_percentile_known':False,'promoted_to_default':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-root',required=True,type=Path);p.add_argument('--roots',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.out)
if __name__=='__main__':main()
