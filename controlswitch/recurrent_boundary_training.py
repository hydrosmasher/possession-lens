"""054: refit turnover evidence for the frozen052 recurrent observation model."""
from pathlib import Path
import argparse,json
import numpy as np
from .ranked_pipeline import load_head
from .narrow_boundary import bundle040
from .gated_rnn import load_checkpoint
from .recurrent_boundary import evidence,event_features,decode_head
from .boundary_had import boundary_observations,targets,adaptive_decode
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .model import fit_classifier,probabilities
from .pipeline import digest

VARIANTS=[{'name':'aligned116','features':116},{'name':'event120','features':120}]

def eligible(folds,name,current,score):
    return {'mean_gain_at_least_003':score['mean_spell_f1']>=current['mean_spell_f1']+.003,
            'prior_conditions':all(f['candidates'][name][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),
            'tracker_noise':all(f['candidates'][name]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),
            'dead_f1':all(f['candidates'][name][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS)}

def run(cache_root,roots,run_root,out):
    run_root=Path(run_root)
    out=Path(out)
    if not json.loads((run_root/'transfer-052/report.json').read_text())['recommended_for_freeze']:raise ValueError('Need passed052 audit')
    if out.exists():raise ValueError('Choose a new run directory')
    out.mkdir(parents=True)
    protocol={'version':'recurrent-boundary-054','reference':'frozen052','hypothesis':'The retained031 turnover head was fitted on older030emissions. Refit to052emissions; compare the same116features with120features including the already-trained GRU boundary signal.',
    'variants':VARIANTS,'training':'Same broad direct H/A-turnover target +-12frames; valid only fully known-live neighbourhoods. Every5th valid frame, positives retained and negatives balanced within training half/condition.25epochs logistic2class,seed154,L2.001;training-only scaling. Four original external training corruptions. No recurrent weight changes.',
    'features':'Aligned116 uses existing invariant features on052emissions. Event120 adds symmetric frozenGRUevent score, centred means radius12/25 and absolute+-12change. No label/timestamp/identity inputs. Balanced head evidence is not a calibrated event probability.',
    'selection':'Fixed full-strength adaptive decoder,cost12,stop3. Highest four-condition mean gain>=.003 over052; per-fold clean/gap decline<=.01,noise no decline,deadF1 decline<=.005. No post-result tuning.',
    'freeze':'One winner/hash frozen before audit5414/5614/6018. Only052/winner audited; mean must improve or retain052, no runner-up reselection.',
    'history':'Two heavily reused existing first-division Bundesliga games, not independent tests. ForwardfitWMX/validateWN1,reversefitWN1/validateWMX. Recurrent/emission backbones saw training match. No challenge data/video/labels/public score used; no competition-rank claim.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records={}
    for mid in ('J03WMX','J03WN1'):
        files=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(files)!=1:raise ValueError('Ambiguous external cache')
        with np.load(files[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong external match')
    folds=[];all_rows=[];all_base=[];all_preds={v['name']:[] for v in VARIANTS}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();base,paths=bundle040(fi,roots,run_root/'joint-040');bundle={'base':base,'gru':load_checkpoint(run_root/f'transfer-052/fold-{fi}/spatial_more.npz'),'mix':.5};old=base['models']['boundary'];xs=[];ys=[];audit=[]
        for half in (0,1):
            seq=period_arrays(records[train],half);target,valid=targets(seq['y'])
            for ci,c in enumerate(CONDITIONS):
                print('Train features',fi,half,c,flush=True);s,mask=condition(seq,c,half,True);p,x,e=evidence(bundle,s,records[train]['pitch'],mask);x=event_features(x,e)
                ix=np.flatnonzero(valid)[::5];pos=ix[target[ix]==1];neg=ix[target[ix]==0]
                if not len(pos):continue
                neg=np.random.default_rng(154+half*4+ci).choice(neg,min(len(neg),len(pos)),replace=False);chosen=np.r_[pos,neg];xs.append(x[chosen]);ys.append(target[chosen]);audit.append({'half':half+1,'condition':c,'positive':len(pos),'negative':len(neg)})
        xx=np.concatenate(xs);yy=np.concatenate(ys);heads={}
        for v in VARIANTS:
            head=fit_classifier(xx[:,:v['features']],yy,classes=2,epochs=25,seed=154,l2=.001);np.savez_compressed(fd/(v['name']+'.npz'),**head);heads[v['name']]=head
        del xs,ys,xx,yy
        rows=[];base=[];preds={v['name']:[] for v in VARIANTS}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validate',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);p,x,e=evidence(bundle,s,records[val]['pitch'],mask);labels=adaptive_decode(p,probabilities(old,x)[:,1],1.)
                with np.load(run_root/f'transfer-052/fold-{fi}/validation-{half*4+ci}.npz') as ref:np.testing.assert_array_equal(labels,ref['spatial_more'])
                rows.append({'y':seq['y'],'condition':c});base.append(labels)
                for v in VARIANTS:preds[v['name']].append(decode_head(heads[v['name']],p,x,e))
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=labels,emissions=p,**{k:v[-1] for k,v in preds.items()})
        f={'train':train,'validation':val,'training_audit':audit,'input_model_sha256':{**{k:digest(p) for k,p in paths.items()},'gru052':digest(run_root/f'transfer-052/fold-{fi}/spatial_more.npz')},'head_sha256':{v['name']:digest(fd/(v['name']+'.npz')) for v in VARIANTS},'current':evaluate_accuracy(rows,base),'candidates':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base
        for k in all_preds:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for v in VARIANTS:
        score=evaluate_accuracy(all_rows,all_preds[v['name']],all_base);guards=eligible(folds,v['name'],current,score);trials.append({**v,'scores':score,'guards':guards,'eligible':all(guards.values())})
    candidates=[t for t in trials if t['eligible']];selected=max(candidates,key=lambda t:t['scores']['mean_spell_f1']) if candidates else None
    freeze={'selected':selected,'head_hashes':[{v['name']:digest(out/f'fold-{i}'/(v['name']+'.npz')) for v in VARIANTS} for i in (0,1)],'audit_not_yet_run':True};(out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print('FROZEN',selected,flush=True)
    audit=None
    if selected:
        rows=[];base=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            component,_=bundle040(fi,roots,run_root/'joint-040');bundle={'base':component,'gru':load_checkpoint(run_root/f'transfer-052/fold-{fi}/spatial_more.npz'),'mix':.5};head=load_head(out/f'fold-{fi}'/(selected['name']+'.npz'))
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),5414+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),5614+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,6018+half)
                    p,x,e=evidence(bundle,s,records[mid]['pitch'],mask);a=adaptive_decode(p,probabilities(bundle['base']['models']['boundary'],x)[:,1],1.);b=decode_head(head,p,x,e);rows.append({'y':seq['y'],'condition':c});base.append(a);preds.append(b);np.savez_compressed(out/f'fold-{fi}'/f'audit-{half*4+CONDITIONS.index(c)}.npz',current=a,candidate=b)
        audit={'current':evaluate_accuracy(rows,base),'candidate':evaluate_accuracy(rows,preds,base),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'competition_percentile_known':False,'promoted_to_default':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-root',required=True,type=Path);p.add_argument('--roots',required=True,type=Path);p.add_argument('--run-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.run_root,a.out)
if __name__=='__main__':main()
