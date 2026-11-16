"""059: external-only collective/contact dead-ball experts on frozen control."""
from pathlib import Path
import argparse,json
import numpy as np,torch
from .robust_dead_state import robust_dead_features,blend_dead
from .spatial_symmetry import reflect_sequence
from .contact_training import infer_models
from .narrow_boundary import bundle040
from .gated_rnn import load_checkpoint
from .boundary_had import adaptive_decode,boundary_observations
from .model import probabilities
from .boundary_alignment import eligible
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest
from . import tree_model
VARIANTS=[{'name':'matched240_half','features':240,'mix':.5},{'name':'robust487_half','features':487,'mix':.5},{'name':'robust487_full','features':487,'mix':1.}]

def reference(run_root):
    for source in ('spell-overlap-058','contact-057','short-spell-053'):
        report=json.loads((Path(run_root)/source/'report.json').read_text())
        if report['recommended_for_freeze']:return source,report['selected']['name'],report
    raise ValueError('No verified reference')

def run(cache_root,roots,run_root,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    run_root=Path(run_root);out=Path(out);source,branch,initial=reference(run_root)
    if initial['selected']['scores']['mean_spell_f1']>=.7:print('Verified reference reaches0.70; no new experiment.',flush=True);return
    if out.exists():raise ValueError('Choose new output directory')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);protocol={'version':'collective-contact-dead-059','reference':source+'/'+branch,'hypothesis':'Static older live/dead experts miss short stop/restart boundaries during poor observations. Add ball-independent collective dynamics and per-player association/quality features; isolate extra features against matched old-feature training.',
    'variants':VARIANTS,'training':'Each head200HistGradientBoostingtrees,31leaves,lr.05,minleaf80,L2=20,seed159,no earlystop. Known H/A/D frames sampled every10frames perhalf/fourfixed training conditions/four coherent x/y views. Dtarget=(externalBallStatusdead); no weak labels. Matched240 uses old240features;robust487adds64invariant contact,156invariant collective/context and27quality features. No labels/timestamps/identities in features.',
    'fusion':'H/Aconditional probabilities/GRU fixed. New dead-tree probability gets0.75logit scaling. Blend its log odds at.5or1 with the reference final dead odds; then recompute old031boundary evidence and decode H/A12,stop3. Matchedheadonlyhalfblend as capacity/augmentation control.',
    'selection':'Mean original four-condition spellF1 gain>=.003 vsreference,perfoldclean/gap decline<=.01,noise no decline,deadF1 decline<=.005 andaggregateFPnoincrease. Highest eligible mean. No post-result search.',
    'audit':'Freeze one head/mix winner before observation seeds9514/9714/10118. Only winner/reference audited. Mean must improve andFPmustnotincrease orretainreference,no runner-up tuning.',
    'history':'Existing two heavily reused first-division Bundesliga matches; no independent test. ForwardWMX/reverseWN1 train,opposite-match development. No provided-match inputs/video/labels/public score used. No newdownloads. All previous work retained.'};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        files=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(files)!=1:raise ValueError('Ambiguous external cache')
        with np.load(files[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
    folds=[];all_rows=[];all_base=[];all_preds={v['name']:[] for v in VARIANTS}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();xs=[];ys=[];training=[];base,_=bundle040(fi,roots,run_root/'joint-040')
        for half in (0,1):
            seq=period_arrays(records[train],half);ix=np.flatnonzero(seq['y']<3)[::10];target=(seq['y'][ix]==2).astype(np.int8)
            for c in CONDITIONS:
                s,mask=condition(seq,c,half,True)
                for view in range(4):
                    print('Training features',fi,half,c,view,flush=True);obs=reflect_sequence(s,bool(view&1),bool(view&2));x=robust_dead_features(obs,records[train]['pitch'],mask);xs.append(x[ix]);ys.append(target);training.append({'half':half+1,'condition':c,'view':view,'sampled_rows':len(ix),'dead_rows':int(target.sum())})
        x=np.concatenate(xs);y=np.concatenate(ys);del xs,ys;heads={}
        for width in (240,487):
            model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=31,learning_rate=.05,min_samples_leaf=80,l2_regularization=20,early_stopping=False,random_state=159)
            with threadpool_limits(limits=1):model.fit(x[:,:width],y)
            tree_model.save(fd/f'dead{width}.npz',model,'hist_boost');head=tree_model.load(fd/f'dead{width}.npz')
            with threadpool_limits(limits=1):np.testing.assert_allclose(tree_model.predict(head,x[:,:width])[:,:2],model.predict_proba(x[:,:width]),atol=1e-9)
            heads[width]=head;print('Fitted/export verified',fi,width,len(y),flush=True);del model
        del x,y;rows=[];base_preds=[];preds={v['name']:[] for v in VARIANTS}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);s={k:v for k,v in s.items() if k!='y'};x=robust_dead_features(s,records[val]['pitch'],mask)
                with np.load(run_root/source/f'fold-{fi}/validation-{half*4+ci}.npz') as a:expected=a[branch].copy();np.testing.assert_array_equal(a['reference'],seq['y']);q0=None if source=='short-spell-053' else a['emissions_'+branch].copy()
                if q0 is None:
                    with np.load(run_root/f'precision-055-cache/fold-{fi}/validation-{half*4+ci}.npz') as a:q0=a['emissions'].copy();np.testing.assert_array_equal(a['current'],expected)
                event=probabilities(base['models']['boundary'],boundary_observations(s,records[val]['pitch'],q0,mask))[:,1];np.testing.assert_array_equal(adaptive_decode(q0,event,1.),expected);rows.append({'y':seq['y'],'condition':c});base_preds.append(expected);nd={width:tree_model.predict(m,x[:,:width])[:,1] for width,m in heads.items()}
                for v in VARIANTS:
                    q=blend_dead(q0,nd[v['features']],v['mix']);e=probabilities(base['models']['boundary'],boundary_observations(s,records[val]['pitch'],q,mask))[:,1];preds[v['name']].append(adaptive_decode(q,e,1.))
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=expected,**{k:v[-1] for k,v in preds.items()},reference_emissions=q0,**{'dead'+str(k):v for k,v in nd.items()})
        f={'train':train,'validation':val,'training_audit':training,'sampled_exposures':sum(a['sampled_rows'] for a in training),'heads_sha256':{str(w):digest(fd/f'dead{w}.npz') for w in heads},'current':evaluate_accuracy(rows,base_preds),'candidates':{k:evaluate_accuracy(rows,p,base_preds) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base_preds
        for k in preds:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);np.testing.assert_allclose(current['mean_spell_f1'],initial['selected']['scores']['mean_spell_f1'],atol=1e-14,rtol=0);trials=[]
    for v in VARIANTS:
        score=evaluate_accuracy(all_rows,all_preds[v['name']],all_base);guards=eligible(folds,v['name'],current,score);guards['false_positive_no_increase']=score['totals']['false_positive']<=current['totals']['false_positive'];trials.append({**v,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'initial_source':source,'initial_name':branch,'head_hashes':[{str(w):digest(out/f'fold-{fi}'/f'dead{w}.npz') for w in (240,487)} for fi in (0,1)],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base_preds=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            base,_=bundle040(fi,roots,run_root/'joint-040');grus={'reference':load_checkpoint(run_root/source/f'fold-{fi}'/(branch+'.npz'))};head=tree_model.load(out/f'fold-{fi}'/f"dead{selected['features']}.npz")
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for ci,c in enumerate(CONDITIONS):
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),9514+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),9714+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,10118+half)
                    p,qs=infer_models(base,grus,s,records[mid]['pitch'],mask);x=robust_dead_features(s,records[mid]['pitch'],mask);q=blend_dead(qs['reference'],tree_model.predict(head,x[:,:selected['features']])[:,1],selected['mix']);e=probabilities(base['models']['boundary'],boundary_observations(s,records[mid]['pitch'],q,mask))[:,1];new=adaptive_decode(q,e,1.);rows.append({'y':seq['y'],'condition':c});base_preds.append(p['reference']);preds.append(new);np.savez_compressed(out/f'fold-{fi}'/f'audit-{half*4+ci}.npz',current=p['reference'],candidate=new)
        a=evaluate_accuracy(rows,base_preds);b=evaluate_accuracy(rows,preds,base_preds);audit={'current':a,'candidate':b,'mean_improves':b['mean_spell_f1']>a['mean_spell_f1'],'false_positive_no_increase':b['totals']['false_positive']<=a['totals']['false_positive'],'same_matches_new_noise_only':True}
    report={'protocol':protocol,'initial_source':source,'initial_name':branch,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves'] and audit['false_positive_no_increase']),'promoted_to_default':False,'independent_test':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials:print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],audit['false_positive_no_increase'],flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','run-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.run_root,a.out)
