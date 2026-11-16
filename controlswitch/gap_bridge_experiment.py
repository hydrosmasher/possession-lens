"""066: isolate team-ownership errors inside bracketed ball gaps."""
from pathlib import Path
import json
import numpy as np
from .gap_bridge import bridge_features,swap_bridge,anchored_gap_decode
from .collective_experiment import spell_weights
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition
from .research_snapshot import cached_case
from .ledger_experiment import short_hits
from .pipeline import digest
from . import tree_model


def run(root,cache):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    root=Path(root).resolve();out=root/'local/gap-bridge-066';out.mkdir(exist_ok=False)
    protocol={'version':'gap-bridge-066','root_issue':'Missing-ball team decisions; no change to seen-ball frames or D mask.',
              'features':'188:90perteam=26collective-shape/motion+25framepastmean+25framefuturemean+6previousseenanchor+6nextseenanchor;8sharedgapage/length/phase/visibilityflags. Only observations,not labels or reconstructed ball locations.',
              'eligibility':'Only bracketed missing-ball runs<=300frames (12s). Open-ended/longer gaps and all observed frames unchanged. Existing D predictions immutable.',
              'training':'Opposite-match WMX/WN1 folds; four independent training corruption conditions. Every5th eligible known-live frame,exactteam swaps,existing bounded inverse-sqrt spell weights. HGB160fixediterations/15leaves/minleaf80/lr.05/L2=20/seed166/noearlystop. No validation-fitted scaler or thresholds.',
              'comparison':['prior_bridge','learned_bridge'],
              'decoder':'Inside eligible predicted-live runs only;2stateViterbi penalty12,locked live flank labels,reset atD. prior_bridgeusesexistingconditionalHA only;learned_bridge50/50prior+specialist. No new global decoder or fake ball coordinates.',
              'selection':'Learned only:mean spellF1 gain>=.002,misses/FP no increase,matched subsecondGT no decrease,each noise fold no decline,othergapfold decline<=.01,eligible live frame accuracy>=reference andmatched prior_bridge. Observed frames,Dmask exact. No promotion before frozen new-draw audit.',
              'audit':'Only frozen learned winner,seeds10517/10717/11121. Same invariant locks,mean improvement andmisses/FP no increase. Not fresh matches. No runner-up or tuning from audit.',
              'history':'Two heavily reused Bundesliga first-division games; development only. No challenge/video/labels/publicscore used; no downloads.',
              'source_sha256':{name:digest(root/'controlswitch'/name) for name in ('gap_bridge.py','gap_bridge_experiment.py')}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');descriptor={'source':'short-spell-053','name':'short_focus'};names=('prior_bridge','learned_bridge');folds=[];allrows=[];allbase=[];allpred={k:[] for k in names};gap_counts={k:[0,0] for k in ('reference',)+names};coverage=[]
    with threadpool_limits(limits=1):
      for fi,(train,val) in enumerate((('J03WMX','J03WN1'),('J03WN1','J03WMX'))):
        fd=out/f'fold-{fi}';fd.mkdir();paths=list(Path(cache).glob(train+'-*.npz'))
        if len(paths)!=1:raise ValueError('Ambiguous cache')
        with np.load(paths[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
        xs=[];ys=[];ws=[];audit=[]
        for half in (0,1):
          seq=period_arrays(record,half);weights=spell_weights(seq['y'])
          for kind in CONDITIONS:
            print('Fit gap features',fi,half,kind,flush=True);s,mask=condition(seq,kind,half,True);x,eligible,seen=bridge_features(s,record['pitch'],mask);ix=np.flatnonzero(eligible&(seq['y']<2))[::5]
            if len(ix):xs.extend([x[ix],swap_bridge(x[ix])]);ys.extend([seq['y'][ix],1-seq['y'][ix]]);ws.extend([weights[ix],weights[ix]])
            audit.append({'half':half+1,'condition':kind,'samples_before_swap':len(ix),'missing_frames':int((~seen).sum()),'eligible_missing_frames':int(eligible.sum())})
        x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=80,learning_rate=.05,l2_regularization=20,random_state=166,early_stopping=False);model.fit(x,y,sample_weight=w);tree_model.save(fd/'gap-specialist.npz',model,'hist_boost');saved=tree_model.load(fd/'gap-specialist.npz');np.testing.assert_allclose(tree_model.predict(saved,x[:10000])[:,:2],model.predict_proba(x[:10000]),atol=1e-12,rtol=1e-12);del xs,ys,ws,x,y,w,model,record
        paths=list(Path(cache).glob(val+'-*.npz'))
        if len(paths)!=1:raise ValueError('Ambiguous cache')
        with np.load(paths[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
        rows=[];base=[];preds={k:[] for k in names};foldg={k:[0,0] for k in gap_counts}
        for half in (0,1):
          seq=period_arrays(record,half)
          for ci,kind in enumerate(CONDITIONS):
            print('Evaluate gap bridge',fi,half,kind,flush=True);s,mask=condition(seq,kind,half,False);x,eligible,seen=bridge_features(s,record['pitch'],mask);a=tree_model.predict(saved,x)[:,:2];b=tree_model.predict(saved,swap_bridge(x))[:,:2];g=(a+b[:,::-1])*.5;y,current,q=cached_case(descriptor,fi,half*4+ci,root/'local');np.testing.assert_array_equal(y,seq['y']);local={};evaluate=eligible&(y<2);coverage.append({'fold':fi,'half':half+1,'condition':kind,'missing_frames':int((~seen).sum()),'eligible_frames':int(eligible.sum()),'eligible_reference_live_frames':int(evaluate.sum())})
            for name,expert in [('prior_bridge',q[:,:2]),('learned_bridge',g)]:
              p=anchored_gap_decode(current,q[:,:2],expert,eligible);np.testing.assert_array_equal(p[~eligible],current[~eligible]);np.testing.assert_array_equal(p==2,current==2);np.testing.assert_array_equal(p[seen],current[seen]);preds[name].append(p);local[name]=p;foldg[name][0]+=int((p[evaluate]==y[evaluate]).sum());foldg[name][1]+=int(evaluate.sum())
            foldg['reference'][0]+=int((current[evaluate]==y[evaluate]).sum());foldg['reference'][1]+=int(evaluate.sum());rows.append({'y':y,'condition':kind});base.append(current);np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=y,current=current,eligible=eligible,seen=seen,**local)
        f={'train':train,'evaluate':val,'training':audit,'model_sha256':digest(fd/'gap-specialist.npz'),'reference':evaluate_accuracy(rows,base),'variants':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()},'eligible_live_accuracy':{k:v[0]/max(v[1],1) for k,v in foldg.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');allrows+=rows;allbase+=base
        for k in names:allpred[k]+=preds[k]
        for k,v in foldg.items():gap_counts[k][0]+=v[0];gap_counts[k][1]+=v[1]
    reference=evaluate_accuracy(allrows,allbase);accuracy={k:v[0]/max(v[1],1) for k,v in gap_counts.items()};short=sum(short_hits(r['y'],p) for r,p in zip(allrows,allbase));trials=[]
    for name,p in allpred.items():
      score=evaluate_accuracy(allrows,p,allbase);sh=sum(short_hits(r['y'],v) for r,v in zip(allrows,p));guards={'spell_gain':score['mean_spell_f1']>=reference['mean_spell_f1']+.002,'misses':score['totals']['missed']<=reference['totals']['missed'],'false_positives':score['totals']['false_positive']<=reference['totals']['false_positive'],'short_possessions':sh>=short,'noise':all(f['variants'][name]['tracker_noise']['spell_f1_mean']>=f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),'gap_noninferior':all(f['variants'][name][c]['spell_f1_mean']>=f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[1:3]),'eligible_accuracy':accuracy[name]>=accuracy['reference'] and accuracy[name]>=accuracy['prior_bridge']};trials.append({'name':name,'scores':score,'short_matched':sh,'guards':guards,'eligible':name=='learned_bridge' and all(guards.values())})
    selected=next((t for t in trials if t['eligible']),None);report={'protocol':protocol,'folds':folds,'coverage':coverage,'reference':reference,'reference_short_matched':short,'eligible_live_accuracy':accuracy,'trials':trials,'selected':selected,'promoted_to_default':False,'independent_test':False,'frozen_audit_required':bool(selected)}
    (out/'freeze.json').write_text(json.dumps({'selected':selected,'models':[f['model_sha256'] for f in folds],'before_new_draw_audit':True},indent=2)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('Eligible gap accuracy',accuracy,flush=True)
    for t in trials:print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--cache-root',type=Path);a=parser.parse_args();run(a.root,a.cache_root or a.root/'data/external/cache')
