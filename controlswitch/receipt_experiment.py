"""064: distinguish close-ball ownership compatibility from proximity alone.

External H/A possession supplies proxy targets, never verified receipt events.
"""
from pathlib import Path
import json
import numpy as np
from .receipt_evidence import receipt_features, suppress_unsupported_pulses, FEATURE_NAMES
from .external import period_arrays
from .tracker_robustness import CONDITIONS, condition
from .research_snapshot import cached_case
from .control_state import evaluate_accuracy
from .ledger_experiment import short_hits
from .pipeline import digest
from . import tree_model

BRANCHES = ('distance_only', 'motion_receipt')


def samples(x, available, y, start, end):
    ix=np.arange(start,end,5)
    good=available[ix] & (x[ix,:,0] <= 2.5) & (y[ix,None] < 2)
    frame,team=np.nonzero(good)
    return x[ix[frame],team], (y[ix[frame]]==team).astype(np.int8)


def component_metrics(y, score, threshold):
    from sklearn.metrics import roc_auc_score, average_precision_score
    positive=y==1;negative=~positive;pred=score>=threshold
    return {'points':len(y),'owning_team_candidates':int(positive.sum()),'opponent_decoy_candidates':int(negative.sum()),
            'owning_candidate_retention':float(pred[positive].mean()) if positive.any() else None,
            'opponent_decoy_acceptance':float(pred[negative].mean()) if negative.any() else None,
            'roc_auc':float(roc_auc_score(y,score)) if positive.any() and negative.any() else None,
            'average_precision':float(average_precision_score(y,score)) if positive.any() else None,
            'brier':float(np.mean((score-y)**2))}


def run(root,cache):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    root=Path(root).resolve();cache=Path(cache);out=root/'local/receipt-064';out.mkdir(exist_ok=False)
    protocol={'version':'receipt-064','question':'Does approach/departure trajectory separate owning-team close-ball candidates from nearby opponents better than distance alone?',
              'label_limit':'Provider full H/A/D reference; label=near candidate team matches current live possession. This is ownership compatibility, NOT verified receipt/touch ground truth. False-labelled proxy examples may include genuine opponent touches without team-possession change.',
              'features':list(FEATURE_NAMES),'observations':'Nearest current player per team; same track slot +/-12frames (0.48s at25Hz), all intervening ball observations required, no interpolation/extrapolation, reject implausible >60m/s ball or >15m/s player trajectory. No IDs/names/labels in features. Offline only.',
              'folds':[{'fit_calibrate':'J03WMX','evaluate':'J03WN1'},{'fit_calibrate':'J03WN1','evaluate':'J03WMX'}],
              'split':'First70% of each training half for fitting, last30% for calibration,25frame embargo each side ofcut. Same temporal split across allfour tracking conditions. Every5thframe,visible valid nearest candidate within2.5m,known-live target only.',
              'fit':'Matched distance-only and all14feature HGB;120fixediterations,max15leaves,minleaf100,learningrate.05,L2=20,seed164,early_stopping=False. No sweeps,validation-based tuning or checkpoint selection.',
              'threshold':'Each model fifth percentile of positive calibration probabilities (method lower), targeting95% own-team close-candidate retention on external calibration only. Not calibrated event probability.',
              'component_acceptance':'Pooled opponent-decoy acceptance at least10% relatively lower than distance-only while own-candidate retention no more than2percentagepoints lower; per-fold retention no more than2points lower. Diagnostic only,not deployment.',
              'bounded_filter':'Inspect existing <=25frame H/A pulse only if flanked by same opposing team; all trajectory windows throughout pulse plusoneframe flank available forbothteams. Suppress only if no pulse-team score reaches calibratedthreshold and bothflanks support otherteam. No new switch, unchanged D mask, unavailable frames immutable. Short strong receipts preserved; no duration floor.',
              'pipeline_acceptance':'Misses nonincrease; subsecond GT matches nondecrease; FP decrease; mean spellF1 noninferior within.002; each noise fold no decline;clean/masked/long_gap fold decline<=.01. No model promotion until frozen different-draw audit passes.',
              'audit_seeds':{'masked':10516,'long_gap':10716,'tracker_noise':11120},
              'history':'Two heavily reused Bundesliga first-division games; development only. No challenge inputs,video,groundtruth,publicscore or new downloads.',
              'source_sha256':{p:digest(root/'controlswitch'/p) for p in ('receipt_evidence.py','receipt_experiment.py')}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    descriptor={'source':'short-spell-053','name':'short_focus'};folds=[];allrows=[];allbase=[];allpred={k:[] for k in BRANCHES};component={k:{'y':[],'p':[]} for k in BRANCHES}
    with threadpool_limits(limits=1):
      for fi,(train,val) in enumerate((('J03WMX','J03WN1'),('J03WN1','J03WMX'))):
        print('Receipt fold',fi,train,'->',val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();datasets={'fit':([],[]),'calibrate':([],[])}
        paths=list(cache.glob(train+'-*.npz'))
        if len(paths)!=1:raise ValueError('Ambiguous external cache')
        with np.load(paths[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
        for half in (0,1):
          seq=period_arrays(record,half);cut=int(.7*len(seq['y']))
          for kind in CONDITIONS:
            print('Training features',fi,half,kind,flush=True);s,mask=condition(seq,kind,half,True);x,v=receipt_features(s,mask)
            for role,start,end in [('fit',0,max(0,cut-25)),('calibrate',cut+25,len(x))]:
              xx,yy=samples(x,v,seq['y'],start,end);datasets[role][0].append(xx);datasets[role][1].append(yy)
        data={k:(np.concatenate(x),np.concatenate(y)) for k,(x,y) in datasets.items()};models={};calibration={};columns={'distance_only':[0],'motion_receipt':list(range(len(FEATURE_NAMES)))}
        for name in BRANCHES:
          cols=columns[name];x,y=data['fit'];model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,min_samples_leaf=100,learning_rate=.05,l2_regularization=20,random_state=164,early_stopping=False);model.fit(x[:,cols],y)
          tree_model.save(fd/(name+'.npz'),model,'hist_boost');saved=tree_model.load(fd/(name+'.npz'));cx,cy=data['calibrate'];p=model.predict_proba(cx[:,cols])[:,1];np.testing.assert_allclose(tree_model.predict(saved,cx[:,cols])[:,1],p,atol=1e-12,rtol=1e-12)
          threshold=float(np.quantile(p[cy==1],.05,method='lower'));calibration[name]={'threshold':threshold,'metrics':component_metrics(cy,p,threshold),'fit_points':len(y),'fit_positive':int(y.sum()),'model_sha256':digest(fd/(name+'.npz'))};models[name]=saved
          print('Calibrated',name,calibration[name],flush=True)
        (fd/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n');del data,datasets,record
        paths=list(cache.glob(val+'-*.npz'))
        if len(paths)!=1:raise ValueError('Ambiguous external cache')
        with np.load(paths[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
        rows=[];base=[];preds={k:[] for k in BRANCHES};comps={k:{'y':[],'p':[]} for k in BRANCHES};edits={k:[] for k in BRANCHES}
        for half in (0,1):
          seq=period_arrays(record,half)
          for ci,kind in enumerate(CONDITIONS):
            print('Evaluate receipt',fi,half,kind,flush=True);s,mask=condition(seq,kind,half,False);x,v=receipt_features(s,mask);y,current,q=cached_case(descriptor,fi,half*4+ci,root/'local');np.testing.assert_array_equal(y,seq['y']);xx,yy=samples(x,v,y,0,len(y));case={}
            for name in BRANCHES:
              cols=columns[name];scores=tree_model.predict(models[name],x.reshape(-1,len(FEATURE_NAMES))[:,cols])[:,1].reshape(len(x),2);cp=tree_model.predict(models[name],xx[:,cols])[:,1];comps[name]['y'].append(yy);comps[name]['p'].append(cp);p,changes=suppress_unsupported_pulses(current,scores,v,calibration[name]['threshold']);np.testing.assert_array_equal(p==2,current==2);np.testing.assert_array_equal(p[~v.all(1)],current[~v.all(1)]);case[name]=p;preds[name].append(p);edits[name].append({'half':half+1,'condition':kind,'changes':changes})
            rows.append({'y':y,'condition':kind});base.append(current);np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=y,current=current,**case)
        f={'train':train,'evaluate':val,'calibration':calibration,'reference':evaluate_accuracy(rows,base),'variants':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()},'component':{k:component_metrics(np.concatenate(comps[k]['y']),np.concatenate(comps[k]['p']),calibration[k]['threshold']) for k in BRANCHES},'edits':edits}
        (fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');folds.append(f);allrows+=rows;allbase+=base
        for k in BRANCHES:allpred[k]+=preds[k];component[k]['y']+=comps[k]['y'];component[k]['p'] += [(p>=calibration[k]['threshold']).astype(float) for p in comps[k]['p']]
    reference=evaluate_accuracy(allrows,allbase);short=sum(short_hits(r['y'],p) for r,p in zip(allrows,allbase));trials=[]
    pooled={k:component_metrics(np.concatenate(v['y']),np.concatenate(v['p']),.5) for k,v in component.items()}
    # Pooled scores are frozen threshold decisions from different fold models;
    # AUC/Brier on these binary decisions are intentionally omitted.
    for m in pooled.values():
      for key in ('roc_auc','average_precision','brier'):m.pop(key)
    a=pooled['distance_only'];b=pooled['motion_receipt'];cg=b['opponent_decoy_acceptance']<=.9*a['opponent_decoy_acceptance'] and b['owning_candidate_retention']>=a['owning_candidate_retention']-.02 and all(f['component']['motion_receipt']['owning_candidate_retention']>=f['component']['distance_only']['owning_candidate_retention']-.02 for f in folds)
    for name,p in allpred.items():
      score=evaluate_accuracy(allrows,p,allbase);sh=sum(short_hits(r['y'],v) for r,v in zip(allrows,p));guards={'misses':score['totals']['missed']<=reference['totals']['missed'],'short_possessions':sh>=short,'false_positives':score['totals']['false_positive']<reference['totals']['false_positive'],'mean_noninferior':score['mean_spell_f1']>=reference['mean_spell_f1']-.002,'noise':all(f['variants'][name]['tracker_noise']['spell_f1_mean']>=f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),'clean_gap':all(f['variants'][name][c]['spell_f1_mean']>=f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3])};trials.append({'name':name,'scores':score,'short_matched':sh,'guards':guards,'eligible':all(guards.values()) and (name!='motion_receipt' or cg)})
    eligible=[t for t in trials if t['eligible']];selected=max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None
    report={'protocol':protocol,'folds':folds,'reference':reference,'reference_short_matched':short,'pooled_component':pooled,'component_gate_passed':bool(cg),'trials':trials,'selected':selected,'promoted_to_default':False,'independent_test':False,'verified_receipts':False}
    (out/'freeze.json').write_text(json.dumps({'selected':selected,'models':[{k:f['calibration'][k]['model_sha256'] for k in BRANCHES} for f in folds],'before_new_noise_audit':True},indent=2)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPONENT',pooled,'gate',cg,flush=True)
    for t in trials:print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    return report

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--cache-root', type=Path)
    args = parser.parse_args()
    run(args.root, args.cache_root or args.root/'data/external/cache')
