"""072: fit/calibrate only fitting-match proposals; evaluate opposite match."""
from pathlib import Path
import json,numpy as np,torch
from .proposal_calibration import observation_case,targets,choose_threshold,apply_scores,retention,FEATURE_NAMES
from .research_snapshot import make_snapshot,cached_case
from .local_temporal import load as load_local
from .tracker_robustness import CONDITIONS,condition
from .external import period_arrays
from .model import fit_classifier,probabilities
from .control_state import evaluate_accuracy
from .local_temporal_experiment import short_matches
from .pipeline import digest
from . import tree_model


def run(root,cache):
    root=Path(root).resolve();out=root/'local/proposal-calibration-072';out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);roots=json.loads((root/'local/roots.json').read_text());desc={'source':'short-spell-053','name':'short_focus','control_source':'short-spell-053','control_name':'short_focus'}
    protocol={'version':'proposal-calibration-072','aim':'Reduce070falsepositives while retaining every same GT spell recovered by070, not just recovery count.',
      'fitting_split':'For each opposite-match fold, first half of fitting match fits proposal scorer; second half calibrates cutoff. Each includesfour fixed training-corruption conditions. No opposite evaluation-match labels enter fit, scaler, threshold, or calibration selection.',
      'generator_limit':'Frozen053/064/069generators previously fitted BOTH halves of fitting match. Calibration excludes scorer fitting only, not generator fitting: in-sample generation can be optimistic. Opposite-game evaluation is heavily reused development, not independent testing.',
      'features':list(FEATURE_NAMES),'features_policy':'17 identity-free relative-to-proposed-team probability/receipt/motion/duration/context summaries. No reference labels/timestamps in features.',
      'classifier':'Regularised binary logistic model60fixedepochs,L2=.05,seed172+fold,training-only scaler. Fit labels: individual insertion recovers at leastone previouslymissed officialIoU.5 GTspell; labels external only. Unknown reference windows excluded. If<10examples or missingclass, no filtering.',
      'calibration':'Enumerate only calibration proposal score cutoffs; zero lost individual070-recovered GTspells required; calibrationFP no worse than070; select fewestFP thenhighestcutoff. Freeze per-fold model hash+threshold BEFORE opposite-match evaluation. No runner-up thresholds after evaluation.',
      'acceptance':'Exactzero loss of070-recoveredindividualspells;FP strictly below070 and<=053;totalmisses<=053;meanF1>=053+.002;shortmatches>053;eachnoisefoldnondecline;clean/gapfolddecline<=.01. AllD/missingframes unchanged. Promotion also requires frozennewdrawaudit.',
      'audit_seeds':{'masked':10519,'long_gap':10719,'tracker_noise':11123},'history':'ExistingtwoBundesliga matches;no challengeinputs/video/labels/publicscore,newdownloadsorverifiedreceiptclaims. Adaptiveafter069–071; notindependenttest.',
      'source_sha256':{name:digest(root/'controlswitch'/name) for name in ('proposal_calibration.py','proposal_calibration_experiment.py')}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');allrows=[];allbase=[];allproto=[];allfiltered=[];folds=[];totalret={'prototype_recovered':0,'retained_recovered':0,'lost_recovered':0}
    for fi,(train,val) in enumerate((('J03WMX','J03WN1'),('J03WN1','J03WMX'))):
      fd=out/f'fold-{fi}';fd.mkdir();snapshot=make_snapshot(desc,fi,roots,root/'local');local=load_local(root/f'local/local-temporal-069/fold-{fi}/local-expert.npz');receipt_root=root/f'local/receipt-064/fold-{fi}';rc=json.loads((receipt_root/'calibration.json').read_text())['motion_receipt'];receipt=tree_model.load(receipt_root/'motion_receipt.npz')
      if digest(receipt_root/'motion_receipt.npz')!=rc['model_sha256']:raise ValueError('Receipt weights changed')
      files=list(Path(cache).glob(train+'-*.npz'))
      if len(files)!=1:raise ValueError('Ambiguous fitting data')
      with np.load(files[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
      cases={'fit':[],'calibrate':[]}
      for half,role in ((0,'fit'),(1,'calibrate')):
       seq=period_arrays(record,half)
       for ci,kind in enumerate(CONDITIONS):
        print('Training-only proposals',fi,role,kind,flush=True);s,mask=condition(seq,kind,half,True);c=observation_case(snapshot,local,receipt,rc['threshold'],s,record['pitch'],mask);c['y']=seq['y'];c['condition']=kind
        keep=[j for j,e in enumerate(c['events']) if (seq['y'][e['start']-1:e['end']+1]<3).all()];c['x']=c['x'][keep];c['events']=[c['events'][j] for j in keep];c['prototype']=apply_scores(c['current'],c['events'],np.ones(len(keep)),0);c['target']=targets(seq['y'],c);cases[role].append(c);np.savez_compressed(fd/f'{role}-{ci}.npz',x=c['x'],target=c['target']);print('Training proposal counts',fi,role,kind,len(keep),int(c['target'].sum()),flush=True)
      x=np.concatenate([c['x'] for c in cases['fit']]);y=np.concatenate([c['target'] for c in cases['fit']]);fallback=len(y)<10 or len(np.unique(y))<2
      if fallback:scorer=None;threshold=0.;calibration={'status':'Too few fitting examples/classes; preserve070withoutfiltering'}
      else:
        scorer=fit_classifier(x,y,classes=2,epochs=60,seed=172+fi,l2=.05);np.savez_compressed(fd/'proposal-scorer.npz',**scorer);scores=[probabilities(scorer,c['x'])[:,1] if len(c['x']) else np.empty(0) for c in cases['calibrate']];threshold,calibration=choose_threshold(cases['calibrate'],scores)
      freeze={'train_match':train,'fit_half':1,'calibration_half':2,'fit_examples':len(y),'fit_positive_recovery_examples':int(y.sum()),'calibration_examples':sum(len(c['events']) for c in cases['calibrate']),'threshold':threshold,'no_filter_fallback':fallback,'calibration':calibration,'model_sha256':None if fallback else digest(fd/'proposal-scorer.npz'),'before_opposite_match_evaluation':True};(fd/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print('FROZEN FILTER',fi,freeze['fit_examples'],freeze['calibration_examples'],threshold,flush=True);del cases,record
      files=list(Path(cache).glob(val+'-*.npz'))
      if len(files)!=1:raise ValueError('Ambiguous evaluation data')
      with np.load(files[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
      rows=[];base=[];proto=[];filtered=[];reviews=[];foldret={k:0 for k in totalret}
      for half in (0,1):
       seq=period_arrays(record,half)
       for ci,kind in enumerate(CONDITIONS):
        print('Frozen filter evaluation',fi,half,kind,flush=True);s,mask=condition(seq,kind,half,False);y,current,q=cached_case(desc,fi,half*4+ci,root/'local')
        with np.load(root/f'local/precision-055-cache/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as a:event=a['event'].copy()
        c=observation_case(snapshot,local,receipt,rc['threshold'],s,record['pitch'],mask,current=current,q=q,event=event)
        with np.load(root/f'local/short-proposals-070/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as a:np.testing.assert_array_equal(c['prototype'],a['receipt_supported'])
        scores=np.ones(len(c['x'])) if scorer is None else probabilities(scorer,c['x'])[:,1];p=apply_scores(current,c['events'],scores,threshold);np.testing.assert_array_equal(p==2,current==2);np.testing.assert_array_equal(p[~c['seen']],current[~c['seen']]);ret=retention(y,current,c['prototype'],p)
        for k,v in ret.items():foldret[k]+=v
        rows.append({'y':y,'condition':kind});base.append(current);proto.append(c['prototype']);filtered.append(p);reviews.append({'half':half+1,'condition':kind,'events':c['events'],'scores':scores.tolist(),'kept':int((scores>=threshold).sum()),'retention':ret});np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=y,current=current,prototype=c['prototype'],filtered=p)
      f={'train':train,'evaluate':val,'freeze':freeze,'reference':evaluate_accuracy(rows,base),'prototype':evaluate_accuracy(rows,proto,base),'filtered':evaluate_accuracy(rows,filtered,base),'retention':foldret,'reviews':reviews};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');allrows+=rows;allbase+=base;allproto+=proto;allfiltered+=filtered
      for k,v in foldret.items():totalret[k]+=v
    reference=evaluate_accuracy(allrows,allbase);prototype=evaluate_accuracy(allrows,allproto,allbase);candidate=evaluate_accuracy(allrows,allfiltered,allbase);short0=sum(short_matches(r['y'],p) for r,p in zip(allrows,allbase));short=sum(short_matches(r['y'],p) for r,p in zip(allrows,allfiltered));guards={'same_recoveries':totalret['lost_recovered']==0,'false_positive_reduction':candidate['totals']['false_positive']<prototype['totals']['false_positive'],'fp_not_above_baseline':candidate['totals']['false_positive']<=reference['totals']['false_positive'],'misses':candidate['totals']['missed']<=reference['totals']['missed'],'spell_gain':candidate['mean_spell_f1']>=reference['mean_spell_f1']+.002,'short_gain':short>short0,'noise':all(f['filtered']['tracker_noise']['spell_f1_mean']>=f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),'clean_gap':all(f['filtered'][c]['spell_f1_mean']>=f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3])};report={'protocol':protocol,'folds':folds,'reference':reference,'prototype':prototype,'candidate':candidate,'retention':totalret,'short_reference':short0,'short_candidate':short,'guards':guards,'eligible':all(guards.values()),'promoted_to_default':False,'independent_test':False,'frozen_audit_required':all(guards.values())};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('RESULT',candidate['mean_spell_f1'],candidate['totals'],'short',short,'retention',totalret,guards,flush=True);return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--cache-root',type=Path);a=p.parse_args();run(a.root,a.cache_root or a.root/'data/external/cache')
