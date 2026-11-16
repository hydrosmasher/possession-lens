"""070: frozen local-expert proposals plus frozen receipt support."""
from pathlib import Path
import json,numpy as np
from .short_proposals import recover_short
from .receipt_evidence import receipt_features
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition
from .local_temporal_experiment import short_matches
from .control_state import evaluate_accuracy
from .pipeline import digest
from . import tree_model


def run(root,cache):
    root=Path(root).resolve();out=root/'local/short-proposals-070';out.mkdir(exist_ok=False);local=root/'local/local-temporal-069';receipt=root/'local/receipt-064';prior=json.loads((local/'report.json').read_text())
    protocol={'version':'short-proposals-070','motivation':'069recovers75shortmatchesbutFP129higher. Isolateboundednewshortspells rather thanreplacingglobalpredictions.',
              'inputs':'Frozen069local_expert labels;frozen064motion_receipt per-foldweights/thresholds,eachtrainedonlycorrespondingfittingmatch. No new fitting or threshold sweep.',
              'rule':'New1..75frameH/Aspell withoppositeproposalflanks;entirecandidateplusoneframeflanksinsideONEopposite053spell;ballseenallthoseframes. Receipt-supportedvariantrequiresatleastonecandidateframewithvalidtrajectory,nearestcandidate<=2.5m,and064score>=itsfrozen95percentretentioncalibrationthreshold. Missingballframes,D,andoutsidecandidateframesunchanged. No minimumdurationfloor.',
              'variants':['bounded_proposals','receipt_supported'],
              'selection':'Onlyreceipt_supported:mean gain>=.002,short<=3smatchesstrictincrease,misses/FPnonincrease,eachnoise foldnondecline,clean/masked/longgapfolddecline<=.01. Reportpairedunfilteredcontrol;freezebeforeaudit.',
              'audit':'Onlywinner,samegamesnewseeds10518/10718/11122,recompute053/069/064fromobservations;meanandshortmatchesimprove,misses/FPnonincrease,Dmask/missingframesexact;no retuning.',
              'history':'Adaptivefollow-upafter069ontworepeatedBundesligagames,notindependenttest. Receiptlabelsareownershipproxies,notverifiedtouches. Nochallengeinputs/video/labels/newdownloads.',
              'sources':{'local_report_sha256':digest(local/'report.json'),'receipt_report_sha256':digest(receipt/'report.json'),'rule_sha256':digest(root/'controlswitch/short_proposals.py')}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');names=('bounded_proposals','receipt_supported');allrows=[];allbase=[];allpred={k:[] for k in names};folds=[]
    for fi,mid in enumerate(('J03WN1','J03WMX')):
      fd=out/f'fold-{fi}';fd.mkdir();cal=json.loads((receipt/f'fold-{fi}/calibration.json').read_text())['motion_receipt'];path=receipt/f'fold-{fi}/motion_receipt.npz'
      if digest(path)!=cal['model_sha256']:raise ValueError('Receipt weights changed')
      model=tree_model.load(path);files=list(Path(cache).glob(mid+'-*.npz'))
      if len(files)!=1:raise ValueError('Ambiguous external cache')
      with np.load(files[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
      rows=[];base=[];preds={k:[] for k in names};reviews=[]
      for half in (0,1):
       seq=period_arrays(record,half)
       for ci,kind in enumerate(CONDITIONS):
        print('Proposal check',fi,half,kind,flush=True);s,mask=condition(seq,kind,half,False);x,valid=receipt_features(s,mask);scores=tree_model.predict(model,x.reshape(-1,14))[:,1].reshape(-1,2);seen=np.isfinite(s['ball']).all(1)
        if mask is not None:seen&=~mask
        with np.load(local/f'fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as a:y=a['reference'].copy();current=a['current'].copy();proposal=a['local_expert'].copy()
        np.testing.assert_array_equal(y,seq['y']);case={}
        for name,require in [('bounded_proposals',False),('receipt_supported',True)]:
          p,events=recover_short(current,proposal,seen,scores,valid,x[:,:,0]<=2.5,cal['threshold'],require_receipt=require);np.testing.assert_array_equal(p==2,current==2);np.testing.assert_array_equal(p[~seen],current[~seen]);preds[name].append(p);case[name]=p;reviews.append({'half':half+1,'condition':kind,'variant':name,'proposals':events})
        rows.append({'y':y,'condition':kind});base.append(current);np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=y,current=current,**case)
      f={'evaluate':mid,'reference':evaluate_accuracy(rows,base),'variants':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()},'reviews':reviews};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');allrows+=rows;allbase+=base
      for k in names:allpred[k]+=preds[k]
    reference=evaluate_accuracy(allrows,allbase);short=sum(short_matches(r['y'],p) for r,p in zip(allrows,allbase));trials=[]
    for name,p in allpred.items():
      score=evaluate_accuracy(allrows,p,allbase);sh=sum(short_matches(r['y'],v) for r,v in zip(allrows,p));guards={'spell_gain':score['mean_spell_f1']>=reference['mean_spell_f1']+.002,'short_recovery':sh>short,'misses':score['totals']['missed']<=reference['totals']['missed'],'false_positives':score['totals']['false_positive']<=reference['totals']['false_positive'],'noise':all(f['variants'][name]['tracker_noise']['spell_f1_mean']>=f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),'clean_gap':all(f['variants'][name][c]['spell_f1_mean']>=f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3])};trials.append({'name':name,'scores':score,'short_matched':sh,'guards':guards,'eligible':name=='receipt_supported' and all(guards.values())})
    selected=next((t for t in trials if t['eligible']),None);report={'protocol':protocol,'folds':folds,'reference':reference,'reference_short_matched':short,'trials':trials,'selected':selected,'promoted_to_default':False,'independent_test':False,'frozen_audit_required':bool(selected)};(out/'freeze.json').write_text(json.dumps({'selected':selected,'source_protocol_sha256':digest(out/'protocol.json'),'before_new_draw_audit':True},indent=2)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials:print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],'short',t['short_matched'],t['guards'],flush=True)
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--cache-root',type=Path);a=p.parse_args();run(a.root,a.cache_root or a.root/'data/external/cache')
