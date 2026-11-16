"""075: bounded external event-phase pilot; not a validated Metrica-tag model."""
from pathlib import Path
import json,numpy as np
from .external_phase_proxy import tracking_clocks,event_phases,corrupt
from .episode_phase_gate import episode_filter
from .proposal_calibration import retention
from .local_temporal_experiment import short_matches
from .tracker_robustness import CONDITIONS
from .control_state import evaluate_accuracy
from .pipeline import digest


def run(root,raw,cache=None):
    root=Path(root).resolve();raw=Path(raw);cache=Path(cache) if cache is not None else root/'data/external/cache';out=root/'local/episode-phase-075';out.mkdir(exist_ok=False);aux=root/'data/external/auxiliary-074';protocol={'version':'episode-phase-075','source':'CCBY4.0 IDSSE/Sportec Bundesliga WMX/WN1, existing tracking plus pinned external provider events. No challenge assets used in experiment.',
      'purpose':'Test whether extra independent-type episode evidence can veto false070recoveries; provider possession-phase annotations are stronger than automatic Metrica tags, possibly share reference-annotation ancestry. This is an optimistic external signal pilot, not validated automatic-tag transfer.',
      'rule':'Onlyexisting070receipt-supportedshortepisodes;vetoonlyifALLepisodeframeshaveunopposedopposite-team phaseevidence. Unknown,conflicting,partiallycoveredepisodesunchanged. No newH/Aepisodes,Dlabels,missing-ballchangesorparametersfitted.',
      'proxy':'First/last Play,ShotAtGoal,BallClaiming perproviderBallPossessionPhase/team +/-1s. TrackingN/T clock only; no tracking BallStatus/Possession or reference labels construct cue.',
      'conditions':['native_event_proxy','corrupted_event_proxy'],'corruption':'Fixed30percentphase dropout,10percentteamflip,+/-2sjitter,seed175+fold*2+half. Samecorruptedcuesacrosstrackingconditions;not a measured model of Metrica errors.',
      'guards':'Reportmean spellF1,misses,FP,<=3smatches andsame070recoveryretention. Desiredzero lost070recoveries,FP<=053and<070,mean>=053+.002,eachnoisefoldnondecline. EvenpassingdoesnotapproveMetrica-tagdeployment.',
      'freeze':'Rule/noisefixedbefore external scoring;noparametersearch or challenge-score selection.',
      'history':'Twoheavilyreusedexternalgames,notfreshindependenttest. Eventproxy/reference mayhavecommonannotationorigin. No challengevideo/outputcomparison/training/tuning.',
      'source_sha256':{k:digest(root/'controlswitch'/k) for k in ('external_phase_proxy.py','episode_phase_gate.py','episode_phase_experiment.py')}};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');names=protocol['conditions'];allrows=[];allbase=[];allproto=[];allpred={k:[] for k in names};ret={k:{'prototype_recovered':0,'retained_recovered':0,'lost_recovered':0} for k in names};folds=[]
    for fi,mid in enumerate(('J03WN1','J03WMX')):
      print('Clock/source audit',mid,flush=True);tracking=raw/f'DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-{mid}.xml';events=(raw if mid=='J03WN1' else aux)/f'DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-{mid}.xml';clocks=tracking_clocks(tracking,mid);phases,audit=event_phases(events,mid,clocks);audit['event_sha256']=digest(events);fd=out/f'fold-{fi}';fd.mkdir();reviews=json.loads((root/f'local/short-proposals-070/fold-{fi}/report.json').read_text())['reviews'];rows=[];base=[];proto=[];preds={k:[] for k in names};actions=[]
      caches=list(cache.glob(mid+'-*.npz'))
      if len(caches)!=1:raise ValueError('Ambiguous cache')
      with np.load(caches[0],allow_pickle=False) as a:frame_ids=[a[f'{h}_frame_ids'].copy() for h in (0,1)]
      for half,section in enumerate(('firstHalf','secondHalf')):
       ids=frame_ids[half];clock=clocks[section]
       if int(ids[0])!=clock['first_frame'] or len(ids)!=clock['count']:raise ValueError('Event/tracking cache clocks differ')
       cues={'native_event_proxy':phases[section],'corrupted_event_proxy':corrupt(phases[section],len(ids),175+fi*2+half)}
       for ci,kind in enumerate(CONDITIONS):
        with np.load(root/f'local/short-proposals-070/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as a:y=a['reference'].copy();current=a['current'].copy();prototype=a['receipt_supported'].copy()
        review=next(v for v in reviews if v['half']==half+1 and v['condition']==kind and v['variant']=='receipt_supported');proposals=[e for e in review['proposals'] if e['accepted']];local={}
        for name in names:
          p,decisions=episode_filter(current,proposals,cues[name]);r=retention(y,current,prototype,p)
          for k,v in r.items():ret[name][k]+=v
          np.testing.assert_array_equal(p==2,current==2);preds[name].append(p);local[name]=p;actions.append({'half':half+1,'condition':kind,'cue':name,'vetoed':sum(e['phase_veto'] for e in decisions),'proposals':len(proposals),'retention':r})
        rows.append({'y':y,'condition':kind});base.append(current);proto.append(prototype);np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=y,current=current,prototype=prototype,**local)
      f={'match':mid,'source_audit':audit,'reference':evaluate_accuracy(rows,base),'prototype':evaluate_accuracy(rows,proto,base),'variants':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()},'actions':actions};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');allrows+=rows;allbase+=base;allproto+=proto
      for k in names:allpred[k]+=preds[k]
    reference=evaluate_accuracy(allrows,allbase);prototype=evaluate_accuracy(allrows,allproto,allbase);trials=[]
    for name,p in allpred.items():
      score=evaluate_accuracy(allrows,p,allbase);short=sum(short_matches(r['y'],v) for r,v in zip(allrows,p));guards={'same_recoveries':ret[name]['lost_recovered']==0,'fp_baseline':score['totals']['false_positive']<=reference['totals']['false_positive'],'fp_prototype_reduced':score['totals']['false_positive']<prototype['totals']['false_positive'],'mean_gain':score['mean_spell_f1']>=reference['mean_spell_f1']+.002,'noise':all(f['variants'][name]['tracker_noise']['spell_f1_mean']>=f['reference']['tracker_noise']['spell_f1_mean'] for f in folds)};trials.append({'name':name,'scores':score,'short_matched':short,'retention':ret[name],'guards':guards,'pilot_meets_guards':all(guards.values())})
    report={'protocol':protocol,'folds':folds,'reference':reference,'prototype':prototype,'trials':trials,'release_eligible':False,'promoted_to_default':False,'metrica_tag_transfer_validated':False,'independent_test':False};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials:print('PILOT',t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['retention'],t['guards'],flush=True)
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--cache-root',type=Path);a=p.parse_args();run(a.root,a.raw_root,a.cache_root)
