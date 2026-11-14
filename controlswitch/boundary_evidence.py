"""Follow-up022: fixed evidence combinations, without refitting020 heads."""
from pathlib import Path
import argparse,json
import numpy as np
from .structured_had import supported_score
from .boundary_had import adaptive_decode
from .tracker_robustness import CONDITIONS,evaluate
from .pipeline import digest

VARIANTS=('minimum','geometric')
VERSION='boundary-evidence-022'

def combine_evidence(p,q,method):
    # Recover direction support independently of the event head; q=1 is neutral
    # here, not a label or a replacement event score.
    support=supported_score(p,np.ones(len(p)))
    q=np.asarray(q)
    if q.shape!=support.shape or not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('Invalid event evidence')
    if method=='minimum':return np.minimum(q,support)
    if method=='geometric':return np.sqrt(q*support)
    raise ValueError('Unknown evidence combination')

def run(previous,out):
    previous=Path(previous);out=Path(out)
    if out.exists():raise ValueError('Choose a new folder')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'history':'Follow-up after020/021 on same extensively reused Bundesliga validation. No independent test.','competition_data_used':False,'input_protocol_sha256':digest(previous/'protocol.json'),'frozen':'019 emissions;020 balanced event heads; .5 adaptive strength; stopcost3; no fitting','variants':list(VARIANTS),'evidence':'minimum(q,direction_support) or sqrt(q*direction_support); direction support fixed021 definition. Event scores are not calibrated probabilities.','selection':'Best mean four-condition spellF1 passing no aggregate misses/merges/FP increase; clean/ball-gap each fold no decline>.01; tracker_noise each fold no decline; deadF1 each fold/condition no decline>.005. Then frozen new-draw stability audit, not retuning.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');all_rows=[];all_current=[];all_preds={k:[] for k in VARIANTS};folds=[]
    for fold in (0,1):
        fd=out/f'fold-{fold}';fd.mkdir();rows=[];current=[];preds={k:[] for k in VARIANTS}
        for half in (0,1):
            for ci,c in enumerate(CONDITIONS):
                i=half*4+ci
                with np.load(previous/f'fold-{fold}/validation-{i}.npz',allow_pickle=False) as a:y=a['reference'];p=a['emissions'];q=a['boundary_score'];base=a['current']
                rows.append({'y':y,'condition':c});current.append(base)
                for k in VARIANTS:preds[k].append(adaptive_decode(p,combine_evidence(p,q,k),.5,'balanced'))
                np.savez_compressed(fd/f'validation-{i}.npz',reference=y,current=base,**{k:v[-1] for k,v in preds.items()})
        result={'validation':'J03WN1' if fold==0 else 'J03WMX','current':evaluate(rows,current),'variants':{k:evaluate(rows,p,current) for k,p in preds.items()}};folds.append(result);(fd/'report.json').write_text(json.dumps(result,indent=2)+'\n');all_rows+=rows;all_current+=current
        for k in VARIANTS:all_preds[k]+=preds[k]
    current=evaluate(all_rows,all_current);trials=[]
    for k in VARIANTS:
        score=evaluate(all_rows,all_preds[k],all_current);a=score['totals'];b=current['totals'];guards={'mean_f1':score['mean_spell_f1']>current['mean_spell_f1'],'misses':a['missed']<=b['missed'],'merges':a['merges']<=b['merges'],'false_positives':a['false_positive']<=b['false_positive'],'prior_conditions':all(f['variants'][k][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'tracker_noise':all(f['variants'][k]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),'dead_f1':all(f['variants'][k][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS)}
        trials.append({'method':k,'scores':score,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None,'best_unrestricted':max(trials,key=lambda t:t['scores']['mean_spell_f1']),'fresh_test_results':None,'promoted_to_default':False,'limitations':['Two reused matches and adaptive follow-ups; no unbiased test or statistical significance claim.','Small validation differences may be noise.','New random stress draws do not make a fresh match.','No provided game data, video, labels or public score used.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'])
    for t in trials:print(t['method'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'])
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.previous,a.out)
if __name__=='__main__':main()
