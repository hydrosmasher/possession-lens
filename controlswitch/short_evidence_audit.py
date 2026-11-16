"""068: external-only diagnosis of short-spell evidence versus decoding."""
from pathlib import Path
import json
import numpy as np
from .pipeline import pm
from .research_snapshot import cached_case
from .tracker_robustness import CONDITIONS
from .control_state import evaluate_accuracy


def diagnose(y,current,q):
    raw=np.asarray(q).argmax(1).astype(np.int8)
    # Preserve deployed D mask to isolate H/A smoothing rather than stoppages.
    raw_live=np.where(current==2,2,q[:,:2].argmax(1)).astype(np.int8)
    runs=pm.valid_runs(y);gt=pm.possession_spells(y,runs)
    def flags(p):return pm.matched_gt_flags(pm.spell_overlaps(gt,pm.possession_spells(p,runs)),len(gt),.5)
    old=flags(current);unsmoothed=flags(raw_live);prob=q[:,:2]/np.maximum(q[:,:2].sum(1,keepdims=True),1e-12);rows=[]
    for i,(a,b,team) in enumerate(gt):
        if b-a>75 or old[i]:continue
        majority=float(np.mean(prob[a:b,team]>.5));dead=float(np.mean(current[a:b]==2))
        kind='decoder_suppressed_recoverable' if unsmoothed[i] else 'dead_mask_blocks' if dead>=.5 else 'weak_team_evidence' if majority<.5 else 'fragmented_or_misaligned_evidence'
        rows.append({'start_frame':int(a),'end_frame_exclusive':int(b),'team':int(team),'cause':kind,'frames':b-a,'mean_correct_team_probability':float(prob[a:b,team].mean()),'correct_team_majority_fraction':majority,'predicted_dead_fraction':dead,'unsmoothed_matched':bool(unsmoothed[i])})
    return rows,raw_live


def run(root):
    root=Path(root);details=[];rows=[];base=[];raw=[]
    for fi in (0,1):
      for i in range(8):
        y,p,q=cached_case({'source':'short-spell-053','name':'short_focus'},fi,i,root/'local');d,r=diagnose(y,p,q);details.extend([{'fold':fi,'half':i//4+1,'condition':CONDITIONS[i%4],**x} for x in d]);rows.append({'y':y,'condition':CONDITIONS[i%4]});base.append(p);raw.append(r)
    counts={}
    for d in details:
        x=counts.setdefault(d['condition'],{});x[d['cause']]=x.get(d['cause'],0)+1
    result={'purpose':'External reference diagnostic only. No-smoothing counterfactual is NOT a proposed deployable algorithm. Frozen D mask isolates H/A smoothing; matching uses official IoU.5.','counts':counts,'missed_short_cases':details,'baseline':evaluate_accuracy(rows,base),'no_smoothing_diagnostic':evaluate_accuracy(rows,raw,base),'history':'Two heavily reused Bundesliga games repeated underfour conditions,not independent testing. No challenge inputs or labels.','promoted_to_default':False};(root/'reports/external/short-evidence-068.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(counts,indent=2));print('COUNTERFACTUAL',result['no_smoothing_diagnostic']['mean_spell_f1'],result['no_smoothing_diagnostic']['totals']);return result

if __name__=='__main__':run(Path.cwd())
