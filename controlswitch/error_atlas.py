"""External-only root-cause audit; reference labels are never model features."""
from pathlib import Path
import json
import numpy as np
from .pipeline import pm
from .research_snapshot import cached_case
from .tracker_robustness import CONDITIONS, condition
from .external import period_arrays


def audit_case(y, prediction, observed):
    runs=pm.valid_runs(y);gt=pm.possession_spells(y,runs);pred=pm.possession_spells(prediction,runs)
    overlaps=pm.spell_overlaps(gt,pred);hits=pm.matched_gt_flags(overlaps,len(gt),.5)
    out=[]
    for gi,(a,b,team) in enumerate(gt):
        if hits[gi]:continue
        coverage=float(np.mean(prediction[a:b]==team));parts=[(iou,pi) for iou,g,pi in overlaps if g==gi and iou>0]
        best=max(parts,default=(0.,None));pi=best[1];best_pred=None if pi is None else pred[pi];num_segments=sum(1 for u,v,k in pred if k==team and u<b and v>a)
        if coverage < .25:root='wrong_or_missing_team'
        elif best_pred and (best_pred[1]-best_pred[0]) > 2*(b-a):root='overlong_merged_spell'
        elif coverage >= .5 and num_segments > 1:root='fragmented_spell'
        else:root='boundary_or_partial_coverage'
        length='under1s' if b-a<25 else '1to3s' if b-a<=75 else '3to10s' if b-a<=250 else 'over10s'
        out.append({'cause':root,'duration':length,'ball_coverage':'mostly_missing' if observed[a:b].mean()<.5 else 'mostly_observed','correct_team_coverage':coverage,'reference_frames':b-a,'best_iou':float(best[0])})
    return out


def run(root,cache):
    root=Path(root);rows=[];summary={}
    for fi,mid in enumerate(('J03WN1','J03WMX')):
        files=list(Path(cache).glob(mid+'-*.npz'))
        if len(files)!=1:raise ValueError('Ambiguous external cache')
        with np.load(files[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
        for half in (0,1):
            seq=period_arrays(record,half)
            for ci,kind in enumerate(CONDITIONS):
                s,mask=condition(seq,kind,half,False);seen=np.isfinite(s['ball']).all(1)
                if mask is not None:seen&=~mask
                y,p,q=cached_case({'source':'short-spell-053','name':'short_focus'},fi,half*4+ci,root/'local');items=audit_case(y,p,seen)
                for item in items:
                    rows.append({'condition':kind,'match':mid,'half':half+1,**item})
                    for view,key in [('cause',item['cause']),('duration',item['duration']),('ball_coverage',item['ball_coverage'])]:
                        group=summary.setdefault(kind,{}).setdefault(view,{});group[key]=group.get(key,0)+1
    report={'purpose':'Diagnose missed external spells at IoU.5 only. Categories are heuristic descriptions, not causal proof or deployable outputs. All labels remain external-reference diagnostics.','reference':'053','scope':'Two heavily reused Bundesliga matches repeated underfour tracking conditions; counts across conditions repeat real spells.','summary':summary,'missed_cases':rows,'competition_data_used':False}
    out=root/'reports/external/error-atlas-065.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2));return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--cache-root',type=Path);a=parser.parse_args();run(a.root,a.cache_root or a.root/'data/external/cache')
