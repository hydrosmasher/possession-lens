"""Duration-stratified external spell recall diagnostics; no model fitting."""
from pathlib import Path
import json,numpy as np
from .pipeline import pm
from .tracker_robustness import CONDITIONS

BINS=(0,.4,1,2,5,10,30,float('inf'))
def audit_arrays(y,pred):
    runs=pm.valid_runs(y);gt=pm.possession_spells(y,runs);ps=pm.possession_spells(pred,runs);flags=pm.matched_gt_flags(pm.spell_overlaps(gt,ps),len(gt),.5);result=[]
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        ix=[i for i,(a,b,_) in enumerate(gt) if lo<=(b-a)/25<hi]
        result.append({'duration_seconds':f'{lo:g}–{hi:g}','reference_spells':len(ix),'matched_at_iou_05':int(flags[ix].sum()) if ix else 0})
    return result

def run(root,out):
    root=Path(root);rows={c:None for c in CONDITIONS}
    for fold in (0,1):
        for half in (0,1):
            for ci,c in enumerate(CONDITIONS):
                with np.load(root/f'fold-{fold}/validation-{half*4+ci}.npz',allow_pickle=False) as a:r=audit_arrays(a['reference'],a['aligned_full'])
                if rows[c] is None:rows[c]=r
                else:
                    for total,row in zip(rows[c],r):
                        for k in ('reference_spells','matched_at_iou_05'):total[k]+=row[k]
    for bins in rows.values():
        for b in bins:b['recall_at_iou_05']=b['matched_at_iou_05']/b['reference_spells'] if b['reference_spells'] else None
    result={'candidate':'031','source':'Reused opposite-match external predictions only','diagnostic_iou':.5,'official_metric':'Official spellF1 averages multiple IoU thresholds; these are duration-stratified recall diagnostics, not that score','conditions':rows,'repeated_physical_possessions_across_conditions':True};Path(out).write_text(json.dumps(result,indent=2)+'\n');return result
