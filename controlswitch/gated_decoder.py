"""Reproduce the Nov12 in-memory decoder study; no training or fresh test.

The grid was already viewed on validation. This run saves/reproduces that study,
not a newly preregistered independent experiment. Original gated-003 stays intact.
"""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from .gated_rnn import load_checkpoint,symmetric,save_checkpoint
from .skillcorner_europe import load_match,digest
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import spell_diagnostics,dwell_decode
from .europe_recovery import totals
from .model import decode

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def full_pulses(diag):return sum(diag[c]['full_prediction_spells_under_0_4_seconds'] for c in ('clean','masked'))

def run(raw,checkpoint,out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new output directory')
    out.mkdir(parents=True);torch.set_num_threads(1)
    protocol={'study':'gated-decoder-004','training_performed':False,'validation':['2269'],'test_loaded':False,'competition_data_used':False,
              'status':'reproduction of already-viewed in-memory validation study; not preregistered new evidence',
              'weights_sha256':digest(checkpoint),'validation_seed':100,'dwell_floors_frames':[1,5,10],
              'penalties':[20,24,28],'strengths':[.75,1.],'guards':'misses<=318, false positives<=115, merge overlaps<=66, full-sequence short predictions=0',
              'selection':'highest eligible mean clean/masked spell F1 above saved gated reference; no promotion'}
    dump(out/'protocol.json',protocol)
    parts,_=load_match(raw,'2269');rows=conditions(parts,100);f=load_checkpoint(checkpoint)
    values=[symmetric(f['model'],s,p,f['mean'],f['scale'],mask) for s,p,c,mask in rows]
    def score(pred):return {**evaluate(rows,pred),'diagnostics':spell_diagnostics(rows,pred)}
    reference=score([decode(p,f['penalty'],b,f['boundary_strength']) for p,b in values]);dwell=[]
    for floor in protocol['dwell_floors_frames']:
        s=score([dwell_decode(p,f['penalty'],b,f['boundary_strength'],floor) for p,b in values]);dwell.append({'minimum_frames':floor,**s})
    trials=[];best=None
    for penalty in protocol['penalties']:
        for strength in protocol['strengths']:
            s=score([decode(p,penalty,b,strength) for p,b in values]);t=totals(s['diagnostics'])
            guards={'misses':t['missed_reference_spells_at_iou_0_5']<=318,'false_positives':t['false_positive_spells_at_iou_0_5']<=115,'merge_overlaps':t['predictions_overlapping_multiple_reference_spells']<=66,'full_short_predictions':full_pulses(s['diagnostics'])==0}
            row={'penalty':penalty,'boundary_strength':strength,'variant':f['config']['cell']+' boundary decoder',**s,'guards':guards,'eligible':all(guards.values())};trials.append(row)
            if row['eligible'] and row['selection_spell_f1']>reference['selection_spell_f1'] and (best is None or row['selection_spell_f1']>best['selection_spell_f1']):best=row
    if best:
        save_checkpoint(out/'candidate.npz',f['model'],f['mean'],f['scale'],best['penalty'],best['boundary_strength'])
        # Check metadata-only decoding update: every original tensor stays bit-identical.
        with np.load(checkpoint,allow_pickle=False) as a,np.load(out/'candidate.npz',allow_pickle=False) as b:
            assert all(np.array_equal(a[k],b[k]) for k in a.files if k!='meta')
        restored=load_checkpoint(out/'candidate.npz');pred=[decode(p,restored['penalty'],b,restored['boundary_strength']) for p,b in values];assert score(pred)=={k:best[k] for k in ['clean','masked','selection_spell_f1','diagnostics']}
    report={'protocol':protocol,'reference_validation':{k:v for k,v in reference.items() if k!='diagnostics'},'reference_diagnostics':reference['diagnostics'],'selected':best,'trials':trials,'dwell_trials':dwell,'fresh_test_results':None,'promoted_to_default':False,
            'checkpoint_sha256':digest(out/'candidate.npz') if best else None,'weights_unchanged':True,'limitations':['Repeated validation selection is not independent test evidence.','Partial provider H/A labels do not establish full-match or challenge accuracy.','Geometric merges and short-run counts are descriptive diagnostics, not verified events.']}
    dump(out/'report.json',report);dump(out/'validation-trials.json',trials)
    print(json.dumps({'reference':reference['selection_spell_f1'],'selected':None if best is None else best['selection_spell_f1'],'short_predictions':None if best is None else full_pulses(best['diagnostics'])}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['raw','checkpoint','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.raw,a.checkpoint,a.out)
