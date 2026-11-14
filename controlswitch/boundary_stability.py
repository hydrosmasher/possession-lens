"""023 frozen boundary-method audit under new simulated observation draws."""
from pathlib import Path
import argparse,json
import numpy as np
from .boundary_had import frozen_emissions,boundary_observations,adaptive_decode
from .structured_had import predict
from .tracker_robustness import CONDITIONS,evaluate,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .external import period_arrays
from .rnn_runtime import load_frozen
from .model import probabilities,decode
from .pipeline import digest
from . import tree_model

def run(cache_root,control_root,dead_root,event_root,team_model,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new directory')
    out.mkdir(parents=True)
    protocol={'experiment':'boundary-stability-023','history':'Unchanged reused matches, new random observation draws only; not fresh-match testing','competition_data_used':False,'frozen_methods':['019 reference','021 supported boundaries','020 balanced_full higher-recall alternative'],'selection':'Diagnostic only; no fitting, decoder changes or candidate re-selection from these results','new_seeds':{'masked':1214,'long_gap':1414,'tracker_noise':1818},'clean':'Same unmodified observations as previous studies','team_sha256':digest(team_model),'event_protocol_sha256':digest(Path(event_root)/'protocol.json')}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');backbone=load_frozen(team_model);rows=[];preds={k:[] for k in ('current','conservative','aggressive')}
    for fold,mid in enumerate(('J03WN1','J03WMX')):
        f=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(f)!=1:raise ValueError('Ambiguous cache')
        with np.load(f[0],allow_pickle=False) as a:d={k:a[k] for k in a.files}
        control=tree_model.load(Path(control_root)/f'fold-{fold}/control-estimator.npz');dead=tree_model.load(Path(dead_root)/f'fold-{fold}/dead-blend-0.5.npz')
        with np.load(Path(event_root)/f'fold-{fold}/boundary-head.npz',allow_pickle=False) as a:head={k:a[k] for k in ('mean','scale','weights')}
        for half in (0,1):
            original=period_arrays(d,half);pitch=d['pitch']
            for c in CONDITIONS:
                print('Audit',fold,half,c,flush=True);mask=None;s={k:v for k,v in original.items() if k!='y'}
                if c=='masked':mask=missing_mask(len(original['y']),1214+half)
                elif c=='long_gap':mask=long_mask(len(original['y']),1414+half)
                elif c=='tracker_noise':s=noisy_tracking(original,1818+half)
                p=frozen_emissions(backbone,control,dead,s,pitch,mask);q=probabilities(head,boundary_observations(s,pitch,p,mask))[:,1];rows.append({'y':original['y'],'condition':c});preds['current'].append(decode(p,12,stop_penalty=3));preds['conservative'].append(predict(p,q,'supported_boundaries'));preds['aggressive'].append(adaptive_decode(p,q,1.,'balanced'))
    current=evaluate(rows,preds['current']);report={'protocol':protocol,'current':current,'conservative':evaluate(rows,preds['conservative'],preds['current']),'aggressive':evaluate(rows,preds['aggressive'],preds['current']),'fresh_test_results':None,'promoted_to_default':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name in ('current','conservative','aggressive'):print(name,report[name]['mean_spell_f1'],report[name]['totals'])
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','control-root','dead-root','event-root','team-model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,a.control_root,a.dead_root,a.event_root,a.team_model,a.out)
if __name__=='__main__':main()
