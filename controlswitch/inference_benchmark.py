"""Reproduce exact019 inference optimisation benchmarks on external tracking."""
from pathlib import Path
import argparse,json,time,statistics
import numpy as np
from .control_correction import load_candidate,infer_candidate,control_features,blend_control
from .rnn_runtime import infer_frozen
from .deadball import observation_features,fuse
from .model import decode
from .external import period_arrays
from .tracker_robustness import condition
from . import tree_model

ROOT=Path(__file__).resolve().parents[1]

def reference_inference(candidate,s,pitch,mask):
    """Unoptimised019 path, retained only for equivalence/timing checks."""
    _,team=infer_frozen(candidate['team'],s,pitch,mask)
    a=tree_model.predict(candidate['correction'],control_features(s,pitch,mask))
    b=tree_model.predict(candidate['correction'],control_features(s,pitch,mask,True))
    correction=(a+b[:,[1,0,2]])*.5
    d=tree_model.predict(candidate['dead'],observation_features(s,pitch,mask))[:,1];cfg=candidate['settings'];p=fuse(blend_control(team,correction,cfg['mix']),d,cfg['dead_weight'])
    return decode(p,cfg['penalty'],stop_penalty=cfg['stop_penalty']),p

def run(cache,out,frames=5000,repeats=3):
    out=Path(out)
    if out.exists():raise ValueError('Use a new output JSON')
    if not 1<=frames<=20000 or not 1<=repeats<=10:raise ValueError('Invalid benchmark size')
    with np.load(cache,allow_pickle=False) as a:data={k:a[k] for k in a.files}
    if str(data['match_id']) not in ('J03WMX','J03WN1'):raise ValueError('Use an audited external Bundesliga cache')
    candidate=load_candidate(ROOT/'reports/external/deadblend-019-settings.json',ROOT/'models/rnn-europe-context-001-experimental.npz',ROOT/'models/archive/deadblend-019-validation-only.npz',ROOT/'models/archive/control-017-validation-only.npz')
    original=period_arrays(data,0);s={k:v if k in ('teams','period') else v[:frames] for k,v in original.items() if k!='y'};rows=[]
    for c in ('clean','long_gap','tracker_noise'):
        obs,mask=condition(s,c,0,False);a,p=reference_inference(candidate,obs,data['pitch'],mask);b,q=infer_candidate(candidate,obs,data['pitch'],mask);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(p,q);times={'old':[],'new':[]}
        for rep in range(repeats):
            calls=[('old',reference_inference),('new',infer_candidate)]
            if rep%2:calls.reverse()
            for name,fn in calls:
                start=time.perf_counter();fn(candidate,obs,data['pitch'],mask);times[name].append(time.perf_counter()-start)
        old=statistics.median(times['old']);new=statistics.median(times['new']);rows.append({'condition':c,'frames':len(s['ball']),'probabilities_exact':True,'decoded_labels_exact':True,'old_median_seconds':old,'new_median_seconds':new,'speed_ratio':old/new,'samples':times})
    result={'experiment':'exact-inference-optimisation-025','rows':rows,'all_probabilities_bitwise_equal':True,'limitations':['Local probes and same machine/environment, not a guarantee on other hardware or load.','Runtime optimisation only; model accuracy is unchanged.']};out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--frames',type=int,default=5000);p.add_argument('--repeats',type=int,default=3);a=p.parse_args();run(a.cache,a.out,a.frames,a.repeats)
if __name__=='__main__':main()
