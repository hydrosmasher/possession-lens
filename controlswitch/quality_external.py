"""Summarise external observation availability without possession labels."""
from pathlib import Path
import json
import numpy as np
from .observation_quality import quality_report
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition


def run(root,cache):
    rows=[]
    for mid in ('J03WMX','J03WN1'):
        files=list(Path(cache).glob(mid+'-*.npz'))
        if len(files)!=1:raise ValueError('Ambiguous external cache')
        with np.load(files[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
        for half in (0,1):
            seq=period_arrays(record,half)
            for kind in CONDITIONS:
                s,mask=condition(seq,kind,half,False);s={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in s.items() if k!='y'}
                if mask is not None:s['ball'][mask]=np.nan
                report=quality_report(s)
                rows.append({'match':mid,'half':half+1,'condition':kind,**report['summary']})
    result={'purpose':'Observation-only availability audit; not prediction confidence or verified events. No possession labels used. Conditions reuse the same two external games.','rows':rows,'possession_labels_used':False,'fitting_performed':False}
    (Path(root)/'reports/external/observation-quality-067.json').write_text(json.dumps(result,indent=2)+'\n');print('Saved16 observation-only cases.');return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--cache-root',type=Path);a=parser.parse_args();run(a.root,a.cache_root or a.root/'data/external/cache')
