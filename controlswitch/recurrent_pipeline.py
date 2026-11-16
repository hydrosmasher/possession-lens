"""Hash-pinned043 recurrent ensemble; local CPU inference, no fitting."""
from pathlib import Path
import argparse,json,platform
import numpy as np
import pandas as pd
from . import ranked_pipeline as base_pipeline
from .boundary_had import boundary_observations,adaptive_decode
from .model import probabilities
from .pipeline import digest,pm,N_FRAMES,BLOCKS,FPS,write_submission,intervals
from .statistics import team_statistics

ROOT=Path(__file__).resolve().parents[1]
VERSION='recurrent-possession-043'

def checked_path(root,asset):
    p=(root/asset['path']).resolve()
    if not p.is_relative_to(root) or digest(p)!=asset['sha256']:raise ValueError('Recurrent asset path/hash mismatch')
    return p

def load_pipeline(config,root=ROOT):
    root=Path(root).resolve();cfg=json.loads(Path(config).read_text())
    if cfg.get('version')!=VERSION or cfg.get('gru_mix')!=.5:raise ValueError('Unsupported frozen recurrent configuration')
    base_path=checked_path(root,cfg['base_config']);gru_path=checked_path(root,cfg['gru'])
    try:
        import torch
        from .gated_rnn import load_checkpoint
    except ImportError as e:raise RuntimeError('Use Python3.12 and install requirements-gru-inference-lock.txt') from e
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    base=base_pipeline.load_pipeline(base_path,root);gated=load_checkpoint(gru_path);c=gated['model'].config
    if c['cell']!='gru' or c['dropout']!=.25 or c.get('input_features')!=508 or c['hidden']!=64 or c['projection']!=64 or c['layers']!=2 or not c['bidirectional'] or not c['boundary']:raise ValueError('Expected frozen508feature64unit bidirectional GRU')
    return {'base':base,'gru':gated,'mix':cfg['gru_mix'],'config':cfg}

def infer(bundle,s,pitch,mask=None):
    from .gated_rnn import symmetric
    _,p=base_pipeline.infer(bundle['base'],s,pitch,mask);m=bundle['gru'];g,_=symmetric(m['model'],s,pitch,m['mean'],m['scale'],mask);g=g[:,:2]/np.maximum(g[:,:2].sum(1,keepdims=True),1e-12);live=p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12);mix=bundle['mix'];h=(1-mix)*live+mix*g;d=p[:,2];q=np.column_stack([h*(1-d[:,None]),d]);head=bundle['base']['models']['boundary'];event=probabilities(head,boundary_observations(s,pitch,q,mask))[:,1];return adaptive_decode(q,event,1.),q

def export(data_root,out,config,filename='possession.csv'):
    from .submission import read_observations,predict_halves,submission_filename
    out=Path(out);filename=submission_filename(filename)
    if out.exists():raise ValueError('Choose a new output folder; earlier predictions are preserved')
    bundle=load_pipeline(config);import torch
    s,pitch,names,files=read_observations(data_root);labels=predict_halves(bundle,s,pitch,inference=infer);reason=pm.validate_submission(pd.DataFrame({'frame':np.arange(N_FRAMES),'label':labels}),N_FRAMES)
    if reason:raise ValueError(reason)
    out.mkdir(parents=True);csv=out/filename;write_submission(str(csv),labels);stats={**team_statistics(labels,FPS),'teams':dict(zip(('H','A'),names)),'status':'Model estimates, not verified match statistics'};(out/'team-statistics.json').write_text(json.dumps(stats,indent=2)+'\n');(out/'timeline.json').write_text(json.dumps(intervals(labels))+'\n')
    from .observation_quality import quality_report
    (out/'observation-quality.json').write_text(json.dumps(quality_report(s,BLOCKS,FPS),indent=2)+'\n')
    manifest={'pipeline':VERSION,'candidate':bundle['config'].get('candidate','043'),'config_sha256':digest(config),'base_config':bundle['config']['base_config'],'assets':{**bundle['base']['config']['assets'],'gru':bundle['config']['gru']},'gru_mix':bundle['mix'],'input_assets':{k:{'sha256':digest(files[k]),'bytes':files[k].stat().st_size} for k in ('metadata','tracking')},'unused_assets':[k for k in files if k not in ('metadata','tracking')],'csv_filename':filename,'csv_sha256':digest(csv),'n_frames':N_FRAMES,'half_blocks':BLOCKS,'official_format_valid':True,'fitting_performed':False,'video_used':False,'ground_truth_used':False,'inference_device':'cpu','torch_threads':1,'deterministic_algorithms':True,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data-root','out','config'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--filename',default='possession.csv');a=p.parse_args();r=export(a.data_root,a.out,a.config,a.filename);print('Full CSV valid; SHA-256:',r['csv_sha256']);print('Provided-match accuracy and competition percentile not measured.')
if __name__=='__main__':main()
