"""Frozen ranking-first030 pipeline: all fitting/selection stays external."""
from pathlib import Path
import json,argparse,platform
import numpy as np
import pandas as pd
from .rnn_runtime import load_frozen
from .europe_context import observation_probabilities as rnn_probabilities
from .control_correction import control_features,swap_control_columns,blend_control
from .control_state import state_features,swap_state_columns
from .quality_gate import quality_features,gated_dead
from .relative_state import relative_features,swap_relative
from .collective_state import collective_state,swap_collective
from .joint_state import joint_features,swap_joint
from .deadball import observation_features,fuse
from .boundary_had import boundary_observations,adaptive_decode
from .model import probabilities,decode
from . import tree_model
from .pipeline import digest,pm,N_FRAMES,BLOCKS,FPS,write_submission,intervals
from .statistics import team_statistics

VERSION='ranked-possession-030'
ROOT=Path(__file__).resolve().parents[1]

SPECS=[
 {'name':'boundary_half','control':'old','dead':'blend','boundary':'balanced','strength':.5},
 {'name':'boundary_full','control':'old','dead':'blend','boundary':'balanced','strength':1.},
 {'name':'boundary_recovery','control':'old','dead':'blend','boundary':'recovery','strength':.5},
 {'name':'state','control':'state','dead':'blend'},
 {'name':'state_half','control':'state_half','dead':'blend'},
 {'name':'quality075','control':'old','dead':'quality','cap':.75},
 {'name':'quality1','control':'old','dead':'quality','cap':1.},
 {'name':'joint_state_quality1','control':'state','dead':'quality','cap':1.},
 {'name':'joint_half_quality075','control':'state_half','dead':'quality','cap':.75},
 {'name':'joint_state_quality1_boundary','control':'state','dead':'quality','cap':1.,'boundary':'balanced','strength':.5},
 {'name':'joint_state_quality1_boundary_full','control':'state','dead':'quality','cap':1.,'boundary':'balanced','strength':1.},
 {'name':'joint_relative_quality1_boundary_full','control':'relative','dead':'quality','cap':1.,'boundary':'balanced','strength':1.},
 {'name':'joint_collective_quality1_boundary_full','control':'collective','dead':'quality','cap':1.,'boundary':'balanced','strength':1.},
 {'name':'joint_motion_quality1_boundary_full','control':'joint','dead':'quality','cap':1.,'boundary':'balanced','strength':1.},
]
REFERENCE={'name':'reference019','control':'old','dead':'blend'}

def load_head(path):
    with np.load(path,allow_pickle=False) as a:head={k:a[k] for k in ('mean','scale','weights')}
    if head['mean'].shape!=(116,) or head['scale'].shape!=(116,) or head['weights'].shape!=(117,2) or (head['scale']<=0).any() or not all(np.isfinite(a).all() for a in head.values()):raise ValueError('Invalid boundary head')
    return head

def make_bundle(spec,paths):
    if spec not in SPECS+[REFERENCE]:raise ValueError('Unrecognised frozen pipeline specification')
    models={}
    for role,path in paths.items():models[role]=load_frozen(path) if role=='team' else load_head(path) if role=='boundary' else tree_model.load(path)
    if not models['team'].get('context') or models['team'].get('interaction_context'):raise ValueError('Expected frozen context RNN')
    return {'spec':spec,'models':models}

def infer(bundle,s,pitch,mask=None):
    spec=bundle['spec'];models=bundle['models'];team=rnn_probabilities(models['team']['model'],s,pitch,mask,context=True)
    if spec['control']=='old':x=control_features(s,pitch,mask);swapped=swap_control_columns(x)
    elif spec['control']=='relative':x=relative_features(s,pitch,mask);swapped=swap_relative(x)
    elif spec['control']=='collective':x=collective_state(s,pitch,mask);swapped=swap_collective(x)
    elif spec['control']=='joint':x=joint_features(s,pitch,mask);swapped=swap_joint(x)
    else:x=state_features(s,pitch,mask);swapped=swap_state_columns(x)
    a=tree_model.predict(models['control'],x);b=tree_model.predict(models['control'],swapped);h=blend_control(team,(a+b[:,[1,0,2]])*.5,.75)
    obs=observation_features(s,pitch,mask)
    if spec['dead']=='blend':d=tree_model.predict(models['dead_blend'],obs)[:,1]
    else:
        old=tree_model.predict(models['dead_old'],obs)[:,1];new=tree_model.predict(models['dead_noise'],obs)[:,1];quality=tree_model.predict(models['quality'],quality_features(s,mask))[:,1];d=gated_dead(old,new,quality,spec['cap'])
    p=fuse(h,d,.75)
    if spec.get('boundary'):
        q=probabilities(models['boundary'],boundary_observations(s,pitch,p,mask))[:,1];labels=adaptive_decode(p,q,spec['strength'],spec['boundary'])
    else:labels=decode(p,12,stop_penalty=3)
    return labels,p

def load_pipeline(config,root=ROOT):
    config=Path(config);root=Path(root).resolve();cfg=json.loads(config.read_text())
    if cfg.get('version')!=VERSION:raise ValueError('Unsupported ranked pipeline')
    spec=cfg['spec']
    if spec not in SPECS+[REFERENCE]:raise ValueError('Unrecognised frozen pipeline specification')
    required={'team','control'}|({'dead_blend'} if spec['dead']=='blend' else {'dead_old','dead_noise','quality'})|({'boundary'} if spec.get('boundary') else set())
    if set(cfg['assets'])!=required:raise ValueError('Unexpected or missing model assets')
    paths={}
    for role,asset in cfg['assets'].items():
        path=(root/asset['path']).resolve()
        if not path.is_relative_to(root) or digest(path)!=asset['sha256']:raise ValueError('Model path/hash mismatch: '+role)
        paths[role]=path
    bundle=make_bundle(spec,paths);bundle['config']=cfg;return bundle

def export(data_root,out,config,filename='possession.csv'):
    from .submission import read_observations,predict_halves,submission_filename
    out=Path(out);filename=submission_filename(filename)
    if out.exists():raise ValueError('Choose a new output folder')
    bundle=load_pipeline(config);s,pitch,names,files=read_observations(data_root);labels=predict_halves(bundle,s,pitch,inference=infer)
    reason=pm.validate_submission(pd.DataFrame({'frame':np.arange(N_FRAMES),'label':labels}),N_FRAMES)
    if reason:raise ValueError(reason)
    out.mkdir(parents=True);csv=out/filename;write_submission(str(csv),labels)
    stats={**team_statistics(labels,FPS),'teams':dict(zip(('H','A'),names)),'status':'Model estimates, not verified match statistics'}
    (out/'team-statistics.json').write_text(json.dumps(stats,indent=2)+'\n');(out/'timeline.json').write_text(json.dumps(intervals(labels))+'\n')
    manifest={'pipeline':VERSION,'candidate':bundle['config'].get('candidate','030'),'spec':bundle['spec'],'config_sha256':digest(config),'assets':bundle['config']['assets'],'input_assets':{k:{'sha256':digest(files[k]),'bytes':files[k].stat().st_size} for k in ('metadata','tracking')},'unused_assets':[k for k in files if k not in ('metadata','tracking')],'csv_filename':filename,'csv_sha256':digest(csv),'n_frames':N_FRAMES,'half_blocks':BLOCKS,'official_format_valid':True,'fitting_performed':False,'video_used':False,'ground_truth_used':False,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--config',required=True,type=Path);p.add_argument('--filename',default='possession.csv');a=p.parse_args();r=export(a.data_root,a.out,a.config,a.filename);print('Full CSV valid; SHA-256:',r['csv_sha256']);print('Accuracy/competition percentile not measured on provided match.')
if __name__=='__main__':main()
