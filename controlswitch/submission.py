"""Generate a private full-length challenge CSV using frozen external models.

No labels, video, fitting, calibration, or network access are used.
"""
from pathlib import Path
import argparse,json,platform
import numpy as np
import pandas as pd
from .pipeline import discover,teams_from_metadata,read_tracking,N_FRAMES,BLOCKS,GAMES,GAME,FPS,digest,write_submission,pm,intervals
from .deadball import load_candidate,infer_candidate
from .statistics import team_statistics

ROOT=Path(__file__).resolve().parents[1]

def read_observations(data_root):
    files=discover(data_root);teams,names=teams_from_metadata(files['metadata'])
    read_tracking(files['tracking'],teams)  # strict IDs, shapes and row-count check
    pitch=np.asarray(GAMES[GAME]['pitch'])
    xy=np.full((N_FRAMES,len(teams),2),np.nan,np.float32);speed=np.full((N_FRAMES,len(teams)),np.nan,np.float32)
    ball=np.full((N_FRAMES,2),np.nan,np.float32);ball_speed=np.full(N_FRAMES,np.nan,np.float32)
    with files['tracking'].open() as f:
        for i,line in enumerate(f):
            _,players,b=line.strip().split(':')
            p=np.asarray([[float(v) for v in g.split(',')] for g in players.rstrip(';').split(';')]);b=np.asarray([float(v) for v in b.split(',')])
            xy[i]=(p[:,:2]-.5)*pitch;speed[i]=p[:,2];ball[i]=(b[:2]-.5)*pitch;ball_speed[i]=b[2]
    return {'xy':xy,'speed':speed,'teams':teams,'ball':ball,'ball_speed':ball_speed},pitch,names,files

def predict_halves(candidate,observations,pitch,blocks=BLOCKS,inference=None):
    inference=infer_candidate if inference is None else inference
    n=len(observations['ball']);labels=np.full(n,'X',dtype='<U1')
    previous_end=0
    for a,b in blocks:
        if not 0<=previous_end<=a<b<=n:raise ValueError('Invalid/overlapping half boundaries')
        s={k:v if k=='teams' else v[a:b] for k,v in observations.items()}
        pred,_=inference(candidate,s,pitch)
        if pred.shape!=(b-a,) or not np.isin(pred,[0,1,2]).all():raise ValueError('Invalid state predictions')
        labels[a:b]=np.array(['H','A','D'])[pred];previous_end=b
    return labels

def submission_filename(value):
    if not isinstance(value,str) or not value or '/' in value or '\\' in value or '\0' in value or Path(value).name!=value or Path(value).suffix.lower()!='.csv':
        raise ValueError('Submission filename must be a CSV filename without folders')
    return value

def run(data_root,out,settings,team_model,dead_model,control_model=None,filename="possession.csv"):
    filename=submission_filename(filename)
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output directory; earlier predictions are preserved')
    if control_model is None:
        candidate=load_candidate(settings,team_model,dead_model);inference=infer_candidate
    else:
        from .control_correction import load_candidate as load_hybrid,infer_candidate as infer_hybrid
        candidate=load_hybrid(settings,team_model,dead_model,control_model);inference=infer_hybrid
    observations,pitch,names,files=read_observations(data_root)
    labels=predict_halves(candidate,observations,pitch,inference=inference)
    reason=pm.validate_submission(pd.DataFrame({'frame':np.arange(N_FRAMES),'label':labels}),N_FRAMES)
    if reason:raise ValueError(reason)
    out.mkdir(parents=True);csv=out/filename;write_submission(str(csv),labels)
    stats={**team_statistics(labels,FPS),'teams':dict(zip(('H','A'),names)),'evaluated_against_competition_truth':False}
    (out/'team-statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    (out/'timeline.json').write_text(json.dumps(intervals(labels))+'\n')
    manifest={'csv_filename':filename,'method':candidate['settings'].get('experiment',candidate['settings']['version']),'csv_sha256':digest(csv),'format_valid':True,'n_frames':N_FRAMES,'blocks':BLOCKS,'settings_sha256':digest(settings),'team_model_sha256':digest(team_model),'dead_model_sha256':digest(dead_model),'used_assets':{k:{'sha256':digest(files[k]),'bytes':files[k].stat().st_size} for k in ('metadata','tracking')},'unused_assets':[k for k in files if k not in ('metadata','tracking')],'fitting_performed':False,'competition_ground_truth_used':False,'video_used':False,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    if control_model is not None:manifest['control_model_sha256']=digest(control_model)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--settings',type=Path,default=ROOT/'reports/external/deadball-014-settings.json')
    p.add_argument('--team-model',type=Path,default=ROOT/'models/rnn-europe-context-001-experimental.npz')
    p.add_argument('--dead-model',type=Path,default=ROOT/'models/archive/deadball-014-validation-only.npz')
    p.add_argument('--control-model',type=Path,help='Optional017 control correction; requires matching --settings')
    p.add_argument('--filename',default='possession.csv',help='Final team CSV filename; quote names containing spaces')
    a=p.parse_args();m=run(a.data_root,a.out,a.settings,a.team_model,a.dead_model,a.control_model,a.filename)
    print('Full-length CSV passed format validation; SHA-256:',m['csv_sha256'])
    print('Private output:',a.out.resolve(),'; accuracy not measured.')
if __name__=='__main__':main()
