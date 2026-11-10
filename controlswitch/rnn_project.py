from pathlib import Path
import json
import numpy as np
from .external import MATCHES,load_sportec
from .experiment import split_data,aggregate
from .rnn import train,predict,rnn_features,RNN_VERSION
from .pipeline import (discover,teams_from_metadata,read_tracking,N_FRAMES,GAMES,GAME,BLOCKS,
                       write_submission,pm,digest,intervals)
from .statistics import team_statistics,tracklet_statistics

def train_external(root,out,epochs=5,single_match_pilot=False):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose an empty RNN run folder')
    records={};sources={}
    for mid in MATCHES:
        files=list(Path(root).rglob(f'*positions_raw_observed*{mid}.xml'))
        if len(files)>1:raise ValueError(f'Duplicate external file for {mid}')
        if files:
            records[mid]=load_sportec(files[0],mid,Path('work/external/cache'))
            sources[mid]=str(records[mid]['source_sha256'])
        else:
            caches=list(Path(root).rglob(f'{mid}-*.npz'))
            if len(caches)>1:raise ValueError(f'Ambiguous external cache for {mid}')
            if caches:
                with np.load(caches[0],allow_pickle=False) as d:
                    if str(d['match_id'])!=mid:raise ValueError('External cache match ID mismatch')
                    records[mid]={k:d[k] for k in d.files}
                sources[mid]={'original_xml_sha256':str(records[mid]['source_sha256']),
                              'cache_sha256':digest(caches[0]),'input_kind':'previously parsed external Sportec cache'}
    if not records:raise ValueError('No supported external match files. Training on provided match files is not supported.')
    training,validation,unused_test,mode,split=split_data(records,single_match_pilot);del unused_test
    pitch=next(iter(records.values()))['pitch']
    sequences=[(rnn_features(s,pitch),s['y']) for s in training]
    model,history=train(sequences,epochs=epochs)
    result=aggregate([(s['y'],np.argmax(predict(model,rnn_features(s,pitch)),axis=1)) for s in validation])
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'rnn-model.npz',**model,version=np.array(RNN_VERSION))
    report={'architecture':RNN_VERSION,'hidden_units':len(model['bh']),'epochs':epochs,'fps':25,
            'training_loss':history,'external_sources_sha256':sources,'mode':mode,'split':split,
            'validation_only':True,'previously_viewed_test_not_scored':True,'validation_metrics':result,
            'competition_data_used_for_training':False,'model_sha256':digest(out/'rnn-model.npz'),
            'limitations':['Initial recurrent model; hyperparameters are not yet externally optimised.',
                           'Validation metrics are development evidence, not a fresh test or competition score.']}
    (out/'training-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def predict_provided(root,checkpoint,out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose an empty prediction folder')
    files=discover(root);teams,names=teams_from_metadata(files['metadata'])
    # The existing strict parser checks all IDs, dimensions and row counts first.
    read_tracking(files['tracking'],teams)
    xy=np.full((N_FRAMES,len(teams),2),np.nan,dtype=np.float32)
    speed=np.full((N_FRAMES,len(teams)),np.nan,dtype=np.float32)
    ball=np.full((N_FRAMES,2),np.nan,dtype=np.float32);bs=np.full(N_FRAMES,np.nan,dtype=np.float32)
    pitch=np.asarray(GAMES[GAME]['pitch'])
    with files['tracking'].open() as f:
        for i,line in enumerate(f):
            _,players,b=line.strip().split(':');p=np.array([[float(x) for x in g.split(',')] for g in players.rstrip(';').split(';')]);b=np.array([float(x) for x in b.split(',')])
            xy[i]=(p[:,:2]-.5)*pitch;speed[i]=p[:,2];ball[i]=(b[:2]-.5)*pitch;bs[i]=b[2]
    with np.load(checkpoint,allow_pickle=False) as d:
        if str(d['version'])!=RNN_VERSION:raise ValueError('Incompatible RNN model version')
        model={k:d[k] for k in ('mean','scale','wx','wh','bh','wy','by')}
    labels=np.full(N_FRAMES,'X',dtype='<U1')
    for a,b in BLOCKS:
        sequence={'xy':xy[a:b],'speed':speed[a:b],'teams':teams,'ball':ball[a:b],'ball_speed':bs[a:b]}
        labels[a:b]=np.array(['H','A','D'])[np.argmax(predict(model,rnn_features(sequence,pitch)),axis=1)]
    import pandas as pd
    reason=pm.validate_submission(pd.DataFrame({'frame':np.arange(N_FRAMES),'label':labels}),N_FRAMES)
    if reason:raise ValueError(reason)
    out.mkdir(parents=True,exist_ok=True);write_submission(str(out/'possession.csv'),labels)
    stats=team_statistics(labels)
    stats.update({'teams':{'H':names[0],'A':names[1]},'model_sha256':digest(checkpoint),
                  'prediction_sha256':digest(out/'possession.csv'),'evaluated_against_competition_truth':False})
    (out/'team-statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    proxy=tracklet_statistics(labels,xy,ball,teams)
    (out/'tracklet-proxies.json').write_text(json.dumps(proxy,indent=2)+'\n')
    (out/'timeline.json').write_text(json.dumps(intervals(labels))+'\n')
    return stats
