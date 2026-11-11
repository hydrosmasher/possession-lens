"""Sportec external-match loader. Labels are kept separate from observable features."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
from .pipeline import digest

MATCHES = {
 'J03WOH': {'home':'DFL-CLU-00000P','away':'DFL-CLU-000011','competition':'000002'},
 'J03WPY': {'home':'DFL-CLU-00000P','away':'DFL-CLU-000005','competition':'000002'},
 'J03WMX': {'home':'DFL-CLU-000008','away':'DFL-CLU-00000G','competition':'000001'},
}
# Kept separate so adding a fresh test cannot change the original three-match split.
FRESH_MATCHES = {
 'J03WN1': {'home':'DFL-CLU-00000S','away':'DFL-CLU-00000B','competition':'000001'},
}
ALL_MATCHES = {**MATCHES, **FRESH_MATCHES}
# Team IDs above are transcribed from the public matchinformation XML (not inferred from labels).
BASE='https://huggingface.co/datasets/pysport/idsse-data/resolve/a715a38dfbaf5f58e431727c2b78d174101a703c/'

def source_files(match_id):
    c=ALL_MATCHES[match_id]['competition']
    return [f'DFL_{kind}_DFL-COM-{c}_DFL-MAT-{match_id}.xml' for kind in ('02_01_matchinformation','04_03_positions_raw_observed')]

def download_external(out, ids):
    """Reproducible user-runnable downloader. Leaves partial downloads with .part suffix."""
    import urllib.request
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for mid in ids:
        for name in source_files(mid):
            dest=out/name
            if dest.exists():continue
            temp=dest.with_suffix('.xml.part')
            print(f'Downloading {name}',flush=True)
            with urllib.request.urlopen(BASE+name,timeout=60) as src,temp.open('wb') as target:
                while chunk:=src.read(1024*1024):target.write(chunk)
            # Streaming syntax check avoids materialising a several-hundred-MB XML tree.
            for _,element in ET.iterparse(temp,events=('end',)):element.clear()
            temp.rename(dest)

def load_sportec(path, match_id, cache_dir):
    if match_id not in ALL_MATCHES:raise ValueError('Unsupported match; add independently verified team metadata first')
    path=Path(path);cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    sha=digest(path);cache=cache_dir/f'{match_id}-{sha[:12]}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as d:return {k:d[k] for k in d.files}
    parts={};current=None;series=[];pitch=None;seen_ids=set()
    for event,e in ET.iterparse(path,events=('start','end')):
        if event=='start' and e.tag=='MetaData' and e.get('MatchId'):
            if not e.get('MatchId').endswith(match_id):raise ValueError('Match identity mismatch')
        if event=='start' and e.tag=='FrameSet':
            current=(e.get('GameSection'),e.get('TeamId'),e.get('PersonId'));series=[]
        if event=='end' and e.tag=='PitchSize':pitch=(float(e.get('X')),float(e.get('Y')))
        if event=='end' and e.tag=='Frame' and current:
            period,team,person=current
            if team in (ALL_MATCHES[match_id]['home'],ALL_MATCHES[match_id]['away'],'BALL'):
                a=e.attrib
                series.append((int(a['N']),float(a.get('X','nan')),float(a.get('Y','nan')),
                               float(a.get('S','nan'))/3.6,int(a.get('BallStatus','-1')),int(a.get('BallPossession','0'))))
            e.clear()
        if event=='end' and e.tag=='FrameSet':
            if series:
                if current in seen_ids:raise ValueError('Duplicate object/period FrameSet')
                seen_ids.add(current);parts[current]=np.asarray(series,dtype=float)
            current=None;series=[];e.clear()
    if pitch is None:raise ValueError('Missing pitch dimensions')
    sequences=[]
    for period in ('firstHalf','secondHalf'):
        b=parts.get((period,'BALL',None))
        if b is None:
            candidates=[a for (p,t,_),a in parts.items() if p==period and t=='BALL']
            if len(candidates)!=1:raise ValueError(f'Missing or ambiguous ball data: {period}')
            b=candidates[0]
        ids=b[:,0].astype(int)
        if not np.array_equal(ids,np.arange(ids[0],ids[0]+len(ids))):raise ValueError('Nonconsecutive ball frame IDs')
        players=[(key,a) for key,a in parts.items() if key[0]==period and key[1]!='BALL']
        xy=np.full((len(ids),len(players),2),np.nan,dtype=np.float32)
        speed=np.full((len(ids),len(players)),np.nan,dtype=np.float32);teams=[]
        for slot,((_,team,_),a) in enumerate(players):
            ai=a[:,0].astype(int)
            if len(np.unique(ai))!=len(ai):raise ValueError('Duplicate player frames')
            ix=ai-ids[0];valid=(ix>=0)&(ix<len(ids));xy[ix[valid],slot]=a[valid,1:3];speed[ix[valid],slot]=a[valid,3]
            teams.append(0 if team==ALL_MATCHES[match_id]['home'] else 1)
        y=np.full(len(ids),3,dtype=np.int8);y[b[:,4]==0]=2
        for possession in (1,2):y[(b[:,4]==1)&(b[:,5]==possession)]=possession-1
        sequences.append({'xy':xy,'speed':speed,'teams':np.array(teams),'ball':b[:,1:3].astype(np.float32),
                          'ball_speed':b[:,3].astype(np.float32),'y':y,'frame_ids':ids,'period':period})
    # Store separate arrays per period: slots can differ after substitutions.
    data={'pitch':np.array(pitch),'source_sha256':np.array(sha),'match_id':np.array(match_id)}
    for i,s in enumerate(sequences):
        for k,v in s.items():data[f'{i}_{k}']=np.array(v)
    np.savez_compressed(cache,**data)
    return data

def period_arrays(data,i):
    return {k[2:]:v for k,v in data.items() if k.startswith(f'{i}_')}
