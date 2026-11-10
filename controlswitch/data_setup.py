"""Build a private local dataset layout without modifying the original downloads."""
from pathlib import Path
import json
from .pipeline import discover

def organise(source,destination):
    source=Path(source).expanduser().resolve();destination=Path(destination).resolve()
    files=discover(source);assets={}
    for recording in ('tactical','broadcast'):
        for suffix in ('_atd.xml','_atd.txt','_homography.csv','_smart_tagging.xml'):
            name=f'BAR-ATH_{recording}{suffix}';matches=list(source.rglob(name))
            if len(matches)>1:raise ValueError(f'Duplicate source asset: {name}')
            if matches:assets[f'{recording}/{name}']=matches[0]
        video=list(source.rglob(f'BAR-ATH_{recording}*.mp4'))
        if len(video)>1:raise ValueError(f'Ambiguous {recording} video')
        if video:assets[f'{recording}/BAR-ATH_{recording}.mp4']=video[0]
    destination.mkdir(parents=True,exist_ok=True)
    for relative,path in assets.items():
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_symlink() and target.resolve()==path.resolve():continue
        if target.exists() or target.is_symlink():raise ValueError(f'Refusing to overwrite {target}')
        target.symlink_to(path.resolve())
    manifest={'mode':'local_symlinks','originals_unchanged':True,'assets':{k:{'source':str(v),'bytes':v.stat().st_size} for k,v in assets.items()}}
    (destination/'local-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
