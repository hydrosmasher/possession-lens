"""Import existing licensed external XMLs. Never downloads or moves originals."""
from pathlib import Path
import json,shutil
import xml.etree.ElementTree as ET
from .external import ALL_MATCHES,source_files
from .pipeline import digest

def verify_xml(path,match):
    identity=False;frames=False
    for _,e in ET.iterparse(path,events=('end',)):
        mid=e.get('MatchId')
        if mid:
            if mid!='DFL-MAT-'+match:raise ValueError(f'External XML identity differs from its filename: {match}')
            identity=True
        if e.tag=='Frame':frames=True
        e.clear()
    if not identity:raise ValueError(f'No verifiable external match identity in {Path(path).name}')
    if 'positions_raw_observed' in Path(path).name and not frames:raise ValueError('Tracking XML contains no frame records')

def import_local(source,destination,matches=None):
    source=Path(source).expanduser().resolve();destination=Path(destination).expanduser().resolve()
    if not source.is_dir():raise ValueError('The local source folder does not exist')
    matches=list(ALL_MATCHES) if matches is None else list(matches)
    if any(m not in ALL_MATCHES for m in matches):raise ValueError('Unsupported external match ID')
    destination.mkdir(parents=True,exist_ok=True);assets=[];availability={}
    for match in matches:
        present=[]
        for name in source_files(match):
            target=destination/name
            found=sorted(set(source.rglob(name)))
            if len(found)>1:
                hashes={digest(p) for p in found}
                if len(hashes)>1:raise ValueError(f'Conflicting duplicate external XML: {name}')
            if target.exists():
                verify_xml(target,match)
                if found and digest(target)!=digest(found[0]):raise ValueError(f'Refusing to overwrite a different existing input: {name}')
                status='already_imported';path=target
            elif found:
                path=found[0];verify_xml(path,match);part=target.with_suffix('.xml.copying')
                shutil.copyfile(path,part)
                if digest(path)!=digest(part):part.unlink();raise ValueError('Local copy integrity check failed')
                part.rename(target);status='copied_from_local_folder'
            else:continue
            present.append(name)
            assets.append({'match_id':match,'name':name,'status':status,'sha256':digest(target),'bytes':target.stat().st_size})
        cache=list((destination.parent/'cache').glob(f'{match}-*.npz'))
        availability[match]={'tracking_xml':any('positions_raw_observed' in x for x in present),
                             'identity_metadata':any('matchinformation' in x for x in present),
                             'parsed_cache_available':bool(cache)}
    report={'mode':'local_import_no_network','originals_unchanged':True,'partial_downloads_ignored':True,
            'assets':assets,'availability':availability,
            'note':'Missing raw inputs are reported, never fetched. A parsed cache can support existing team-model experiments.'}
    (destination/'local-import-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
