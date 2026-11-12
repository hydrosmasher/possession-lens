"""Pinned external European SkillCorner import and label audit; never challenge data."""
from pathlib import Path
from collections import Counter
import hashlib,json,zipfile,gc
REVISION='1ed50b775f93075c2a6d00d0d795f13b7b89709e'
SPLIT={'train':['4039','3749','3518'],'validation':['2269'],'test':['3442']}

def import_archive(archive,catalog,destination):
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);records=[]
 with zipfile.ZipFile(archive) as z:
  for match in catalog['matches']:
   for asset in match['files']:
    suffix=f"/data/matches/{match['match_id']}/{asset['name']}";members=[i for i in z.infolist() if i.filename.endswith(suffix)]
    if len(members)!=1 or members[0].file_size!=asset['size']:raise ValueError('Archive member missing or incorrect size')
    target=destination/match['match_id']/asset['name'];target.parent.mkdir(exist_ok=True)
    if target.exists():raise ValueError('Refusing to overwrite an existing external import')
    git=hashlib.sha1(f'blob {asset["size"]}\0'.encode());sha=hashlib.sha256()
    with z.open(members[0]) as src,target.open('wb') as dst:
     while chunk:=src.read(1024*1024):git.update(chunk);sha.update(chunk);dst.write(chunk)
    if git.hexdigest()!=asset['sha']:raise ValueError('Source Git blob integrity mismatch')
    records.append({'match_id':match['match_id'],'file':asset['name'],'bytes':asset['size'],'git_blob':git.hexdigest(),'sha256':sha.hexdigest()})
  licence=z.read(next(n for n in z.namelist() if n.endswith('/LICENSE')))
 (destination/'LICENSE.SkillCorner').write_bytes(licence)
 result={'source_revision':REVISION,'files':records,'held_out_contents_parsed':False,'split':SPLIT}
 (destination/'import-manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result

def audit(path):
 rows=json.loads(Path(path).read_text());counts=Counter();keys=Counter();period=Counter();objects=Counter();ids=set();anon=set();empty=0;frameids=[];first=None
 for frame in rows:
  keys.update(frame.keys());period[str(frame.get('period'))]+=1;pos=frame.get('possession')
  value=json.dumps(pos,sort_keys=True);counts[value]+=1
  ds=frame.get('data',[]);empty+=not bool(ds);frameids.append(frame.get('frame'))
  for o in ds:
   objects[str(o.get('group_name'))]+=1
   if o.get('trackable_object') is not None:ids.add(str(o['trackable_object']))
   elif o.get('track_id') is not None:anon.add((str(o.get('group_name')),str(o['track_id'])))
  if first is None:first={'frame_keys':list(frame),'possession_example':pos,'object_schema':[{k:type(v).__name__ for k,v in o.items()} for o in ds[:2]]}
 report={'frames':len(rows),'period_counts':dict(period),'frame_field_counts':dict(keys),'possession_value_counts':dict(counts),'empty_tracking_frames':empty,'object_groups':dict(objects),'identified_entities':len(ids),'anonymous_track_ids':len(anon),'first_frame_schema':first,'frame_ids_strictly_increasing':all(a<b for a,b in zip(frameids,frameids[1:]))}
 del rows;gc.collect();return report
