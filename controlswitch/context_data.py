"""Pinned MIT SkillCorner additions; verification never parses reserved test labels."""
from pathlib import Path
import json,tempfile,shutil
from .skillcorner_legacy import import_archive,REVISION
ASSETS={
 '2068':[('match_data.json',16914,'27760525d6ced01a11b0ef72502f3c7b09817640'),('structured_data.json',56097681,'ed545d22859747c7404aa11a9109f1f359a2b474')],
 '2440':[('match_data.json',14366,'4ccdba9d49b2d1be218d5bd08c9aa9552e3467b5'),('structured_data.json',53770663,'5e840ded4a4a5fe3a5586e976c033407e3e7f7b0')],
 '2841':[('match_data.json',14377,'f39e3776d6b682c097018f704294076b5b9d6841'),('structured_data.json',77100038,'917cb292078120b48d59d141220b65c77c073bbd')]}
SPLIT={'train':['4039','3749','3518','2068','2440'],'validation':['2269'],'test':['2841']}

def import_additions(archive,root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 if any((root/mid).exists() for mid in ASSETS):raise ValueError('Context additions already exist; refusing overwrite')
 catalog={'matches':[{'match_id':mid,'files':[{'name':n,'size':s,'sha':h} for n,s,h in files]} for mid,files in ASSETS.items()]}
 with tempfile.TemporaryDirectory(prefix='lens-context-import-') as tmp:
  verified=Path(tmp)/'verified';manifest=import_archive(archive,catalog,verified);manifest['split']=SPLIT;manifest['reserved_tracking_parsed']=False
  for mid in ASSETS:shutil.copytree(verified/mid,root/mid)
  if not (root/'LICENSE.SkillCorner').exists():shutil.copy2(verified/'LICENSE.SkillCorner',root/'LICENSE.SkillCorner')
 (root/'context-import-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
