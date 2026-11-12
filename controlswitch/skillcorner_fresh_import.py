"""Verified import of the reserved sixth match; does not parse tracking or labels."""
from pathlib import Path
import tempfile,json,shutil
from .skillcorner_legacy import import_archive

FRESH_CATALOG={'matches':[{'match_id':'2417','files':[
 {'name':'match_data.json','size':15172,'sha':'9ec7c9ea82f9fc053a1afa24c2316a39370f9fa4'},
 {'name':'structured_data.json','size':54257176,'sha':'81a3f335841cc09b6814a180425e8c534ca8dc98'}]}]}

def import_fresh_test(archive,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'2417').exists():raise ValueError('Reserved match already imported; refusing overwrite')
    with tempfile.TemporaryDirectory(prefix='possession-lens-import-') as tmp:
        manifest=import_archive(archive,FRESH_CATALOG,Path(tmp)/'verified')
        manifest['split']={'train':['4039','3749','3518'],'validation':['2269'],'test':['2417']}
        manifest['held_out_contents_parsed']=False
        shutil.copytree(Path(tmp)/'verified/2417',root/'2417')
        (root/'2417/sealed-import-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if not (root/'LICENSE.SkillCorner').exists():shutil.copy2(Path(tmp)/'verified/LICENSE.SkillCorner',root/'LICENSE.SkillCorner')
    return manifest
