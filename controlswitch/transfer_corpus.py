"""Disk-backed508-feature corpus from existing licensed top-five-league data."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from .joint_state import joint_features
from .spell_training import spell_weights
from .narrow_boundary import narrow_targets
from .skillcorner_europe import load_match
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition

WEAK_IDS=('4039','3749','3518','2068','2440')
STRONG_IDS=('J03WMX','J03WN1')

def build(cache_root,raw,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new corpus directory')
    if shutil.disk_usage(out.parent).free<20*2**30:raise ValueError('At least20GB free needed for this optional disk-backed corpus')
    out.mkdir(parents=True);protocol={'version':'joint508-transfer-corpus-v1','weak_matches':WEAK_IDS,'strong_matches':STRONG_IDS,'sources':'Historical MIT SkillCorner first-division data and CC BY4.0 Sportec/IDSSE first-division data; existing local files only','features':'508joint features, rawfloat32 memory-mapped arrays; no team/spatial swap duplication. Labels stored separately.','weak_supervision':'Known H/A only, unknown stays-1; known-frame weight.5; no weak dead labels or boundary targets.','strong_supervision':'Known live H/A only with032spell weights; narrow+-2event target and fully known-live+-12valid neighbourhood. Dead/unknown excluded from neural control loss.','conditions':list(CONDITIONS),'seeds':'Existing fixed training-condition seeds; no validation conditions/features included.','competition_data_used':False,'excluded_weak_validation_or_test_ids':['2269','2841','3442','2417']};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');entries=[];audits=[]
    def append_parts(mid,source,parts,audit):
        audits.append({'match_id':mid,'source':source,'audit':audit})
        for half,(seq,pitch) in enumerate(parts):
            y=np.where(seq['y']<2,seq['y'],-1).astype(np.int8)
            if source=='strong':w=spell_weights(seq['y']).astype(np.float32);t,v=narrow_targets(seq['y'],2);v=v.astype(np.float32)
            else:w=np.where(y>=0,.5,0).astype(np.float32);t=np.zeros(len(y),np.float32);v=np.zeros(len(y),np.float32)
            label_name=f'{mid}-h{half}-labels.npz';np.savez_compressed(out/label_name,y=y,w=w,t=t.astype(np.float32),v=v)
            for c in CONDITIONS:
                print('Corpus',mid,half,c,flush=True);s,mask=condition(seq,c,half,True);x=joint_features(s,pitch,mask);name=f'{mid}-h{half}-{c}-features.npy';tmp=out/(name+'.partial')
                with tmp.open('wb') as f:np.save(f,x,allow_pickle=False)
                tmp.rename(out/name);entries.append({'match_id':mid,'source':source,'half':half+1,'condition':c,'features':name,'labels':label_name,'rows':len(y),'columns':508,'known_frames':int((y>=0).sum())});del x
                (out/'progress.json').write_text(json.dumps({'entries_complete':len(entries),'entries':entries},indent=2)+'\n')
    for mid in STRONG_IDS:
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous strong source')
        with np.load(fs[0],allow_pickle=False) as a:d={k:a[k] for k in a.files}
        if str(d['match_id'])!=mid:raise ValueError('Wrong strong match')
        parts=[(period_arrays(d,h),d['pitch']) for h in (0,1)];append_parts(mid,'strong',parts,{'frames':sum(len(s['y']) for s,p in parts)});del d,parts
    for mid in WEAK_IDS:
        print('Load weak match',mid,flush=True);parts,audit=load_match(raw,mid);append_parts(mid,'weak',parts,audit);del parts
    manifest={'protocol':protocol,'entries':entries,'source_audit':audits,'complete':True,'feature_bytes':sum((out/e['features']).stat().st_size for e in entries),'all_matches_previously_viewed':True,'independent_test':False};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('CORPUS COMPLETE',len(entries),manifest['feature_bytes'],flush=True);return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','raw','out'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();build(a.cache_root,a.raw,a.out)
if __name__=='__main__':main()
