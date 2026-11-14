"""045: predeclared fusion of frozen043 GRU and external tree variants."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from .narrow_boundary import bundle040
from .ranked_pipeline import infer as infer_base
from .gated_rnn import load_checkpoint,symmetric
from .joint_state import joint_features,swap_joint
from .latent_gap import latent_features,swap_latent
from .europe_context import observation_probabilities
from .boundary_had import boundary_observations,adaptive_decode
from .model import probabilities
from .boundary_alignment import eligible
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest
from . import tree_model

def decode_live(base,p,live,s,pitch,mask):
    if live.ndim!=2 or live.shape[1]!=2 or not np.isfinite(live).all() or (live<0).any() or not np.allclose(live.sum(1),1,atol=1e-6):raise ValueError('Expected conditional live probabilities')
    d=p[:,2];q=np.column_stack([live*(1-d[:,None]),d]);event=probabilities(base['models']['boundary'],boundary_observations(s,pitch,q,mask))[:,1];return adaptive_decode(q,event,1.)

def inference_family(base,gru,models,families,s,pitch,mask=None):
    _,p=infer_base(base,s,pitch,mask);g,_=symmetric(gru['model'],s,pitch,gru['mean'],gru['scale'],mask);g=g[:,:2]/np.maximum(g[:,:2].sum(1,keepdims=True),1e-12);h=p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12);current=decode_live(base,p,.5*h+.5*g,s,pitch,mask);legacy=observation_probabilities(base['models']['team']['model'],s,pitch,mask,context=True)[:,:2];legacy=legacy/np.maximum(legacy.sum(1,keepdims=True),1e-12);cache={};out={}
    for f in families:
        kind=f['feature']
        if kind not in cache:
            x=joint_features(s,pitch,mask) if kind=='joint' else latent_features(s,pitch,mask);cache[kind]=(x,swap_joint(x) if kind=='joint' else swap_latent(x))
        x,sx=cache[kind];a=tree_model.predict(models[f['name']],x);b=tree_model.predict(models[f['name']],sx);tree=(a+b[:,[1,0,2]])[:,:2]*.5;mix=f['old_tree_mix'];live=.5*((1-mix)*legacy+mix*tree)+.5*g;out[f['name']]=decode_live(base,p,live,s,pitch,mask)
    return current,out

def run(cache_root,roots,run_root,protocol_file,out):
    out=Path(out);run_root=Path(run_root)
    if out.exists():raise ValueError('Choose a new output folder')
    r=json.loads((run_root/'motion-gru-043/report.json').read_text())
    if not r['recommended_for_freeze'] or r['selected']['gru_mix']!=.5:raise ValueError('Expected frozen043 winner and passed audit')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out.mkdir(parents=True);protocol=json.loads(Path(protocol_file).read_text());(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');families=protocol['candidate_families'];records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
    def bundles(fi):
        base,paths=bundle040(fi,roots,run_root/'joint-040');gp=run_root/f'motion-gru-043/fold-{fi}/gru.npz';gru=load_checkpoint(gp);models={};hashes={**{k:digest(p) for k,p in paths.items()},'gru':digest(gp)}
        for f in families:
            directory,name=f['source'].split('/');p=run_root/directory/f'fold-{fi}'/(name+'.npz');models[f['name']]=tree_model.load(p);hashes[f['name']]=digest(p)
        return base,gru,models,hashes
    folds=[];all_rows=[];all_base=[];all_preds={f['name']:[] for f in families}
    for fi,mid in enumerate(('J03WN1','J03WMX')):
        print('FOLD',fi,mid,flush=True);fd=out/f'fold-{fi}';fd.mkdir();base,gru,models,hashes=bundles(fi);rows=[];current=[];preds={f['name']:[] for f in families}
        for half in (0,1):
            seq=period_arrays(records[mid],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);a,b=inference_family(base,gru,models,families,s,records[mid]['pitch'],mask)
                with np.load(run_root/f'motion-gru-043/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as old:np.testing.assert_array_equal(a,old['gru0.5'])
                rows.append({'y':seq['y'],'condition':c});current.append(a)
                for k in preds:preds[k].append(b[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=a,**b)
        f={'validation':mid,'component_hashes':hashes,'current':evaluate_accuracy(rows,current),'candidates':{k:evaluate_accuracy(rows,p,current) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=current
        for k in preds:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for f in families:
        score=evaluate_accuracy(all_rows,all_preds[f['name']],all_base);guards=eligible(folds,f['name'],current,score);trials.append({**f,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'component_hashes':[f['component_hashes'] for f in folds],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];current=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            base,gru,models,_=bundles(fi)
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),2814+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),3014+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,3418+half)
                    a,b=inference_family(base,gru,models,[selected],s,records[mid]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});current.append(a);preds.append(b[selected['name']])
        audit={'current':evaluate_accuracy(rows,current),'candidate':evaluate_accuracy(rows,preds,current),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','run-root','protocol','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.run_root,a.protocol,a.out)
if __name__=='__main__':main()
