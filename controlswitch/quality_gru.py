"""036: observation-quality-gated GRU evidence; external-only frozen components."""
from pathlib import Path
import argparse,json
import numpy as np
from .spell_gru import reference_bundle
from .gated_rnn import load_checkpoint,symmetric
from .ranked_pipeline import infer
from .control_state import state_features,swap_state_columns,evaluate_accuracy
from .quality_gate import quality_features
from .boundary_had import boundary_observations,adaptive_decode
from .boundary_alignment import eligible
from .model import probabilities
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest
from . import tree_model

VARIANTS=(.25,.5)
def quality_mix(quality,cap):
    q=np.asarray(quality)
    if not np.isfinite(q).all() or ((q<0)|(q>1)).any() or cap not in VARIANTS:raise ValueError('Invalid quality or cap')
    return cap*(1-q)

def infer_set(bundle,gated,s,pitch,mask=None,caps=VARIANTS):
    base,p=infer(bundle,s,pitch,mask);g,_=symmetric(gated['model'],s,pitch,gated['mean'],gated['scale'],mask);g=g[:,:2]/np.maximum(g[:,:2].sum(1,keepdims=True),1e-12);x=state_features(s,pitch,mask);a=tree_model.predict(bundle['models']['control'],x);b=tree_model.predict(bundle['models']['control'],swap_state_columns(x));tree=(a+b[:,[1,0,2]])[:,:2]*.5;quality=tree_model.predict(bundle['models']['quality'],quality_features(s,mask))[:,1];out={}
    # Keep the original032 RNN+tree conditional live estimate as the gate anchor.
    live_base=p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12);d=p[:,2]
    for cap in caps:
        mix=quality_mix(quality,cap)[:,None];live=(1-mix)*live_base+mix*g;q=np.column_stack([live*(1-d[:,None]),d]);event=probabilities(bundle['models']['boundary'],boundary_observations(s,pitch,q,mask))[:,1];out[f'quality_gru{cap:g}']=adaptive_decode(q,event,1.)
    return base,out

def run(cache_root,roots,control_root,gru_root,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output directory')
    out.mkdir(parents=True)
    protocol={'version':'quality-gru-036','reference':'frozen032','history':'After035 rejection for noisy-tracking fold regression. Same two extensively reused external first-division matches; no independent test.',
     'hypothesis':'Use the already external-trained029 observation-quality model to attenuate GRU evidence under degraded tracking.',
     'frozen':'All032 weights and035 final12iteration GRUs unchanged; no refitting. Gated residual blend is between complete032 conditional live evidence (including original RNN) and035GRU, not035tree-only anchor.',
     'variants':'GRU weight=.25*(1-quality) or.5*(1-quality); quality gate never receives possession labels or augmentation flags at inference. No fitted threshold.',
     'selection':'Mean official four-condition spellF1 gain>=.003 over032; per clean/gap fold decline<=.01, noise fold no decline, deadF1 decline<=.005. Highest eligible mean, FP reported.',
     'audit':'Freeze one winner and component hashes before different-draw audit seeds1814/2014/2418; only winner/032 decoded. Audit mean must improve, otherwise retain032 without runner-up selection.',
     'competition_data_used':False,'competition_percentile_known':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong external match')
    names=[f'quality_gru{cap:g}' for cap in VARIANTS];folds=[];all_rows=[];all_base=[];all_preds={k:[] for k in names}
    for fi,mid in enumerate(('J03WN1','J03WMX')):
        print('FOLD',fi,mid,flush=True);fd=out/f'fold-{fi}';fd.mkdir();bundle,paths=reference_bundle(fi,roots,control_root);gated=load_checkpoint(Path(gru_root)/f'fold-{fi}/gru.npz');rows=[];base=[];preds={k:[] for k in names}
        for half in (0,1):
            seq=period_arrays(records[mid],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);a,b=infer_set(bundle,gated,s,records[mid]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});base.append(a)
                for k in names:preds[k].append(b[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=a,**b)
        f={'validation':mid,'component_hashes':{**{k:digest(p) for k,p in paths.items()},'gru':digest(Path(gru_root)/f'fold-{fi}/gru.npz')},'current':evaluate_accuracy(rows,base),'candidates':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base
        for k in names:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for cap,name in zip(VARIANTS,names):
        score=evaluate_accuracy(all_rows,all_preds[name],all_base);guards=eligible(folds,name,current,score);trials.append({'name':name,'cap':cap,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'component_hashes':[f['component_hashes'] for f in folds],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            bundle,_=reference_bundle(fi,roots,control_root);gated=load_checkpoint(Path(gru_root)/f'fold-{fi}/gru.npz')
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),1814+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),2014+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,2418+half)
                    a,b=infer_set(bundle,gated,s,records[mid]['pitch'],mask,caps=(selected['cap'],));rows.append({'y':seq['y'],'condition':c});base.append(a);preds.append(b[selected['name']])
        audit={'current':evaluate_accuracy(rows,base),'candidate':evaluate_accuracy(rows,preds,base),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','control-root','gru-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.control_root,a.gru_root,a.out)
if __name__=='__main__':main()
