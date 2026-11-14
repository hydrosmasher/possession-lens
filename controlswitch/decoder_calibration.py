"""048: two predeclared H/A penalties on frozen043 neural emissions."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from .recurrent_pipeline import infer,load_pipeline
from .narrow_boundary import bundle040
from .gated_rnn import load_checkpoint
from .boundary_had import boundary_observations,adaptive_decode
from .model import probabilities
from .boundary_alignment import eligible
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest

PENALTIES=(6.,9.)
def external_bundle(fi,roots,run_root):
    if fi==0:return load_pipeline('reports/external/motion-gru-043-config.json')
    base,_=bundle040(fi,roots,Path(run_root)/'joint-040');return {'base':base,'gru':load_checkpoint(Path(run_root)/f'motion-gru-043/fold-{fi}/gru.npz'),'mix':.5}

def run(cache_root,roots,run_root,out):
    out=Path(out);run_root=Path(run_root)
    if out.exists():raise ValueError('Choose new output folder')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);protocol={'version':'decoder-calibration-048','reference':'frozen043','hypothesis':'Penalty12 was selected on earlier emissions. Test6/9 on the richer fixed neural ensemble to recover short possessions; stop/restart3 and full-strength031event head stay fixed.',
    'penalties':PENALTIES,'fitting':'None; no feature,model or emission-weight changes. External-only parameter comparison.',
    'selection':'Mean four-condition official spellF1 gain>=.003 over043; per clean/gap fold decline<=.01, noise fold no decline, deadF1 decline<=.005. Highest eligible mean; report FP trade-offs.',
    'audit':'Freeze one winner before new observation draws seeds3114/3314/3718; only winner/043 decoded. Audit mean must improve or retain043, no runner-up selection.',
    'history':'Two heavily reused external first-division Bundesliga matches. Not independent testing; no challenge inputs/video/labels/public score used. No competition-percentile claim.'};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
    names=[f'penalty{x:g}' for x in PENALTIES];folds=[];all_rows=[];all_base=[];all_preds={k:[] for k in names}
    for fi,mid in enumerate(('J03WN1','J03WMX')):
        print('FOLD',fi,mid,flush=True);fd=out/f'fold-{fi}';fd.mkdir();b=external_bundle(fi,roots,run_root);rows=[];current=[];preds={k:[] for k in names}
        for half in (0,1):
            seq=period_arrays(records[mid],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);a,p=infer(b,s,records[mid]['pitch'],mask);q=probabilities(b['base']['models']['boundary'],boundary_observations(s,records[mid]['pitch'],p,mask))[:,1]
                with np.load(run_root/f'motion-gru-043/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as old:np.testing.assert_array_equal(a,old['gru0.5'])
                rows.append({'y':seq['y'],'condition':c});current.append(a)
                for name,penalty in zip(names,PENALTIES):preds[name].append(adaptive_decode(p,q,1.,penalty=penalty))
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=a,**{k:v[-1] for k,v in preds.items()})
        f={'validation':mid,'current':evaluate_accuracy(rows,current),'candidates':{k:evaluate_accuracy(rows,p,current) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=current
        for k in names:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for name,penalty in zip(names,PENALTIES):
        score=evaluate_accuracy(all_rows,all_preds[name],all_base);guards=eligible(folds,name,current,score);trials.append({'name':name,'penalty':penalty,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'reference_config_sha256':digest('reports/external/motion-gru-043-config.json'),'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];current=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            b=external_bundle(fi,roots,run_root)
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),3114+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),3314+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,3718+half)
                    a,p=infer(b,s,records[mid]['pitch'],mask);q=probabilities(b['base']['models']['boundary'],boundary_observations(s,records[mid]['pitch'],p,mask))[:,1];rows.append({'y':seq['y'],'condition':c});current.append(a);preds.append(adaptive_decode(p,q,1.,penalty=selected['penalty']))
        audit={'current':evaluate_accuracy(rows,current),'candidate':evaluate_accuracy(rows,preds,current),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','run-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.run_root,a.out)
if __name__=='__main__':main()
