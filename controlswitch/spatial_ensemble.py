"""050: frozen043 prediction averaging under physical field reflections."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from .decoder_calibration import external_bundle
from .recurrent_pipeline import infer
from .spatial_symmetry import reflect_sequence
from .boundary_had import boundary_observations,adaptive_decode
from .model import probabilities
from .boundary_alignment import eligible
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask

VIEWS={'original':(False,False),'x':(True,False),'y':(False,True),'xy':(True,True)}
FAMILIES={'mirror_x':('original','x'),'mirror_y':('original','y'),'four_views':tuple(VIEWS)}

def infer_views(bundle,s,pitch,mask=None,families=FAMILIES):
    needed=set(v for views in families.values() for v in views)|{'original'};ps={};qs={};current=None
    for name in VIEWS:
        if name not in needed:continue
        fx,fy=VIEWS[name];view=s if name=='original' else reflect_sequence(s,fx,fy);labels,p=infer(bundle,view,pitch,mask);ps[name]=p;qs[name]=probabilities(bundle['base']['models']['boundary'],boundary_observations(view,pitch,p,mask))[:,1]
        if name=='original':current=labels
    return current,{name:adaptive_decode(np.mean([ps[v] for v in views],axis=0),np.mean([qs[v] for v in views],axis=0),1.) for name,views in families.items()}

def run(cache_root,roots,run_root,out):
    out=Path(out);run_root=Path(run_root)
    if out.exists():raise ValueError('Choose new output folder')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);protocol={'version':'field-reflection-050','reference':'frozen043','hypothesis':'H/A/D labels are unchanged by coherent field reflections. Average frozen-model emissions/event evidence across spatial views to reduce coordinate-specific bias.',
    'families':FAMILIES,'parameters':'No fitting or weight changes. Reflections transform all ball/player xy consistently; speeds, team membership, timing and masks stay unchanged. No label is an input. Original missingness is retained.',
    'selection':'Mean official four-condition spellF1 gain>=.003 over043; per clean/gap fold decline<=.01,noise fold no decline,deadF1 decline<=.005. Highest eligible mean, report FP trade-off.',
    'audit':'Freeze one view family before new draws seeds3314/3514/3918. Only selected family and043 decoded; audit mean must improve or retain043, no runner-up selection.',
    'history':'Two heavily reused external first-division games; reflected observations are not new or independent matches. No challenge inputs/labels/video/public score used. No percentile claim.'};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
    folds=[];all_rows=[];all_base=[];all_preds={k:[] for k in FAMILIES}
    for fi,mid in enumerate(('J03WN1','J03WMX')):
        print('FOLD',fi,mid,flush=True);fd=out/f'fold-{fi}';fd.mkdir();bundle=external_bundle(fi,roots,run_root);rows=[];base=[];preds={k:[] for k in FAMILIES}
        for half in (0,1):
            seq=period_arrays(records[mid],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation views',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);a,b=infer_views(bundle,s,records[mid]['pitch'],mask)
                with np.load(run_root/f'motion-gru-043/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as old:np.testing.assert_array_equal(a,old['gru0.5'])
                rows.append({'y':seq['y'],'condition':c});base.append(a)
                for k in FAMILIES:preds[k].append(b[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=a,**b)
        f={'validation':mid,'current':evaluate_accuracy(rows,base),'candidates':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base
        for k in FAMILIES:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for name,views in FAMILIES.items():
        score=evaluate_accuracy(all_rows,all_preds[name],all_base);guards=eligible(folds,name,current,score);trials.append({'name':name,'views':views,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            bundle=external_bundle(fi,roots,run_root)
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit views',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),3314+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),3514+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,3918+half)
                    a,b=infer_views(bundle,s,records[mid]['pitch'],mask,{selected['name']:selected['views']});rows.append({'y':seq['y'],'condition':c});base.append(a);preds.append(b[selected['name']])
        audit={'current':evaluate_accuracy(rows,base),'candidate':evaluate_accuracy(rows,preds,base),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','run-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.run_root,a.out)
if __name__=='__main__':main()
