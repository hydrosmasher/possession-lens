"""030 ranking-first external comparison and one frozen new-draw audit."""
from pathlib import Path
import argparse,json
import numpy as np
from .ranked_pipeline import SPECS,REFERENCE,make_bundle,infer,VERSION
from .control_state import evaluate_accuracy
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .external import period_arrays
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest

SINGLE_CACHE={'boundary_half':('event','balanced_half'),'boundary_full':('event','balanced_full'),'boundary_recovery':('event','recovery_half'),'state':('state','state'),'state_half':('state','half_blend'),'quality075':('quality','cap0.75'),'quality1':('quality','cap1.0')}

def asset_paths(fold,spec,roots):
    result={'team':Path(roots['team']),'control':Path(roots['control'])/f'fold-{fold}/control-estimator.npz' if spec['control']=='old' else Path(roots['state'])/f'fold-{fold}'/'control-state.npz'}
    # Path construction remains explicit for the two296-feature control models.
    if spec['control']!='old':result['control']=Path(roots['state'])/f'fold-{fold}'/('control-state.npz' if spec['control']=='state' else 'control-half-blend.npz')
    if spec['dead']=='blend':result['dead_blend']=Path(roots['dead'])/f'fold-{fold}/dead-blend-0.5.npz'
    else:
        result['dead_old']=Path(roots['old_forward'] if fold==0 else roots['old_reverse']);result['dead_noise']=Path(roots['noise'])/f'fold-{fold}/dead-estimator.npz';result['quality']=Path(roots['quality'])/f'fold-{fold}/quality-estimator.npz'
    if spec.get('boundary'):result['boundary']=Path(roots['event'])/f'fold-{fold}/boundary-head.npz'
    return result

def run(cache_root,roots,protocol_file,out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new output directory')
    out.mkdir(parents=True);protocol=json.loads(Path(protocol_file).read_text());(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    all_rows=[];all_current=[];all_preds={s['name']:[] for s in SPECS};folds=[];joint=[s for s in SPECS if s['name'].startswith('joint')]
    for fold,mid in enumerate(('J03WN1','J03WMX')):
        print('Selection fold',fold,mid,flush=True);files=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(files)!=1:raise ValueError('Ambiguous cache')
        with np.load(files[0],allow_pickle=False) as a:data={k:a[k] for k in a.files}
        if str(data['match_id'])!=mid:raise ValueError('Match mismatch')
        bundles={s['name']:make_bundle(s,asset_paths(fold,s,roots)) for s in joint};rows=[];current=[];preds={s['name']:[] for s in SPECS};fd=out/f'fold-{fold}';fd.mkdir()
        for half in (0,1):
            original=period_arrays(data,half)
            for ci,c in enumerate(CONDITIONS):
                i=half*4+ci;s,mask=condition(original,c,half,False);rows.append({'y':original['y'],'condition':c})
                with np.load(Path(roots['event'])/f'fold-{fold}/validation-{i}.npz',allow_pickle=False) as a:y=a['reference'];base=a['current']
                np.testing.assert_array_equal(y,original['y']);current.append(base)
                for name,(key,array_key) in SINGLE_CACHE.items():
                    with np.load(Path(roots[key])/f'fold-{fold}/validation-{i}.npz',allow_pickle=False) as a:
                        np.testing.assert_array_equal(a['reference'],y);preds[name].append(a[array_key])
                for spec in joint:
                    print('Joint',fold,half,c,spec['name'],flush=True);preds[spec['name']].append(infer(bundles[spec['name']],s,data['pitch'],mask)[0])
                np.savez_compressed(fd/f'validation-{i}.npz',reference=y,current=base,**{k:v[-1] for k,v in preds.items()})
        r={'validation':mid,'current':evaluate_accuracy(rows,current),'candidates':{k:evaluate_accuracy(rows,p,current) for k,p in preds.items()}};folds.append(r);(fd/'selection.json').write_text(json.dumps(r,indent=2)+'\n');all_rows+=rows;all_current+=current
        for k in all_preds:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_current);trials=[]
    for spec in SPECS:
        name=spec['name'];scores=evaluate_accuracy(all_rows,all_preds[name],all_current);guards={'spell_f1':scores['mean_spell_f1']>current['mean_spell_f1'],'prior_conditions':all(f['candidates'][name][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'tracker_noise':all(f['candidates'][name]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),'dead_f1':all(f['candidates'][name][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS)}
        trials.append({'spec':spec,'scores':scores,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']]
    if not eligible:raise ValueError('No ranking-first candidate eligible; retain019')
    chosen=max(eligible,key=lambda t:(round(t['scores']['mean_spell_f1'],6),np.mean([t['scores'][c]['macro_f1'] for c in CONDITIONS]),-t['scores']['totals']['false_positive']))
    freeze={'selected':chosen,'asset_hashes':[{role:digest(path) for role,path in asset_paths(fold,chosen['spec'],roots).items()} for fold in (0,1)],'selection_complete_before_new_draw_audit':True,'new_draw_seeds':protocol['new_draw_seeds']};(out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print('FROZEN',chosen['spec'],chosen['scores']['mean_spell_f1'],flush=True)
    # Only the frozen winner and predefined019 reference are evaluated on new draws.
    audit_rows=[];audit_current=[];audit_candidate=[]
    for fold,mid in enumerate(('J03WN1','J03WMX')):
        files=list(Path(cache_root).glob(mid+'-*.npz'))
        with np.load(files[0],allow_pickle=False) as a:data={k:a[k] for k in a.files}
        ref=make_bundle(REFERENCE,asset_paths(fold,REFERENCE,roots));candidate=make_bundle(chosen['spec'],asset_paths(fold,chosen['spec'],roots))
        for half in (0,1):
            original=period_arrays(data,half)
            for c in CONDITIONS:
                print('Frozen audit',fold,half,c,flush=True);s={k:v for k,v in original.items() if k!='y'};mask=None
                if c=='masked':mask=missing_mask(len(original['y']),1314+half)
                elif c=='long_gap':mask=long_mask(len(original['y']),1514+half)
                elif c=='tracker_noise':s=noisy_tracking(original,1918+half)
                audit_rows.append({'y':original['y'],'condition':c});audit_current.append(infer(ref,s,data['pitch'],mask)[0]);audit_candidate.append(infer(candidate,s,data['pitch'],mask)[0])
    audit={'current':evaluate_accuracy(audit_rows,audit_current),'candidate':evaluate_accuracy(audit_rows,audit_candidate,audit_current),'same_matches_new_noise_only':True,'retuning_performed':False};audit['mean_spell_f1_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':chosen,'freeze':freeze,'new_draw_audit':audit,'recommended_for_freeze':audit['mean_spell_f1_improves'],'fresh_match_test_results':None,'competition_percentile_known':False,'promoted_to_default':False,'limitations':['Ranking proxy uses two heavily reused provider-labelled matches; competitors/private scores unknown.','New-draw audit changes missingness only and is not independent-match testing.','False-positive counts may rise under the clarified ranking-first policy; full trade-off is reported.','Synthetic quality gate may not recognise actual supplied tracking failures.','No supplied-game labels, video, tracking or public score enter training/selection.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('REFERENCE',current['mean_spell_f1'],current['totals']);print('SELECTED',chosen['spec'],chosen['scores']['mean_spell_f1'],chosen['scores']['totals']);print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'])
    return report
