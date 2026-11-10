"""External-only training and held-out evaluation with a recorded, fixed protocol."""
from pathlib import Path
import json
import hashlib
import platform
from datetime import datetime,timezone
import html
import numpy as np
from .external import MATCHES,load_sportec,period_arrays
from .model import (features,fit_classifier,probabilities,boundary_features,boundary_targets,
                    decode,nearest_codes,FEATURE_VERSION)
from .pipeline import pm,smooth,digest

PROTOCOL={'seed':42,'feature_version':FEATURE_VERSION,'fps':25,
          'penalty_grid':[3,6,12,24,48],'boundary_strength_grid':[.5,1,2],
          'baseline_half_windows':[0,25,50,125],
          'stress':'Random contiguous 0.8–4 second ball gaps; fixed 40% target; no player masking in this pilot.',
          'selection_metric':'mean spellF1 across clean and masked validation conditions',
          'competition_data_used':False,'competition_ground_truth_used':False}

def missing_mask(n,seed):
    rng=np.random.default_rng(seed);mask=np.zeros(n,bool)
    while mask.mean()<.4:
        a=int(rng.integers(0,n));mask[a:min(n,a+int(rng.integers(20,101)))]=True
    return mask

def crop_sequence(s,start,end):
    return {k:(v[start:end].copy() if k not in ('teams','period') else v.copy()) for k,v in s.items()}

def metric(y,pred):
    result=pm.evaluate(y,pred)
    return {k:result[k] for k in ('spell_f1_mean','macro_f1','alive_dead_f1','n_gt_spells','n_pred_spells','n_graded')}

def baseline_prediction(s,window,mask=None):
    return pm.label_codes(smooth(nearest_codes(s,mask),window))

def prepare(s,pitch,mask=None,include_interactions=False):return features(s,pitch,ball_missing=mask,include_interactions=include_interactions)

def train_models(train,pitch,include_interactions=False):
    xs=[];ys=[]
    for si,s in enumerate(train):
        for swapped in (False,True):
            for masked in (False,True):
                mask=missing_mask(len(s['y']),42+si) if masked else None
                x=features(s,pitch,swap=swapped,ball_missing=mask,include_interactions=include_interactions)
                y=s['y'].copy()
                if swapped:
                    h=y==0;a=y==1;y[h]=1;y[a]=0
                # 5 Hz training samples; every inference and scoring frame remains at 25 Hz.
                xs.append(x[::5]);ys.append(y[::5])
    x=np.concatenate(xs);y=np.concatenate(ys)
    classifier=fit_classifier(x,y)
    bx=[];by=[]
    for i,s in enumerate(train):
        for masked in (False,True):
            mask=missing_mask(len(s['y']),42+i) if masked else None
            x=prepare(s,pitch,mask,include_interactions);p=probabilities(classifier,x)
            labels=boundary_targets(s['y']);valid=s['y']<3
            indices=np.flatnonzero(valid)[::5]
            bx.append(boundary_features(x,p)[indices]);by.append(labels[indices])
    verifier=fit_classifier(np.concatenate(bx),np.concatenate(by),classes=2,epochs=35)
    return classifier,verifier

def extract_predictions(s,pitch,classifier,verifier,mask=None,include_interactions=False):
    x=prepare(s,pitch,mask,include_interactions);p=probabilities(classifier,x)
    b=probabilities(verifier,boundary_features(x,p))[:,1]
    return p,b

def split_data(records,single_match_pilot):
    if all(mid in records for mid in ('J03WOH','J03WPY','J03WMX')):
        train=[period_arrays(records['J03WOH'],i) for i in (0,1)]
        val=[period_arrays(records['J03WPY'],i) for i in (0,1)]
        test=[period_arrays(records['J03WMX'],i) for i in (0,1)]
        mode='three_match_pilot';split={'train':['J03WOH'],'validation':['J03WPY'],'test':['J03WMX']}
    elif single_match_pilot and len(records)==1:
        mid=next(iter(records));first=period_arrays(records[mid],0);cut=int(len(first['y'])*.7);purge=100
        train=[crop_sequence(first,0,cut-purge)];val=[crop_sequence(first,cut+purge,len(first['y']))]
        test=[period_arrays(records[mid],1)]
        mode='single_match_temporal_pilot';split={'match':mid,'train_first_half_frames':[0,cut-purge],
              'validation_first_half_frames':[cut+purge,len(first['y'])],'test':'entire second half',
              'purged_frames':purge*2,'warning':'Same teams and match in all partitions. This does not establish cross-match generalisation.'}
    else:raise ValueError('Need J03WOH train, J03WPY validation and J03WMX test. Use --single-match-pilot only for explicitly exploratory one-match evidence.')
    return train,val,test,mode,split

def aggregate(pairs):
    # An X separator ensures the scorer never joins possessions across periods/matches.
    y=np.concatenate([np.r_[a,np.int8(3)] for a,b in pairs]);p=np.concatenate([np.r_[b,np.int8(3)] for a,b in pairs])
    return metric(y,p)

def benchmark(external_root,out,single_match_pilot=False):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new, empty experiment folder')
    records={};sources={}
    for mid in MATCHES:
        found=list(Path(external_root).rglob(f'*positions_raw_observed*{mid}.xml'))
        if len(found)>1:raise ValueError(f'Duplicate files for {mid}')
        if found:
            print(f'Loading external match {mid}',flush=True)
            records[mid]=load_sportec(found[0],mid,Path('work/external/cache'))
            sources[mid]={'path':str(found[0].resolve()),'sha256':str(records[mid]['source_sha256'])}
    if not records:raise ValueError('No supported complete external match tracking XML found')
    train,val,test,mode,split=split_data(records,single_match_pilot)
    pitch=next(iter(records.values()))['pitch']
    if any(not np.array_equal(r['pitch'],pitch) for r in records.values()):raise ValueError('Mixed pitch dimensions require per-match normalisation support')
    out.mkdir(parents=True,exist_ok=True)
    protocol={**PROTOCOL,'mode':mode,'split':split,'sources':sources,'dataset_license':'CC-BY-4.0',
              'dataset_citation':'Bassek et al. (2025), doi:10.1038/s41597-025-04505-y',
              'timing_note':'All validation conditions and parameter grids fixed before test scoring; test scored once after model selection.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('Fitting state classifier and boundary verifier on training partition only',flush=True)
    classifier,verifier=train_models(train,pitch)
    vconditions=[]
    for i,s in enumerate(val):
        for condition in ('clean','masked'):
            mask=missing_mask(len(s['y']),100+i) if condition=='masked' else None
            p,b=extract_predictions(s,pitch,classifier,verifier,mask)
            vconditions.append((s,mask,p,b))
    print('Selecting settings on external validation partition only',flush=True)
    trials=[]
    for window in PROTOCOL['baseline_half_windows']:
        score=aggregate([(s['y'],baseline_prediction(s,window,mask)) for s,mask,_,_ in vconditions])['spell_f1_mean']
        trials.append({'model':'baseline','window':window,'validation_spellF1':score})
    best_base=max((t for t in trials if t['model']=='baseline'),key=lambda t:t['validation_spellF1'])
    for penalty in PROTOCOL['penalty_grid']:
        pred=[(s['y'],decode(p,penalty)) for s,_,p,_ in vconditions]
        trials.append({'model':'sequence','penalty':penalty,'validation_spellF1':aggregate(pred)['spell_f1_mean']})
        for strength in PROTOCOL['boundary_strength_grid']:
            pred=[(s['y'],decode(p,penalty,b,strength)) for s,_,p,b in vconditions]
            trials.append({'model':'control_switch','penalty':penalty,'strength':strength,'validation_spellF1':aggregate(pred)['spell_f1_mean']})
    chosen={name:max((t for t in trials if t['model']==name),key=lambda t:t['validation_spellF1']) for name in ('baseline','sequence','control_switch')}
    (out/'validation.json').write_text(json.dumps({'trials':trials,'selected':chosen},indent=2)+'\n')
    weights={f'state_{k}':v for k,v in classifier.items()};weights.update({f'boundary_{k}':v for k,v in verifier.items()})
    weights['settings']=np.array(json.dumps(chosen));weights['feature_version']=np.array(FEATURE_VERSION)
    np.savez_compressed(out/'model.npz',**weights)
    print('Settings frozen. Scoring held-out test partition now',flush=True)
    results=[]
    for condition in ('clean','masked'):
        collected={name:[] for name in ('baseline_selected','baseline_smooth50','direct_classifier','sequence','control_switch')}
        for i,s in enumerate(test):
            mask=missing_mask(len(s['y']),200+i) if condition=='masked' else None
            p,b=extract_predictions(s,pitch,classifier,verifier,mask)
            preds={'baseline_selected':baseline_prediction(s,best_base['window'],mask),
                   'baseline_smooth50':baseline_prediction(s,50,mask),'direct_classifier':np.argmax(p,axis=1),
                   'sequence':decode(p,chosen['sequence']['penalty']),
                   'control_switch':decode(p,chosen['control_switch']['penalty'],b,chosen['control_switch']['strength'])}
            for name,pred in preds.items():collected[name].append((s['y'],pred))
            # External-only intermediate outputs, retained for audit, excluded from source bundles.
            np.savez_compressed(out/f'test-{condition}-{i}.npz',frame_ids=s['frame_ids'],reference=s['y'],boundary_probability=b,**preds)
        for name,pairs in collected.items():results.append({'condition':condition,'model':name,**aggregate(pairs)})
    report={'status':'completed_external_pilot','mode':mode,'protocol':protocol,'selected':chosen,'results':results,
            'model_sha256':digest(out/'model.npz'),'versions':{'python':platform.python_version(),'numpy':np.__version__},
            'limitations':['One held-out half of one external match' if single_match_pilot else 'One held-out external match',
                          'No competition accuracy measured; no competition inputs used in fitting or selection.',
                          'No confidence interval or multi-match significance claim.',
                          'Only ball dropout tested; real tracker errors and tag integration are not covered.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_report(report,out/'report.html')
    return report

def write_report(report,out):
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k])) if k in ("condition","model") else format(r[k],".4f")}</td>' for k in ('condition','model','spell_f1_mean','macro_f1','alive_dead_f1'))+'</tr>' for r in report['results'])
    warnings=''.join(f'<li>{html.escape(t)}</li>' for t in report['limitations'])
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Control Switch · External experiment</title><style>body{{font:16px system-ui;background:#f5f5ee;color:#173c32;max-width:1000px;margin:50px auto;padding:20px}}h1{{font-size:42px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:14px;text-align:left;border-bottom:1px solid #ddd}}.note{{padding:20px;background:#e7ebdc}}pre{{white-space:pre-wrap;font-size:12px}}</style><p>CONTROL SWITCH / EXTERNAL EVALUATION</p><h1>Does boundary verification help?</h1><div class="note">{html.escape(report['mode'])}. Exploratory external evidence; no competition score.</div><table><tr><th>Condition</th><th>Model</th><th>spellF1</th><th>macroF1</th><th>Alive/dead F1</th></tr>{rows}</table><h2>Limits</h2><ul>{warnings}</ul><h2>Frozen settings</h2><pre>{html.escape(json.dumps(report['selected'],indent=2))}</pre><p>Data: Bassek et al. (2025), IDSSE/Sportec, CC BY 4.0. doi:10.1038/s41597-025-04505-y.</p></html>'''
    Path(out).write_text(content)
