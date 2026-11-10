"""Validation-only interaction and causal forecasting experiments.

Never scores the previously viewed pilot test partition. A fresh match is needed
to evaluate changes made after the original test results were inspected.
"""
from pathlib import Path
import json
import html
import numpy as np
from .external import load_sportec
from .experiment import split_data,missing_mask,train_models,extract_predictions,aggregate,baseline_prediction
from .model import causal_features,turnover_targets,fit_classifier,probabilities,decode,interaction_features
from .pipeline import digest

def binary_metrics(y,p,prior):
    valid=y>=0;y=y[valid];p=p[valid];positive=int(y.sum())
    order=np.argsort(-p,kind='stable');ranked_y=y[order];ranked_p=p[order]
    ends=np.r_[np.flatnonzero(ranked_p[:-1]!=ranked_p[1:]),len(p)-1]
    tp=np.cumsum(ranked_y)[ends];precision=tp/(ends+1);recall=tp/max(1,positive)
    ap=float(np.sum(np.diff(np.r_[0,recall])*precision))
    alarm=p>=.2;hits=int((alarm&(y==1)).sum())
    return {'eligible_frames':len(y),'positive_frames':positive,'prevalence':float(y.mean()),
            'average_precision':ap,'constant_score_average_precision':float(y.mean()),
            'brier_score':float(np.mean((p-y)**2)),'training_prior_brier_score':float(np.mean((prior-y)**2)),
            'threshold':.2,'precision_at_threshold':hits/max(1,int(alarm.sum())),
            'recall_at_threshold':hits/max(1,positive),
            'mean_predicted_probability':float(p.mean())}

def develop(external_root,pilot_dir,out):
    out=Path(out);pilot_dir=Path(pilot_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty development folder')
    old=json.loads((pilot_dir/'report.json').read_text());records={}
    for mid,source in old['protocol']['sources'].items():
        files=list(Path(external_root).rglob(f'*positions_raw_observed*{mid}.xml'))
        if len(files)!=1:raise ValueError(f'Need exactly one original external source for {mid}')
        records[mid]=load_sportec(files[0],mid,Path('work/external/cache'))
        if str(records[mid]['source_sha256'])!=source['sha256']:raise ValueError('External source changed since pilot')
    train,val,unused_test,mode,split=split_data(records,old['mode']=='single_match_temporal_pilot')
    del unused_test
    pitch=next(iter(records.values()))['pitch'];out.mkdir(parents=True,exist_ok=True)
    protocol={'status':'validation_only','original_test_report_sha256':digest(pilot_dir/'report.json'),
              'test_partition_scored':False,'competition_data_used':False,'split':split,
              'interaction_evidence':'Proximity, ball direction change, relative velocity, alignment and closing movement. No touch ground truth.',
              'interaction_candidate_threshold':.25,'interaction_candidate_separation_frames':12,
              'forecast_horizon_frames':25,'forecast_threshold':.2,'fps':25,
              'forecast_target':'Direct H↔A change within the next second, before any stoppage; current dead frames excluded.',
              'forecast_inputs':'Current and trailing observations only. No offline detector probabilities or future observations.',
              'provider_caveat':'Source tracking may have been smoothed offline; causal feature construction does not establish live sensor availability.',
              'settings':'Use the original validation-selected sequence penalties without re-tuning for the new features.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('Training interaction-enhanced detector on external training partition only',flush=True)
    classifier,verifier=train_models(train,pitch,include_interactions=True)
    print('Training independent causal one-second turnover forecaster',flush=True)
    xs=[];ys=[]
    for i,s in enumerate(train):
        y=turnover_targets(s['y'])
        for swap in (False,True):
            for masked in (False,True):
                mask=missing_mask(len(y),42+i) if masked else None
                xs.append(causal_features(s,pitch,swap,mask)[::5]);ys.append(y[::5])
    fx=np.concatenate(xs);fy=np.concatenate(ys);forecaster=fit_classifier(fx,fy,classes=2,epochs=45)
    prior=float(fy[fy>=0].mean());weights={}
    for prefix,m in [('state',classifier),('boundary',verifier),('forecast',forecaster)]:
        weights.update({prefix+'_'+k:v for k,v in m.items()})
    weights['feature_version']=np.array('interactions-context-v2');weights['forecast_horizon_frames']=np.array(25)
    np.savez_compressed(out/'development-models.npz',**weights)
    detection=[];forecast=[]
    for condition in ('clean','masked'):
        predictions={k:[] for k in ('baseline_smooth50','interaction_sequence','interaction_control_switch')};yforecast=[];pforecast=[]
        for i,s in enumerate(val):
            mask=missing_mask(len(s['y']),100+i) if condition=='masked' else None
            p,b=extract_predictions(s,pitch,classifier,verifier,mask,include_interactions=True)
            sequence_setting=old['selected']['sequence'];cs=old['selected']['control_switch']
            methods={'baseline_smooth50':baseline_prediction(s,50,mask),
                     'interaction_sequence':decode(p,sequence_setting['penalty']),
                     'interaction_control_switch':decode(p,cs['penalty'],b,cs['strength'])}
            for name,pred in methods.items():predictions[name].append((s['y'],pred))
            target=turnover_targets(s['y']);risk=probabilities(forecaster,causal_features(s,pitch,ball_missing=mask))[:,1]
            yforecast.append(target);pforecast.append(risk)
            evidence=interaction_features(s,ball_missing=mask)
            np.savez_compressed(out/f'validation-{condition}-{i}.npz',frame_ids=s['frame_ids'],risk_next_1s=risk,forecast_reference=target,interaction_evidence=evidence,**methods)
            # Heuristic candidate export, not a validated touch detector. A peak may be a duel or rebound.
            import csv
            with (out/f'interaction-candidates-{condition}-{i}.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['source_frame','team','candidate_score','interpretation'])
                for team,col in [('H',1),('A',6)]:
                    score=evidence[:,col];peaks=np.flatnonzero((score>.25)&(score>=np.r_[0,score[:-1]])&(score>np.r_[score[1:],0]));last=-12
                    for ix in peaks:
                        if ix-last>=12:writer.writerow([int(s['frame_ids'][ix]),team,float(score[ix]),'unverified_2d_interaction_candidate']);last=ix
        for name,pairs in predictions.items():detection.append({'condition':condition,'model':name,**aggregate(pairs)})
        forecast.append({'condition':condition,**binary_metrics(np.concatenate(yforecast),np.concatenate(pforecast),prior)})
    report={'status':'completed_validation_only','protocol':protocol,'detection':detection,'forecast':forecast,
            'limitations':['Validation data has been used for development; results are not unbiased test performance.',
                          'No confirmed ball-contact labels: only interaction evidence is implemented.',
                          'Forecast metrics are frame-level and do not establish useful event-level warnings.',
                          'Fresh external matches are required before promoting either addition.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k])) if k in ("condition","model") else format(r[k],".4f")}</td>' for k in ('condition','model','spell_f1_mean','macro_f1'))+'</tr>' for r in detection)
    frows=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k])) if k=="condition" else format(r[k],".4f")}</td>' for k in ('condition','average_precision','constant_score_average_precision','brier_score','training_prior_brier_score'))+'</tr>' for r in forecast)
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Control Switch · Interaction and forecasting development</title><style>body{{font:16px system-ui;max-width:1100px;margin:50px auto;padding:20px;background:#f5f5ee;color:#173c32}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}.notice{{background:#e7ebdc;padding:20px}}pre{{white-space:pre-wrap}}</style><h1>Interactions + one-second turnover risk</h1><p class="notice">VALIDATION ONLY. The previously viewed test partition was not scored again. No competition data was used.</p><h2>Possession detection</h2><table><tr><th>Condition</th><th>Method</th><th>spellF1 ↑</th><th>macroF1 ↑</th></tr>{rows}</table><h2>Causal turnover forecast</h2><p>Predict a direct live team-to-team change within the next second. No future observations or offline detector probabilities are inputs. AP should exceed its prevalence reference; lower Brier is better.</p><table><tr><th>Condition</th><th>Average precision ↑</th><th>Prevalence reference</th><th>Brier ↓</th><th>Training-prior Brier ↓</th></tr>{frows}</table><h2>What this does not prove</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in report['limitations'])}</ul><h2>Protocol</h2><pre>{html.escape(json.dumps(protocol,indent=2))}</pre><p>External data: IDSSE/Sportec, Bassek et al. (2025), CC BY 4.0. doi:10.1038/s41597-025-04505-y.</p></html>'''
    (out/'report.html').write_text(doc)
    return report
