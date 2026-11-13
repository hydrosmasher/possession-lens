"""Exploratory emission-protected boundary verification after crossfit008."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import torch
from .crossfit_precision import dump
from .gated_rnn import load_checkpoint,symmetric
from .model import probabilities,decode
from .skillcorner_europe import load_match,digest
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import invariant_boundary_features,spell_diagnostics
from .europe_recovery import totals
from .boundary_precision import blend_boundary,retention_counts

def protected_boundary(p,head,q,mix,threshold):
    p=np.asarray(p)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)!=len(head) or not np.isfinite(p).all() or not np.isfinite(threshold) or not 0<threshold<=1:raise ValueError('Invalid emission protection')
    index=np.arange(len(p));contrast=np.max(np.abs(p[np.minimum(index+25,len(p)-1),:2]-p[np.maximum(index-25,0),:2]),axis=1)
    blended=blend_boundary(head,q,mix)
    return np.where(contrast>=threshold,head,blended).astype(np.float32)

def run(raw,reference,backbone,verifier,out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new output folder')
    out.mkdir(parents=True);torch.set_num_threads(1)
    protocol={'experiment':'selective-precision-009','status':'Exploratory post008 follow-up; specified after viewing008; not fresh-test evidence','training_performed':False,'validation':['2269'],'test_loaded':False,'competition_data_used':False,'backbone_sha256':digest(backbone),'verifier_sha256':digest(verifier),'policy':'Preserve original boundary head when either team probability differs across +/-25 frames by >= threshold; otherwise use verifier blend','mixes':[.25,.5],'thresholds':[.25,.5],'penalty':28,'selection':'same008 safeguards; spell F1>high-recall, misses<=292,FP<=112,merges<=66,actual short0, lost same recovered spells0'};dump(out/'protocol.json',protocol)
    f=load_checkpoint(backbone);ref=load_checkpoint(reference)
    with np.load(verifier,allow_pickle=False) as a:v={k:a[k].copy() for k in ('weights','mean','scale')}
    parts,_=load_match(raw,'2269');rows=conditions(parts,100);values=[];current=[];recovery=[]
    for s,p,c,mask in rows:
        prob,head=symmetric(f['model'],s,p,f['mean'],f['scale'],mask);q=probabilities(v,invariant_boundary_features(s,p,prob,mask))[:,1];values.append((prob,head,q));recovery.append(decode(prob,28,head,1.));pp,bb=symmetric(ref['model'],s,p,ref['mean'],ref['scale'],mask);current.append(decode(pp,ref['penalty'],bb,ref['boundary_strength']))
    def score(pred):return {**evaluate(rows,pred),'diagnostics':spell_diagnostics(rows,pred)}
    rscore=score(recovery);assert abs(rscore['selection_spell_f1']-.5791444378581307)<1e-12
    trials=[]
    for mix in protocol['mixes']:
        for threshold in protocol['thresholds']:
            pred=[decode(p,28,protected_boundary(p,b,q,mix,threshold),1.) for p,b,q in values];r=score(pred);t=totals(r['diagnostics']);counts=[retention_counts(s['y'],a,b,c) for (s,_,_,_),a,b,c in zip(rows,current,recovery,pred)];ret={k:sum(v[k] for v in counts) for k in ['recovered_spells','lost_recovered_spells']};guard={'misses':t['missed_reference_spells_at_iou_0_5']<=292,'false_positives':t['false_positive_spells_at_iou_0_5']<=112,'merges':t['predictions_overlapping_multiple_reference_spells']<=66,'short_predictions':sum(r['diagnostics'][c]['full_prediction_spells_under_0_4_seconds'] for c in ['clean','masked'])==0,'retention':ret['lost_recovered_spells']==0,'spell_f1':r['selection_spell_f1']>rscore['selection_spell_f1']};trials.append({'mix':mix,'threshold':threshold,**r,'retention_audit':ret,'guards':guard,'eligible':all(guard.values())})
    eligible=[r for r in trials if r['eligible']];best=max(eligible,key=lambda r:r['selection_spell_f1']) if eligible else None;verification=None
    if best:
        bundle=out/'candidate';bundle.mkdir();shutil.copy2(backbone,bundle/'backbone.npz');shutil.copy2(verifier,bundle/'verifier.npz');dump(bundle/'decoder.json',{'penalty':28,'mix':best['mix'],'threshold':best['threshold'],'policy':'emission-protected-v1','validation_only':True});restored=load_checkpoint(bundle/'backbone.npz')
        with np.load(bundle/'verifier.npz',allow_pickle=False) as a:vv={k:a[k].copy() for k in ('weights','mean','scale')}
        config=json.loads((bundle/'decoder.json').read_text());pred=[]
        for s,p,c,mask in rows:
            pp,bb=symmetric(restored['model'],s,p,restored['mean'],restored['scale'],mask);qq=probabilities(vv,invariant_boundary_features(s,p,pp,mask))[:,1];pred.append(decode(pp,config['penalty'],protected_boundary(pp,bb,qq,config['mix'],config['threshold']),1.))
        assert score(pred)=={k:best[k] for k in ['clean','masked','selection_spell_f1','diagnostics']};verification={'reloaded_predictions_match':True}
    report={'protocol':protocol,'reference':score(current),'recovery_reference':rscore,'selected':best,'best_unrestricted':max(trials,key=lambda r:r['selection_spell_f1']),'trials':trials,'verification':verification,'fresh_test_results':None,'promoted_to_default':False};dump(out/'report.json',report);dump(out/'trials.json',trials);print('COMPLETE',report['best_unrestricted']['selection_spell_f1'],'selected',best is not None,flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('raw','reference','backbone','verifier','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.raw,a.reference,a.backbone,a.verifier,a.out)
