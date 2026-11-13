"""External-only switch verification; fixed possession emissions, learned costs."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import torch
from .gated_recovery import run as reproduce_recovery,dump
from .gated_rnn import load_checkpoint,symmetric
from .interaction_context import TRAIN
from .skillcorner_europe import load_match,digest
from .experiment import missing_mask
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import invariant_boundary_features,spell_diagnostics
from .europe_recovery import boundary_supervision,totals
from .model import decode,fit_classifier,probabilities

def blend_boundary(head,verifier,mix):
    head=np.asarray(head);verifier=np.asarray(verifier)
    if head.shape!=verifier.shape or head.ndim!=1 or not np.isfinite(head).all() or not np.isfinite(verifier).all() or ((head<0)|(head>1)).any() or ((verifier<0)|(verifier>1)).any() or not np.isfinite(mix) or not 0<=mix<=1:raise ValueError('Invalid boundary blend')
    return ((1-mix)*head+mix*verifier).astype(np.float32)

def fitting_indices(valid,head):
    """Natural-frequency sample plus extra model-suspected boundaries; labels never select negatives."""
    valid=np.asarray(valid,dtype=bool);head=np.asarray(head)
    if valid.shape!=head.shape:raise ValueError('Mismatched supervision')
    natural=np.flatnonzero(valid)[::5]
    hard=np.flatnonzero(valid&(head>=.5))[::2]
    return np.r_[natural,hard]

def infer_bundle(bundle,s,p,mask=None):
    bundle=Path(bundle);f=load_checkpoint(bundle/'backbone.npz');config=json.loads((bundle/'decoder.json').read_text())
    with np.load(bundle/'verifier.npz',allow_pickle=False) as a:v={k:a[k].copy() for k in ('weights','mean','scale')}
    prob,head=symmetric(f['model'],s,p,f['mean'],f['scale'],mask);q=probabilities(v,invariant_boundary_features(s,p,prob,mask))[:,1]
    return decode(prob,config['penalty'],blend_boundary(head,q,config['mix']),1.)

def retention_counts(y,current,recovery,candidate):
    from .pipeline import pm
    runs=pm.valid_runs(y);gt=pm.possession_spells(y,runs)
    def flags(pred):
        return pm.matched_gt_flags(pm.spell_overlaps(gt,pm.possession_spells(pred,runs)),len(gt),.5)
    recovered=flags(recovery)&~flags(current)
    return {'recovered_spells':int(recovered.sum()),'lost_recovered_spells':int((recovered&~flags(candidate)).sum())}

def audit_retention(raw,reference,out):
    """Add same-spell retention audit; no fitting and no new decoder trials."""
    out=Path(out);report=json.loads((out/'report.json').read_text());torch.set_num_threads(1)
    parts,_=load_match(raw,'2269');rows=conditions(parts,100);f=load_checkpoint(out/'backbone-replay/final-trained.npz');ref=load_checkpoint(reference)
    with np.load(out/'verifier.npz',allow_pickle=False) as a:v={k:a[k].copy() for k in ('weights','mean','scale')}
    values=[];current=[];recovery=[]
    for s,p,c,mask in rows:
        prob,head=symmetric(f['model'],s,p,f['mean'],f['scale'],mask);q=probabilities(v,invariant_boundary_features(s,p,prob,mask))[:,1];values.append((prob,head,q));recovery.append(decode(prob,28,head,1.))
        pp,bb=symmetric(ref['model'],s,p,ref['mean'],ref['scale'],mask);current.append(decode(pp,ref['penalty'],bb,ref['boundary_strength']))
    for trial in report['trials']:
        pred=[decode(p,trial['penalty'],blend_boundary(b,q,trial['mix']),1.) for p,b,q in values]
        counts=[retention_counts(s['y'],a,b,c) for (s,_,_,_),a,b,c in zip(rows,current,recovery,pred)]
        trial['retention_audit']={key:sum(r[key] for r in counts) for key in ['recovered_spells','lost_recovered_spells']};trial['guards']['same_recovered_spells_retained']=trial['retention_audit']['lost_recovered_spells']==0;trial['eligible']=all(trial['guards'].values())
    eligible=[t for t in report['trials'] if t['eligible']];best=max(eligible,key=lambda t:t['selection_spell_f1']) if eligible else None
    report['selected']=best;report['best_unrestricted']=max(report['trials'],key=lambda t:t['selection_spell_f1']);report['retention_audit_completed']=True;report['retention_audit_rule']='Zero loss of same IoU0.5 recovered reference spells within known-label runs; specified before viewing007 trials'
    if best:
        bundle=out/'candidate';bundle.mkdir(exist_ok=True);shutil.copy2(out/'backbone-replay/final-trained.npz',bundle/'backbone.npz');shutil.copy2(out/'verifier.npz',bundle/'verifier.npz');dump(bundle/'decoder.json',{'penalty':best['penalty'],'mix':best['mix'],'validation_only':True});pred=[infer_bundle(bundle,s,p,mask) for s,p,c,mask in rows];assert abs(evaluate(rows,pred)['selection_spell_f1']-best['selection_spell_f1'])<1e-12;report['verification']={'reloaded_predictions_match':True}
    elif (out/'candidate').exists():
        (out/'candidate').rename(out/'rejected-before-retention-audit');report['verification']=None
    dump(out/'report.json',report);dump(out/'trials.json',report['trials']);return report

def run(raw,reference,out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new experiment directory')
    out.mkdir(parents=True);torch.set_num_threads(1)
    protocol={'experiment':'boundary-precision-007','train':list(TRAIN),'validation':['2269'],'test_loaded':False,'competition_data_used':False,'reference_sha256':digest(reference),'backbone':'deterministic replay of focused006, then frozen; no new training settings','verifier':'regularised logistic boundary classifier,20 epochs,seed91; training matches only; known-neighbourhood supervision','sampling':'every5th valid frame plus every2nd head>=0.5 frame; no validation hard-negative mining','mixes':[.25,.5,.75,1.],'penalties':[12,20,28,40],'selection':'highest mean clean/dropout spell F1 >0.5791444378581307; pooled misses<=292, false positives<=112, merges<=66, actual short predictions=0','same_spell_retention':'Also require zero loss of individual recovered reference spells under IoU0.5 matching','status':'exploratory reused validation following005/006; no independent test or automatic promotion'}
    dump(out/'protocol.json',protocol)
    reproduce_recovery(raw,reference,out/'backbone-replay',followup=True)
    checkpoint=out/'backbone-replay/final-trained.npz';f=load_checkpoint(checkpoint)
    xs=[];ys=[];audit=[]
    for i,mid in enumerate(TRAIN):
        parts,_=load_match(raw,mid)
        for j,(s,p) in enumerate(parts):
            t,valid=boundary_supervision(s['y'])
            for masked in (False,True):
                mask=missing_mask(len(t),42+2*i+j) if masked else None
                prob,head=symmetric(f['model'],s,p,f['mean'],f['scale'],mask);x=invariant_boundary_features(s,p,prob,mask);ix=fitting_indices(valid,head);xs.append(x[ix]);ys.append(t[ix]);audit.append({'match':mid,'period':j+1,'masked':masked,'samples':len(ix),'positives':int(t[ix].sum())})
        print('Verifier training features',mid,flush=True)
    x=np.concatenate(xs);y=np.concatenate(ys);verifier=fit_classifier(x,y,classes=2,epochs=20,seed=91);np.savez_compressed(out/'verifier.npz',**verifier);dump(out/'training-audit.json',audit)
    del xs,ys,x,y
    parts,_=load_match(raw,'2269');rows=conditions(parts,100)
    def score(pred):return {**evaluate(rows,pred),'diagnostics':spell_diagnostics(rows,pred)}
    ref=load_checkpoint(reference);current=score([decode(p,ref['penalty'],b,ref['boundary_strength']) for p,b in [symmetric(ref['model'],s,p,ref['mean'],ref['scale'],mask) for s,p,c,mask in rows]])
    values=[]
    for s,p,c,mask in rows:
        prob,head=symmetric(f['model'],s,p,f['mean'],f['scale'],mask);q=probabilities(verifier,invariant_boundary_features(s,p,prob,mask))[:,1];values.append((prob,head,q))
    recovery=score([decode(p,28,b,1.) for p,b,q in values]);assert abs(recovery['selection_spell_f1']-.5791444378581307)<1e-12,'Focused replay mismatch'
    trials=[];best=None
    for mix in protocol['mixes']:
        for penalty in protocol['penalties']:
            result=score([decode(p,penalty,blend_boundary(b,q,mix),1.) for p,b,q in values]);t=totals(result['diagnostics']);guards={'recovered_possessions_retained':t['missed_reference_spells_at_iou_0_5']<=292,'false_positives':t['false_positive_spells_at_iou_0_5']<=112,'merges':t['predictions_overlapping_multiple_reference_spells']<=66,'full_short_predictions':sum(result['diagnostics'][c]['full_prediction_spells_under_0_4_seconds'] for c in ('clean','masked'))==0,'spell_f1':result['selection_spell_f1']>recovery['selection_spell_f1']}
            trial={'mix':mix,'penalty':penalty,**result,'guards':guards,'eligible':all(guards.values())};trials.append(trial)
            if trial['eligible'] and (best is None or trial['selection_spell_f1']>best['selection_spell_f1']):best=trial
    verification=None
    if best:
        bundle=out/'candidate';bundle.mkdir();shutil.copy2(checkpoint,bundle/'backbone.npz');shutil.copy2(out/'verifier.npz',bundle/'verifier.npz');dump(bundle/'decoder.json',{'penalty':best['penalty'],'mix':best['mix'],'validation_only':True});restored=score([infer_bundle(bundle,s,p,mask) for s,p,c,mask in rows]);assert restored=={k:best[k] for k in ('clean','masked','selection_spell_f1','diagnostics')};verification={'reloaded_predictions_match':True}
    report={'protocol':protocol,'reference':current,'recovery_reference':recovery,'selected':best,'best_unrestricted':max(trials,key=lambda t:t['selection_spell_f1']),'trials':trials,'verification':verification,'fresh_test_results':None,'promoted_to_default':False,'backbone_sha256':digest(checkpoint),'verifier_sha256':digest(out/'verifier.npz')}
    dump(out/'report.json',report);dump(out/'trials.json',trials);report=audit_retention(raw,reference,out);best=report['selected'];print('COMPLETE',json.dumps({'reference':current['selection_spell_f1'],'recovery':recovery['selection_spell_f1'],'selected':None if best is None else best['selection_spell_f1'],'best':report['best_unrestricted']['selection_spell_f1']}),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('raw','reference','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--audit-existing',action='store_true')
    a=p.parse_args();(audit_retention if a.audit_existing else run)(a.raw,a.reference,a.out)
