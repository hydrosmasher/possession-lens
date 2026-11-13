"""OOF-trained short-excursion verifier; no blanket frame smoothing."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import torch
from .gated_recovery import dump
from .interaction_context import TRAIN
from .gated_rnn import load_checkpoint,symmetric
from .skillcorner_europe import load_match,digest
from .experiment import missing_mask
from .model import instantaneous_features,interaction_features,fit_classifier,probabilities,decode
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import spell_diagnostics
from .boundary_precision import retention_counts
from .europe_recovery import totals

def proposals(pred,max_frames=250):
    pred=np.asarray(pred);edges=np.r_[0,np.flatnonzero(pred[1:]!=pred[:-1])+1,len(pred)];items=[]
    for i in range(1,len(edges)-2):
        a,b=int(edges[i]),int(edges[i+1]);outer=int(pred[edges[i-1]]);team=int(pred[a])
        if outer in (0,1) and team in (0,1) and outer!=team and int(pred[b])==outer and b-a<=max_frames and a-edges[i-1]>=10 and edges[i+2]-b>=10:items.append((a,b,team,outer,int(edges[i-1]),int(edges[i+2])))
    return items

def role_features(base,interaction,team):
    other=1-team
    return np.column_stack([base[:,:6],base[:,6+12*team:18+12*team],base[:,6+12*other:18+12*other],interaction[:,5*team:5*team+5],interaction[:,5*other:5*other+5]])

def event_features(roles,p,head,event):
    a,b,team,outer,left,right=event;x=roles[team];middle=x[a:b];before=x[max(left,a-25):a].mean(0);after=x[b:min(right,b+25)].mean(0);mean=middle.mean(0);flank=(before+after)*.5;odds=np.log(np.maximum(p[a:b,team],1e-8)/np.maximum(p[a:b,outer],1e-8))
    confidence=np.array([np.log1p((b-a)/25),p[a:b,team].mean(),p[a:b,team].std(),p[a:b,outer].mean(),odds.mean(),odds.std(),head[max(0,a-5):a+6].mean(),head[b-5:min(len(head),b+6)].mean(),head[a:b].mean(),p[a:b,2].mean()],np.float32)
    return np.r_[mean,flank,np.abs(mean-flank),np.abs(before-after),confidence].astype(np.float32)

def compact_features(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=170:raise ValueError('Expected170 event features')
    middle=x[:,:40];flank=x[:,40:80]
    diffs=np.column_stack([middle[:,a]-middle[:,b] for a,b in [(6,18),(30,35),(31,36),(34,39),(32,37),(33,38)]])
    distance_delta=(middle[:,6]-middle[:,18])-(flank[:,6]-flank[:,18])
    return np.column_stack([x[:,160:170],middle[:,:3],diffs,distance_delta]).astype(np.float32)


def label_event(y,event):
    a,b,team,_,_,_=event;context=np.asarray(y)[max(0,a-25):min(len(y),b+25)]
    if ((context<0)|(context>=2)).any():return None
    purity=float(np.mean(np.asarray(y)[a:b]==team))
    return 1 if purity>=.8 else 0 if purity<=.2 else None

def apply_edits(pred,events,scores,threshold,max_frames):
    edited=np.asarray(pred).copy();accepted=0
    for index in sorted(range(len(events)),key=lambda i:(float(scores[i]),events[i][0])):
        a,b,team,outer,_,_=events[index]
        if scores[index]>=threshold or b-a>max_frames:continue
        # Do not cascade edits or recompute candidates after neighbouring changes.
        if edited[a-1]==outer and edited[b]==outer and np.all(edited[a:b]==team):edited[a:b]=outer;accepted+=1
    return edited,accepted

def infer(model,s,p,verifier,mask=None,threshold=.1,max_frames=125,compact=False):
    prob,head=symmetric(model['model'],s,p,model['mean'],model['scale'],mask);pred=decode(prob,28,head,1.);events=proposals(pred);base=instantaneous_features(s,p,ball_missing=mask);interaction=interaction_features(s,ball_missing=mask);roles=[role_features(base,interaction,t) for t in (0,1)]
    x=np.stack([event_features(roles,prob,head,e) for e in events]) if events else None
    scores=probabilities(verifier,compact_features(x) if compact else x)[:,1] if events else np.empty(0)
    return apply_edits(pred,events,scores,threshold,max_frames)

def run(raw,reference,backbone,fold_dir,out,compact=False):
    out=Path(out);fold_dir=Path(fold_dir)
    if out.exists():raise ValueError('Use a new output folder')
    out.mkdir(parents=True);torch.set_num_threads(1)
    spec=json.loads((fold_dir/'protocol.json').read_text());integrity=json.loads((fold_dir/'fold-integrity.json').read_text())
    if {f['held'] for f in spec['folds']}!=set(TRAIN) or any(set(f['train'])!=set(TRAIN)-{f['held']} for f in spec['folds']) or not all(f['training_only_scaling_verified'] for f in integrity):raise ValueError('Need verified match-excluded folds')
    protocol={'experiment':'event-precision-010','train':list(TRAIN),'validation':['2269'],'test_loaded':False,'competition_data_used':False,'fold_sha256':{m:digest(fold_dir/f'fold-{m}.npz') for m in TRAIN},'backbone_sha256':digest(backbone),'reference_sha256':digest(reference),'training':'OOF GRU A-B-A proposals, middle <=10s, outer flanks>=0.4s; logistic45 epochs seed110','target':'Only completely known +/-1s neighbourhoods: middle-team purity>=.8 positive,<=.2 negative, ambiguous excluded','features':'Role-ordered proximity, ball/player movement and trajectory pooled over spell and flanks; probability/head confidence and duration; no labels at inference','thresholds':[.1,.25],'max_frames':[125,250],'edit_policy':'Remove low-validity middle spell, sorted confidence; no new candidates or cascading edits; all other boundaries unchanged','selection':'mean spell F1>high-recall,misses<=292,FP<=112,merges<=66,actual short0,same recovered spells lost0','status':'exploratory reused validation following008/009; no fresh test or default promotion'}
    if compact:protocol.update(experiment='event-precision-011',feature_dimensions=20,l2=.05,status='Post010 exploratory compact+ridge follow-up; fixed before011 evaluation; no fresh test')
    dump(out/'protocol.json',protocol)
    xs=[];ys=[];audit=[]
    for i,mid in enumerate(TRAIN):
        f=load_checkpoint(fold_dir/f'fold-{mid}.npz');parts,_=load_match(raw,mid)
        for j,(s,p) in enumerate(parts):
            for masked in (False,True):
                mask=missing_mask(len(s['y']),42+2*i+j) if masked else None;prob,head=symmetric(f['model'],s,p,f['mean'],f['scale'],mask);pred=decode(prob,28,head,1.);events=proposals(pred);base=instantaneous_features(s,p,ball_missing=mask);interaction=interaction_features(s,ball_missing=mask);roles=[role_features(base,interaction,t) for t in (0,1)];positive=negative=excluded=0
                for e in events:
                    label=label_event(s['y'],e)
                    if label is None:excluded+=1;continue
                    xs.append(event_features(roles,prob,head,e));ys.append(label);positive+=label;negative+=1-label
                audit.append({'match':mid,'period':j+1,'masked':masked,'proposals':len(events),'positive':positive,'negative':negative,'excluded':excluded})
        print('OOF events',mid,'total labels',len(ys),'negative',int(np.sum(np.asarray(ys)==0)),flush=True)
    dump(out/'training-audit.json',audit)
    if set(ys)!={0,1}:raise ValueError('Insufficient positive/negative event supervision')
    x=np.stack(xs);x=compact_features(x) if compact else x
    verifier=fit_classifier(x,np.array(ys),classes=2,epochs=45,seed=110,l2=.05 if compact else .0005);np.savez_compressed(out/'verifier.npz',**verifier)
    return evaluate_frozen(raw,reference,backbone,verifier,out,protocol,compact)

def evaluate_frozen(raw,reference,backbone,verifier,out,protocol,compact):
    parts,_=load_match(raw,'2269');rows=conditions(parts,100);f=load_checkpoint(backbone);ref=load_checkpoint(reference);values=[];recovery=[];current=[]
    for s,p,c,mask in rows:
        prob,head=symmetric(f['model'],s,p,f['mean'],f['scale'],mask);pred=decode(prob,28,head,1.);recovery.append(pred);events=proposals(pred);base=instantaneous_features(s,p,ball_missing=mask);interaction=interaction_features(s,ball_missing=mask);roles=[role_features(base,interaction,t) for t in (0,1)];event_x=np.stack([event_features(roles,prob,head,e) for e in events]) if events else None;scores=probabilities(verifier,compact_features(event_x) if compact else event_x)[:,1] if events else np.empty(0);values.append((pred,events,scores));pp,bb=symmetric(ref['model'],s,p,ref['mean'],ref['scale'],mask);current.append(decode(pp,ref['penalty'],bb,ref['boundary_strength']))
    def score(pred):return {**evaluate(rows,pred),'diagnostics':spell_diagnostics(rows,pred)}
    rscore=score(recovery);assert abs(rscore['selection_spell_f1']-.5791444378581307)<1e-12
    trials=[]
    for threshold in protocol['thresholds']:
        for max_frames in protocol['max_frames']:
            edits=[apply_edits(p,e,q,threshold,max_frames) for p,e,q in values];pred=[p for p,n in edits];r=score(pred);dt=totals(r['diagnostics']);counts=[retention_counts(s['y'],a,b,c) for (s,_,_,_),a,b,c in zip(rows,current,recovery,pred)];ret={k:sum(v[k] for v in counts) for k in ['recovered_spells','lost_recovered_spells']};guard={'misses':dt['missed_reference_spells_at_iou_0_5']<=292,'false_positives':dt['false_positive_spells_at_iou_0_5']<=112,'merges':dt['predictions_overlapping_multiple_reference_spells']<=66,'short_predictions':sum(r['diagnostics'][c]['full_prediction_spells_under_0_4_seconds'] for c in ['clean','masked'])==0,'retention':ret['lost_recovered_spells']==0,'spell_f1':r['selection_spell_f1']>rscore['selection_spell_f1']};trials.append({'threshold':threshold,'max_frames':max_frames,**r,'edited_spells':sum(n for p,n in edits),'retention_audit':ret,'guards':guard,'eligible':all(guard.values())})
    eligible=[r for r in trials if r['eligible']];best=max(eligible,key=lambda r:r['selection_spell_f1']) if eligible else None;verification=None
    if best:
        bundle=out/'candidate';bundle.mkdir();shutil.copy2(backbone,bundle/'backbone.npz');shutil.copy2(out/'verifier.npz',bundle/'verifier.npz');dump(bundle/'decoder.json',{'policy':'short-excursion-v1','compact':compact,'threshold':best['threshold'],'max_frames':best['max_frames'],'validation_only':True});restored=load_checkpoint(bundle/'backbone.npz')
        with np.load(bundle/'verifier.npz',allow_pickle=False) as a:vv={k:a[k].copy() for k in ('weights','mean','scale')}
        settings=json.loads((bundle/'decoder.json').read_text())
        pp=[infer(restored,s,p,vv,mask,settings['threshold'],settings['max_frames'],settings['compact'])[0] for s,p,c,mask in rows];assert score(pp)=={k:best[k] for k in ['clean','masked','selection_spell_f1','diagnostics']};verification={'reloaded_predictions_match':True}
    report={'protocol':protocol,'reference':score(current),'recovery_reference':rscore,'selected':best,'best_unrestricted':max(trials,key=lambda r:r['selection_spell_f1']),'trials':trials,'verification':verification,'fresh_test_results':None,'promoted_to_default':False,'verifier_sha256':digest(out/'verifier.npz'),'feature_mode':'compact20' if compact else 'pooled170'};dump(out/'report.json',report);dump(out/'trials.json',trials);print('COMPLETE',report['best_unrestricted']['selection_spell_f1'],'selected',best is not None,flush=True);return report

def duration_refinement(raw,reference,backbone,verifier_path,out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new output folder')
    out.mkdir(parents=True);torch.set_num_threads(1)
    protocol={'experiment':'event-duration-012','status':'Post011 exploratory interpolation of duration cap; specified before012 evaluation; reused validation, not independent evidence','training_performed':False,'validation':['2269'],'test_loaded':False,'competition_data_used':False,'backbone_sha256':digest(backbone),'verifier_sha256':digest(verifier_path),'thresholds':[.1],'max_frames':[150,175,200],'compact':True,'selection':'mean spell F1>high-recall,misses<=292,FP<=112,merges<=66,actual short0,same recovered spells lost0'}
    dump(out/'protocol.json',protocol);shutil.copy2(verifier_path,out/'verifier.npz')
    with np.load(out/'verifier.npz',allow_pickle=False) as a:verifier={k:a[k].copy() for k in ('weights','mean','scale')}
    if verifier['mean'].shape!=(20,) or verifier['scale'].shape!=(20,) or verifier['weights'].shape!=(21,2) or any(not np.isfinite(a).all() for a in verifier.values()) or (verifier['scale']<=0).any():raise ValueError('Expected valid20-feature compact verifier')
    return evaluate_frozen(raw,reference,backbone,verifier,out,protocol,True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('raw','reference','backbone','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--fold-dir',type=Path);p.add_argument('--verifier',type=Path);p.add_argument('--compact',action='store_true');p.add_argument('--duration-refinement',action='store_true')
    a=p.parse_args()
    if a.duration_refinement:
        if a.verifier is None:p.error('--duration-refinement requires --verifier')
        duration_refinement(a.raw,a.reference,a.backbone,a.verifier,a.out)
    else:
        if a.fold_dir is None:p.error('training requires --fold-dir')
        run(a.raw,a.reference,a.backbone,a.fold_dir,a.out,a.compact)
