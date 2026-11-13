"""Match-excluded, from-scratch GRU predictions for external verifier training."""
from pathlib import Path
import argparse,gc,json,shutil,time
import numpy as np
import torch
from .gated_rnn import GatedPossession,scaling,normalise,make_window,save_checkpoint,load_checkpoint,symmetric
from .gated_experiment import masked_loss
from .gated_recovery import dump
from .interaction_context import TRAIN,observations
from .skillcorner_europe import load_match,digest
from .europe_recovery import boundary_supervision,totals
from .experiment import missing_mask
from .model import fit_classifier,probabilities,decode
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import invariant_boundary_features,spell_diagnostics
from .boundary_precision import fitting_indices,blend_boundary,retention_counts,infer_bundle

def folds(matches):
    matches=list(matches)
    if len(matches)<3 or len(set(matches))!=len(matches):raise ValueError('Need distinct match IDs')
    return [{'held':m,'train':[x for x in matches if x!=m]} for m in matches]

def training_files(paths,spec):
    expected=set(paths)-{spec['held']}
    if spec['held'] not in paths or len(spec['train'])!=len(set(spec['train'])) or set(spec['train'])!=expected:
        raise ValueError('Fold must exclude exactly its held match')
    return [p for mid in spec['train'] for p in paths[mid]]

def prepare(raw,cache):
    cache.mkdir();paths={}
    for i,mid in enumerate(TRAIN):
        parts,_=load_match(raw,mid);paths[mid]=[]
        for j,(s,p) in enumerate(parts):
            y=np.where(s['y']<2,s['y'],-1);t,v=boundary_supervision(s['y']);items={'y':y,'target':t.astype(np.float32),'valid':v.astype(np.float32)}
            for masked in (False,True):
                mask=missing_mask(len(y),42+2*i+j) if masked else None
                for swap in (False,True):items[f'x{int(masked)*2+int(swap)}']=observations(s,p,mask,swap)
            path=cache/f'{mid}-{j}.npz';np.savez(path,**items);paths[mid].append(path)
        del parts;gc.collect();print('Prepared training-only features',mid,flush=True)
    return paths

def fit_fold(paths,spec,index,out):
    fold_files=training_files(paths,spec);torch.manual_seed(108+index);rng=np.random.default_rng(108+index);seq=[]
    for path in fold_files:
        with np.load(path,allow_pickle=False) as a:
            y=a['y'];t=a['target'];v=a['valid']
            for k in range(4):seq.append((a[f'x{k}'],np.where(y<0,-1,1-y) if k%2 else y,t,v))
    mean,scale=scaling(seq);seq=[(normalise(x,mean,scale),y,t,v) for x,y,t,v in seq];gc.collect()
    model=GatedPossession(cell='gru',layers=2,bidirectional=True,boundary=True);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    pos=sum(float((t*v).sum()) for x,y,t,v in seq);neg=sum(float(((1-t)*v).sum()) for x,y,t,v in seq);pw=min(10.,neg/max(pos,1));history=[]
    for epoch in range(1,13):
        model.train();losses=[];norms=[];start=time.monotonic()
        for _ in range(16):
            samples=[]
            for _ in range(32):
                x,y,t,v=seq[rng.integers(len(seq))];samples.append(make_window(x,y,int(rng.integers(len(x))),t,v))
            x,y,t,v=[torch.from_numpy(np.stack([s[k] for s in samples])) for k in range(4)];opt.zero_grad(set_to_none=True);logits,head=model(x);loss=masked_loss(logits,y,head,t,v,pw)
            if not torch.isfinite(loss):raise ValueError('Nonfinite fold training')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
        history.append({'held_match':spec['held'],'epoch':epoch,'training_objective':float(np.mean(losses)),'max_gradient_norm':max(norms),'fraction_clipped':float(np.mean(np.asarray(norms)>1)),'seconds':time.monotonic()-start});dump(out/f'fold-{spec["held"]}-learning.json',history)
        if epoch%3==0:print('Fold',spec['held'],'epoch',epoch,'loss',round(np.mean(losses),4),flush=True)
    save_checkpoint(out/f'fold-{spec["held"]}.npz',model,mean,scale,28,1.);del seq,opt;gc.collect();return {'model':model,'mean':mean,'scale':scale}

def verify_fold_scaling(paths,out):
    out=Path(out);moments={}
    for mid,files in paths.items():
        count=0;total=np.zeros(130,np.float64);squares=np.zeros(130,np.float64)
        for path in files:
            with np.load(path,allow_pickle=False) as a:
                for k in range(4):
                    x=a[f'x{k}'].astype(np.float64);count+=len(x);total+=x.sum(0);squares+=np.einsum('ij,ij->j',x,x)
        moments[mid]=(count,total,squares)
    audit=[]
    for spec in folds(list(paths)):
        count=sum(moments[m][0] for m in spec['train']);mean=sum(moments[m][1] for m in spec['train'])/count;scale=np.maximum(np.sqrt(np.maximum(0,sum(moments[m][2] for m in spec['train'])/count-mean*mean)),.05)
        f=load_checkpoint(out/f'fold-{spec["held"]}.npz');valid=np.allclose(f['mean'],mean,rtol=1e-5,atol=1e-6) and np.allclose(f['scale'],scale,rtol=1e-5,atol=1e-6)
        if not valid:raise ValueError('Fold scaling mismatch')
        audit.append({'held_match':spec['held'],'training_only_scaling_verified':bool(valid),'training_matches':spec['train']})
    dump(out/'fold-integrity.json',audit);return audit

def run(raw,reference,backbone,out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new output folder')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol={'experiment':'crossfit-boundary-008','folds':folds(TRAIN),'validation':['2269'],'test_loaded':False,'competition_data_used':False,'initialisation':'each fold from scratch; no pretrained weights; fold-only scaling','fold_architecture':'2-layer bidirectional GRU with boundary head;12 epochs,16 steps,32 windows; AdamW.001,weight decay.0001,clip1,seed108+foldindex','verifier':'same natural+high-head sampling as007; logistic20 epochs,seed91; only match-excluded predictions','mixes':[.25,.5],'penalties':[20,28,40],'reference_sha256':digest(reference),'backbone_sha256':digest(backbone),'selection':'mean spell F1>recovery; misses<=292,FP<=112,merges<=66,actual short=0 and zero loss of same recovered reference spells','limitations':['External validation reused; not fresh-test evidence.','Fold predictions excluded from fold weights and scaling, but all five matches are used to train the verifier.','Scratch-fold and fine-tuned final-backbone distributions may differ.','Provider labels are partial automatic supervision; no default promotion.']}
    dump(out/'protocol.json',protocol);paths=prepare(raw,out/'feature-cache');xs=[];ys=[];audit=[]
    for i,spec in enumerate(protocol['folds']):
        f=fit_fold(paths,spec,i,out);parts,_=load_match(raw,spec['held'])
        for j,(s,p) in enumerate(parts):
            t,valid=boundary_supervision(s['y'])
            for masked in (False,True):
                mask=missing_mask(len(t),42+2*i+j) if masked else None;prob,head=symmetric(f['model'],s,p,f['mean'],f['scale'],mask);x=invariant_boundary_features(s,p,prob,mask);ix=fitting_indices(valid,head);xs.append(x[ix]);ys.append(t[ix]);audit.append({'held_match':spec['held'],'excluded_from_weights_and_scaling':spec['held'] not in spec['train'],'period':j+1,'masked':masked,'samples':len(ix),'positives':int(t[ix].sum())})
        del f,parts;gc.collect();print('Excluded-match verifier features',spec['held'],flush=True)
    verify_fold_scaling(paths,out)
    dump(out/'fold-learning.json',[r for p in sorted(out.glob('fold-*-learning.json')) for r in json.loads(p.read_text())])
    verifier=fit_classifier(np.concatenate(xs),np.concatenate(ys),classes=2,epochs=20,seed=91);np.savez_compressed(out/'verifier.npz',**verifier);dump(out/'training-audit.json',audit);del xs,ys;gc.collect()
    parts,_=load_match(raw,'2269');rows=conditions(parts,100);f=load_checkpoint(backbone);ref=load_checkpoint(reference);values=[];current=[];recovery=[]
    for s,p,c,mask in rows:
        prob,head=symmetric(f['model'],s,p,f['mean'],f['scale'],mask);q=probabilities(verifier,invariant_boundary_features(s,p,prob,mask))[:,1];values.append((prob,head,q));recovery.append(decode(prob,28,head,1.));pp,bb=symmetric(ref['model'],s,p,ref['mean'],ref['scale'],mask);current.append(decode(pp,ref['penalty'],bb,ref['boundary_strength']))
    def score(pred):return {**evaluate(rows,pred),'diagnostics':spell_diagnostics(rows,pred)}
    reference_score=score(current);recovery_score=score(recovery);assert abs(recovery_score['selection_spell_f1']-.5791444378581307)<1e-12
    trials=[]
    for mix in protocol['mixes']:
        for penalty in protocol['penalties']:
            pred=[decode(p,penalty,blend_boundary(b,q,mix),1.) for p,b,q in values];r=score(pred);dt=totals(r['diagnostics']);counts=[retention_counts(s['y'],a,b,c) for (s,_,_,_),a,b,c in zip(rows,current,recovery,pred)];ret={k:sum(v[k] for v in counts) for k in ['recovered_spells','lost_recovered_spells']};guard={'misses':dt['missed_reference_spells_at_iou_0_5']<=292,'false_positives':dt['false_positive_spells_at_iou_0_5']<=112,'merges':dt['predictions_overlapping_multiple_reference_spells']<=66,'short_predictions':sum(r['diagnostics'][c]['full_prediction_spells_under_0_4_seconds'] for c in ['clean','masked'])==0,'retention':ret['lost_recovered_spells']==0,'spell_f1':r['selection_spell_f1']>recovery_score['selection_spell_f1']};trials.append({'mix':mix,'penalty':penalty,**r,'retention_audit':ret,'guards':guard,'eligible':all(guard.values())});print('Trial',mix,penalty,round(r['selection_spell_f1'],4),guard,flush=True)
    eligible=[r for r in trials if r['eligible']];best=max(eligible,key=lambda r:r['selection_spell_f1']) if eligible else None;verification=None
    if best:
        bundle=out/'candidate';bundle.mkdir();shutil.copy2(backbone,bundle/'backbone.npz');shutil.copy2(out/'verifier.npz',bundle/'verifier.npz');dump(bundle/'decoder.json',{'penalty':best['penalty'],'mix':best['mix'],'validation_only':True});restored=score([infer_bundle(bundle,s,p,mask) for s,p,c,mask in rows]);assert restored=={k:best[k] for k in ['clean','masked','selection_spell_f1','diagnostics']};verification={'reloaded_predictions_match':True}
    report={'protocol':protocol,'reference':reference_score,'recovery_reference':recovery_score,'selected':best,'best_unrestricted':max(trials,key=lambda t:t['selection_spell_f1']),'trials':trials,'verification':verification,'fresh_test_results':None,'promoted_to_default':False,'verifier_sha256':digest(out/'verifier.npz')};dump(out/'report.json',report);dump(out/'trials.json',trials);print('COMPLETE',report['best_unrestricted']['selection_spell_f1'],'selected',best is not None,flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('raw','reference','backbone','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.raw,a.reference,a.backbone,a.out)
