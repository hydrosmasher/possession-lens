"""Predeclared optional GRU/LSTM comparison on existing external splits only."""
from pathlib import Path
import argparse,json,time,platform
import numpy as np
import torch
from torch.nn import functional as F
from .gated_rnn import GatedPossession,scaling,normalise,make_window,predict,save_checkpoint,CORE,CONTEXT
from .interaction_context import observations,TRAIN
from .europe_recovery import boundary_supervision,totals,accepted
from .skillcorner_europe import load_match,digest
from .rnn_runtime import load_frozen,infer_frozen
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import spell_diagnostics
from .experiment import missing_mask,baseline_prediction
from .model import decode

VARIANTS=[
 {'name':'gru1','cell':'gru','layers':1,'bidirectional':False,'boundary':False},
 {'name':'bigru2','cell':'gru','layers':2,'bidirectional':True,'boundary':False},
 {'name':'bilstm2','cell':'lstm','layers':2,'bidirectional':True,'boundary':False},
 {'name':'bigru2_boundary','cell':'gru','layers':2,'bidirectional':True,'boundary':True},
]

def masked_loss(logits,y,head=None,target=None,valid=None,pos_weight=1.):
    known=y>=0
    ce=F.cross_entropy(logits[known],y[known]) if known.any() else logits.sum()*0
    if head is None:return ce
    bce=F.binary_cross_entropy_with_logits(head,target,reduction='none',pos_weight=head.new_tensor(pos_weight))
    return ce+.1*(bce*valid).sum()/valid.sum().clamp_min(1)

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def run(raw,reference,anchor,out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new experiment directory')
    out.mkdir(parents=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol={'experiment':'gated-003','train':TRAIN,'validation':['2269'],'viewed_tests_loaded':False,'competition_data_used':False,
      'variants':VARIANTS,'seed':42,'iterations':12,'windows_per_iteration':512,'batch_size':32,'core_frames':CORE,'context_each_side_frames':CONTEXT,
      'scoring_iterations':[4,8,12],'decoder_penalties':[24,36,48,60],'boundary_strengths':[0,.5,1],
      'features':'130 geometry/interaction values; learned continuous 48-dimensional projection, no player/team ID embedding',
      'regularisation':{'dropout':.15,'weight_decay':.0001,'layer_norm':'projection and recurrent output per timestep','clip_max_norm':1.,'initialisation':'Xavier input/output, orthogonal recurrent gate blocks, LSTM input forget bias 1'},
      'training':'AdamW lr .001; bounded-window BPTT; no cross-window state, no teacher forcing; unknown/context/padded labels excluded',
      'inference':'matching overlap-save windows, team-symmetric average, offline bidirectional context where configured',
      'selection':'highest validation clean/masked spell F1 > reference, with pooled misses, merge overlaps and false positives <= reference and historical short-pulse guard',
      'reference_sha256':digest(reference),'anchor_sha256':digest(anchor),'torch':torch.__version__,'python':platform.python_version(),
      'inapplicable_methods':{'scheduled_sampling':'no previous label is ever an input; no teacher-forcing exposure bias','word_embeddings':'continuous tracking features, not text tokens; use learned continuous projection','language_RL':'no language generation or diversity objective; use supervised labels and guarded spell decoding'}}
    write(out/'protocol.json',protocol);print('Protocol frozen. Loading only five train matches and validation 2269.',flush=True)
    parts=[]
    for mid in TRAIN:
        items,_=load_match(raw,mid);parts.extend(items);print('Loaded training match',mid,flush=True)
    validation,_=load_match(raw,'2269');rows=conditions(validation,100)
    previous=load_frozen(reference);refpred=[infer_frozen(previous,s,p,mask)[0] for s,p,c,mask in rows]
    refscore=evaluate(rows,refpred);refdiag=spell_diagnostics(rows,refpred);reftot=totals(refdiag)
    old=load_frozen(anchor);anchor_diag=spell_diagnostics(rows,[infer_frozen(old,s,p,mask)[0] for s,p,c,mask in rows])
    baseline=evaluate(rows,[baseline_prediction(s,50,mask) for s,p,c,mask in rows]);seq=[]
    for i,(s,p) in enumerate(parts):
        target,valid=boundary_supervision(s['y']);y=np.where(s['y']<2,s['y'],-1)
        for swap in (False,True):
            yy=np.where(y<0,-1,1-y) if swap else y.copy()
            for masked in (False,True):
                seq.append((observations(s,p,missing_mask(len(y),42+i) if masked else None,swap),yy,target,valid.astype(np.float32)))
    mean,scale=scaling(seq)
    seq=[(normalise(x,mean,scale),y,t,v) for x,y,t,v in seq]
    valx=[(observations(s,p,mask),observations(s,p,mask,True)) for s,p,c,mask in rows]
    valseq=[(normalise(observations(s,p),mean,scale),np.where(s['y']<2,s['y'],-1),*boundary_supervision(s['y'])) for s,p in validation]
    # Fixed clean probes, independent of the stochastic optimisation sampler.
    clean=[(normalise(observations(s,p),mean,scale),np.where(s['y']<2,s['y'],-1),*boundary_supervision(s['y'])) for s,p in parts]
    def probe_samples(items):
        rng=np.random.default_rng(101);samples=[]
        for _ in range(64):
            x,y,t,v=items[rng.integers(len(items))];samples.append(make_window(x,y,int(rng.integers(len(x))),t,v))
        return [torch.from_numpy(np.stack([s[k] for s in samples])) for k in range(4)]
    train_probe=probe_samples(clean);val_probe=probe_samples(valseq)
    pos=sum(float((t*v).sum()) for x,y,t,v in seq);neg=sum(float(((1-t)*v).sum()) for x,y,t,v in seq);pos_weight=min(10.,neg/max(pos,1))
    history=[];trials=[];best=None
    for variant in VARIANTS:
        torch.manual_seed(42);rng=np.random.default_rng(42);config={k:v for k,v in variant.items() if k!='name'};model=GatedPossession(**config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        print('Training',variant['name'],'parameters',sum(p.numel() for p in model.parameters()),flush=True)
        for epoch in range(1,13):
            model.train();losses=[];norms=[];start_time=time.monotonic()
            for step in range(16):
                samples=[]
                for _ in range(32):
                    x,y,t,v=seq[rng.integers(len(seq))];samples.append(make_window(x,y,int(rng.integers(len(x))),t,v))
                x,y,t,v=[torch.from_numpy(np.stack([s[k] for s in samples])) for k in range(4)]
                optimizer.zero_grad(set_to_none=True);logits,head=model(x);loss=masked_loss(logits,y,head,t,v,pos_weight)
                if not torch.isfinite(loss):raise ValueError('Nonfinite gated training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                losses.append(float(loss.detach()));norms.append(float(norm))
            def ce_probe(tensors):
                with torch.inference_mode():
                    x,y,_,_=tensors;logits,_=model(x);known=y>=0
                    return float(F.cross_entropy(logits[known],y[known]))
            model.eval();tr=ce_probe(train_probe);va=ce_probe(val_probe)
            h={'variant':variant['name'],'iteration':epoch,'training_objective':float(np.mean(losses)),'train_probe_cross_entropy':tr,'validation_probe_cross_entropy':va,'generalisation_gap':va-tr,'mean_gradient_norm_before_clip':float(np.mean(norms)),'max_gradient_norm_before_clip':max(norms),'fraction_steps_clipped':float(np.mean(np.array(norms)>1)),'optimiser_steps':epoch*16,'training_seconds':time.monotonic()-start_time};history.append(h);write(out/'learning.json',history)
            print(variant['name'],epoch,'train CE',round(tr,4),'val CE',round(va,4),'gap',round(va-tr,4),flush=True)
            if epoch not in (4,8,12):continue
            probabilities=[];boundaries=[]
            for a,b in valx:
                pa,ba=predict(model,a,mean,scale);pb,bb=predict(model,b,mean,scale);probabilities.append((pa+pb[:,[1,0,2]])*.5);boundaries.append((ba+bb)*.5)
            for strength in ([0,.5,1] if variant['boundary'] else [0]):
                for penalty in protocol['decoder_penalties']:
                    preds=[decode(p,penalty,b if strength else None,strength) for p,b in zip(probabilities,boundaries)];score=evaluate(rows,preds);diag=spell_diagnostics(rows,preds);tt=totals(diag)
                    guard={k:tt[k]<=reftot[k] for k in ('missed_reference_spells_at_iou_0_5','false_positive_spells_at_iou_0_5','predictions_overlapping_multiple_reference_spells')};guard['historical_switch_guard']=accepted(diag,anchor_diag)
                    eligible=all(guard.values()) and score['selection_spell_f1']>refscore['selection_spell_f1']
                    trial={'variant':variant['name'],'iteration':epoch,'penalty':penalty,'boundary_strength':strength,**score,'diagnostics':diag,'guards':guard,'eligible':eligible};trials.append(trial)
                    if eligible and (best is None or score['selection_spell_f1']>best['selection_spell_f1']):best=trial;save_checkpoint(out/'candidate.npz',model,mean,scale,penalty,strength)
            write(out/'validation-trials.json',trials);print('Best eligible:',None if best is None else best['selection_spell_f1'],flush=True)
    report={'protocol':protocol,'baseline_validation':baseline,'reference_validation':refscore,'reference_diagnostics':refdiag,'anchor_diagnostics':anchor_diag,'selected':best,'best_unrestricted_by_variant':[max([t for t in trials if t['variant']==v['name']],key=lambda t:t['selection_spell_f1']) for v in VARIANTS],'validation_trials':len(trials),'improved_validation':best is not None,'fresh_test_results':None,'promoted_to_default':False,'limitations':['Repeated validation selection can overfit 2269.','Partial provider labels are weak supervision, not full-match independent ground truth.','No new test or competition labels used.','Bidirectional context is offline and bounded by the documented window policy.','Architecture comparison is bounded, not proof of the globally best possible RNN.']}
    write(out/'report.json',report);print('Completed',json.dumps({'improved_validation':best is not None,'selected':None if best is None else best['selection_spell_f1']}),flush=True);return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','reference','anchor','out'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();run(a.raw,a.reference,a.anchor,a.out)
