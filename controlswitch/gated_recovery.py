"""Bounded fine-tuning on existing external train/validation matches only."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from .gated_rnn import load_checkpoint,save_checkpoint,normalise,make_window,symmetric,CONTEXT,CORE
from .interaction_context import observations,TRAIN
from .europe_recovery import boundary_supervision,loss_weights,totals
from .skillcorner_europe import load_match,digest
from .experiment import missing_mask
from .rnn_benchmark import conditions,evaluate
from .rnn_boundaries import spell_diagnostics
from .model import decode

def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def transition_indices(y):
    y=np.asarray(y)
    return np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[1:]<2)&(y[:-1]>=0)&(y[:-1]<2))+1

def sample_start(y,rng,focused=False):
    events=transition_indices(y) if focused else []
    if len(events) and rng.random()<.5:
        return int(np.clip(rng.choice(events)-rng.integers(CORE//4,3*CORE//4),0,len(y)-1))
    return int(rng.integers(len(y)))

def weighted_ce(logits,y,w):
    known=(y>=0)&(y<3)
    if not known.any():return logits.sum()*0
    loss=F.cross_entropy(logits[known],y[known],reduction='none');weight=w[known]
    return (loss*weight).sum()/weight.sum().clamp_min(1)

def run(raw,checkpoint,out,followup=False):
    out=Path(out)
    if out.exists():raise ValueError('Use a new output directory')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol={'experiment':'gated-recovery-005','train':list(TRAIN),'validation':['2269'],'test_loaded':False,'competition_data_used':False,'seed':73,'epochs':4,'steps_per_epoch':16,'batch':16,'lr':.0002,'weight_decay':.0001,'clip_norm':1.,'variants':['uniform','spell_boundary_focus'],'score_epochs':[2,4],'penalties':[16,20,24],'boundary_strength':1.,'checkpoint_sha256':digest(checkpoint),'selection':'highest eligible validation mean clean/dropout spell F1 above reference; misses, false positives and merge overlaps no worse; full-sequence short predictions no worse','limitations':'Repeated existing validation; no fresh test, no automatic promotion. Focused sampling changes training distribution; compare uniform fine-tuning control.'}
    if followup:
        protocol.update(experiment='gated-recovery-006',variants=['spell_boundary_focus'],score_epochs=[4],penalties=[28,32,40,48],status='Exploratory follow-up after viewing005: best focused trial at upper penalty24 boundary; fixed stronger penalties before this run')
    dump(out/'protocol.json',protocol);print('Protocol saved; external-only five training matches, validation 2269.',flush=True)
    frozen=load_checkpoint(checkpoint);parts=[]
    for mid in TRAIN:
        p,_=load_match(raw,mid);parts.extend(p);print('Loaded',mid,flush=True)
    val,_=load_match(raw,'2269');rows=conditions(val,100)
    def score(pred):return {**evaluate(rows,pred),'diagnostics':spell_diagnostics(rows,pred)}
    vals=[symmetric(frozen['model'],s,p,frozen['mean'],frozen['scale'],mask) for s,p,c,mask in rows]
    reference=score([decode(p,frozen['penalty'],b,frozen['boundary_strength']) for p,b in vals]);rt=totals(reference['diagnostics'])
    pulse=lambda d:sum(d[c]['full_prediction_spells_under_0_4_seconds'] for c in ('clean','masked'))
    seq=[]
    for i,(s,p) in enumerate(parts):
        target,valid=boundary_supervision(s['y']);y=np.where(s['y']<2,s['y'],-1);weights=loss_weights(s['y'])
        for swap in (False,True):
            yy=np.where(y<0,-1,1-y) if swap else y.copy()
            for masked in (False,True):
                x=normalise(observations(s,p,missing_mask(len(y),42+i) if masked else None,swap),frozen['mean'],frozen['scale'])
                seq.append((x,yy,target,valid.astype(np.float32),weights))
    pos=sum(float((t*v).sum()) for x,y,t,v,w in seq);neg=sum(float(((1-t)*v).sum()) for x,y,t,v,w in seq);posweight=min(10.,neg/max(pos,1))
    # Fixed clean probes across every training/validation part, independent of training sampler.
    def probes(items):
        rng=np.random.default_rng(101);batch=[]
        for s,p in items:
            x=normalise(observations(s,p),frozen['mean'],frozen['scale']);y=np.where(s['y']<2,s['y'],-1)
            for _ in range(4):batch.append(make_window(x,y,int(rng.integers(len(y)))))
        return torch.from_numpy(np.stack([b[0] for b in batch])),torch.from_numpy(np.stack([b[1] for b in batch]))
    trainprobe=probes(parts);valprobe=probes(val)
    def probe(model,p):
        with torch.inference_mode():
            x,y=p;logits,_=model(x);return float(weighted_ce(logits,y,torch.ones_like(y,dtype=torch.float32)))
    history=[];trials=[];best=None
    for variant in protocol['variants']:
        torch.manual_seed(73);rng=np.random.default_rng(73);model=load_checkpoint(checkpoint)['model'];opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay']);focused=variant!='uniform'
        for epoch in range(1,5):
            model.train();losses=[];norms=[];start=time.monotonic()
            for _ in range(protocol['steps_per_epoch']):
                samples=[];ww=[]
                for _ in range(protocol['batch']):
                    x,y,t,v,w=seq[rng.integers(len(seq))];a=sample_start(y,rng,focused);samples.append(make_window(x,y,a,t,v));weight=np.zeros(CORE+2*CONTEXT,np.float32);size=min(CORE,len(y)-a);weight[CONTEXT:CONTEXT+size]=w[a:a+size] if focused else 1;ww.append(weight)
                x,y,t,v=[torch.from_numpy(np.stack([s[k] for s in samples])) for k in range(4)];w=torch.from_numpy(np.stack(ww));opt.zero_grad(set_to_none=True);logits,head=model(x);loss=weighted_ce(logits,y,w)
                loss=loss+.1*(F.binary_cross_entropy_with_logits(head,t,reduction='none',pos_weight=head.new_tensor(posweight))*v).sum()/v.sum().clamp_min(1)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
            model.eval();tr=probe(model,trainprobe);va=probe(model,valprobe);row={'variant':variant,'epoch':epoch,'objective':float(np.mean(losses)),'train_probe_ce':tr,'validation_probe_ce':va,'generalisation_gap':va-tr,'max_gradient_norm':max(norms),'fraction_steps_clipped':float(np.mean(np.array(norms)>1)),'seconds':time.monotonic()-start};history.append(row);dump(out/'learning.json',history);print(json.dumps(row),flush=True)
            if epoch not in protocol['score_epochs']:continue
            values=[symmetric(model,s,p,frozen['mean'],frozen['scale'],mask) for s,p,c,mask in rows]
            for penalty in protocol['penalties']:
                result=score([decode(p,penalty,b,1.) for p,b in values]);dt=totals(result['diagnostics']);guards={k:dt[k]<=rt[k] for k in ('missed_reference_spells_at_iou_0_5','false_positive_spells_at_iou_0_5','predictions_overlapping_multiple_reference_spells')};guards['full_short_predictions']=pulse(result['diagnostics'])<=pulse(reference['diagnostics']);trial={'variant':variant,'epoch':epoch,'penalty':penalty,**result,'guards':guards,'eligible':all(guards.values())};trials.append(trial)
                if trial['eligible'] and result['selection_spell_f1']>reference['selection_spell_f1'] and (best is None or result['selection_spell_f1']>best['selection_spell_f1']):
                    best=trial;save_checkpoint(out/'candidate.npz',model,frozen['mean'],frozen['scale'],penalty,1.)
            dump(out/'trials.json',trials);print('Best eligible',None if best is None else best['selection_spell_f1'],flush=True)
    # Preserve the final fine-tuned weights for external-only follow-up studies.
    save_checkpoint(out/'final-trained.npz',model,frozen['mean'],frozen['scale'],28,1.)
    verification=None
    if best:
        f=load_checkpoint(out/'candidate.npz');reloaded=score([decode(p,f['penalty'],b,f['boundary_strength']) for p,b in [symmetric(f['model'],s,p,f['mean'],f['scale'],mask) for s,p,c,mask in rows]])
        assert reloaded=={k:best[k] for k in ('clean','masked','selection_spell_f1','diagnostics')};verification={'reload_matches':True,'sha256':digest(out/'candidate.npz')}
    report={'protocol':protocol,'reference':reference,'selected':best,'best_unrestricted':max(trials,key=lambda t:t['selection_spell_f1']),'trials':trials,'checkpoint_verification':verification,'fresh_test_results':None,'promoted_to_default':False}
    dump(out/'report.json',report);print('COMPLETE',json.dumps({'reference':reference['selection_spell_f1'],'selected':None if best is None else best['selection_spell_f1'],'unrestricted':report['best_unrestricted']['selection_spell_f1']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('raw','checkpoint','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--decoder-followup',action='store_true')
    a=p.parse_args();run(a.raw,a.checkpoint,a.out,a.decoder_followup)
