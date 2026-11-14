"""047: masked-ball auxiliary supervision for the richer-motion GRU."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from .gated_rnn import GatedPossession,scaling,normalise,make_window,symmetric,save_checkpoint,load_checkpoint,CORE,CONTEXT
from .joint_state import joint_features,swap_joint
from .narrow_boundary import bundle040,narrow_targets

def observations(s,pitch,mask=None,swap=False):
    x=joint_features(s,pitch,mask);return swap_joint(x) if swap else x

def targets(y):return narrow_targets(y,2)
from .spell_training import spell_weights,bundle031
from .boundary_had import boundary_observations,adaptive_decode
from .ranked_pipeline import infer
from .control_state import evaluate_accuracy
from .joint_state import joint_features as state_features,swap_joint as swap_state_columns
from .control_correction import blend_control
from .boundary_alignment import eligible
from .model import probabilities
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .external import period_arrays
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest
from . import tree_model
from .recurrent_pipeline import infer as reference_infer

MIXES=(.25,.5,1.)
def weighted_window(x,y,w,t,v,start,ball,ball_valid):
    xx,yy,tt,vv=make_window(x,y,start,t,v);ww=np.zeros(len(yy),np.float32);n=min(CORE,len(x)-start);ww[CONTEXT:CONTEXT+n]=w[start:start+n];ww[yy<0]=0
    bt=np.zeros((len(yy),2),np.float32);bv=np.zeros(len(yy),np.float32);bt[CONTEXT:CONTEXT+n]=ball[start:start+n];bv[CONTEXT:CONTEXT+n]=ball_valid[start:start+n];bv[yy<0]=0
    return xx,yy,ww,tt,vv,bt,bv

def loss_value(logits,y,w,head,t,v,pos_weight):
    known=y>=0
    ce=(F.cross_entropy(logits[known],y[known],reduction='none')*w[known]).sum()/w[known].sum().clamp_min(1) if known.any() else logits.sum()*0
    boundary=F.binary_cross_entropy_with_logits(head,t,reduction='none',pos_weight=head.new_tensor(pos_weight))
    return ce+.1*(boundary*v).sum()/v.sum().clamp_min(1)

def auxiliary_losses(model,aux,x,y,w,t,v,ball,ball_valid,pos_weight):
    h=model.representation(x);logits=model.classifier(h);head=model.boundary_head(h).squeeze(-1)
    control=loss_value(logits,y,w,head,t,v,pos_weight);estimate=.6*torch.tanh(aux(h));mse=(((estimate-ball)**2)*ball_valid[:,:,None]).sum()/(2*ball_valid.sum().clamp_min(1))
    return control+5*mse,control,mse

def reference_bundle(fi,roots,control_root):
    return bundle040(fi,roots,control_root)

def predict_variants(bundle,gated,s,pitch,mask=None):
    base,p=infer(bundle,s,pitch,mask);g,_=symmetric(gated['model'],s,pitch,gated['mean'],gated['scale'],mask);g=g[:,:2]/np.maximum(g[:,:2].sum(1,keepdims=True),1e-12);x=state_features(s,pitch,mask);a=tree_model.predict(bundle['models']['control'],x);b=tree_model.predict(bundle['models']['control'],swap_state_columns(x));tree=(a+b[:,[1,0,2]])[:,:2]*.5;d=p[:,2];out={}
    for mix in MIXES:
        live=(1-mix)*(p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12))+mix*g;q=np.column_stack([live*(1-d[:,None]),d]);event=probabilities(bundle['models']['boundary'],boundary_observations(s,pitch,q,mask))[:,1];out[f'gru{mix:g}']=adaptive_decode(q,event,1.)
    return base,out

def run(cache_root,roots,control_root,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output folder')
    prereq=json.loads((Path(control_root)/'report.json').read_text())
    if not prereq['recommended_for_freeze'] or prereq['selected']['name']!='joint200':raise ValueError('Expected frozen040 selection')
    old_report=json.loads((Path(roots['motion_gru_root'])/'report.json').read_text())
    if not old_report['recommended_for_freeze']:raise ValueError('Expected frozen043 audit')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol={'version':'ball-auxiliary-047','reference':'frozen043','source':'Only opposite-match existing first-division Sportec WMX/WN1; no challenge inputs, labels or video. Both games heavily reused; no independent test.',
     'auxiliary_ball':'External original tracked ball xy / pitch dimensions, only known-live synthetically missing core frames. Separate128to2 linear+tanh*.6 head, weight5 MSE. Targets never enter inputs. Auxiliary head is discarded at inference; only learned GRU representation is retained.',
     'architecture':'508joint motion/shape/control features; learned64projection+LayerNorm+GELU; two64unit bidirectional GRU layers; locked dropout.25,outputLayerNorm, H/A classifier with unusedD logit, auxiliary direct-turnover head.',
     'training':'From scratch, same143base seed; auxiliary head uses isolated147seed;24fixed iterations,512sampled windows/iteration,batch32; core256,context64eachside; same finite-context overlap-save inference. AdamWlr.0005,weightdecay.001,maxgradientnorm1, Xavier input/output and orthogonal recurrent gates.',
     'supervision':'Known-live H/A only; train-half inverse-sqrt spell weights from032; D/X/padding/context excluded from CE. Aux boundary BCE weight.1,positiveweight capped10; entirely known-live +-12frame valid neighbourhood, +-2frame auxiliary event target.',
     'augmentation':'Four fixed clean/shortgap/longgap/tracker-noise conditions and exact team swaps. Scaling from training observations only. No previous label input, scheduled sampling inapplicable.',
     'variants':'GRU conditional live evidence weights .25/.5/1, remaining weight040conditional live evidence. Frozen040 dead experts and031boundary decoder. One fixed final checkpoint, no epoch selection.',
     'selection':'Official four-condition mean spellF1 gain>=.003 over043; each clean/gap fold decline<=.01, noise fold no decline, deadF1 decline<=.005. Highest eligible mean. Report false-positive trade-off.',
     'audit':'Freeze one winner plus hashes before new-observation-draw audit seeds2914/3114/3518. Only winner and043 decoded; audit mean must improve or retain043, no runner-up selection.',
     'competition_data_used':False,'competition_percentile_known':False,'torch':torch.__version__}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous external cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
    folds=[];all_rows=[];all_base=[];names=[f'gru{m:g}' for m in MIXES];all_preds={k:[] for k in names}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();bundle,paths=reference_bundle(fi,roots,control_root);seqs=[]
        for half in (0,1):
            seq=period_arrays(records[train],half);yy=np.where(seq['y']<2,seq['y'],-1);w=spell_weights(seq['y']).astype(np.float32);t,v=targets(seq['y']);v=v.astype(np.float32);ball_target=np.nan_to_num(seq['ball']/records[train]['pitch'],nan=0).astype(np.float32);original_valid=np.isfinite(seq['ball']).all(1)&(seq['y']<2)
            for c in CONDITIONS:
                print('GRU training features',fi,half,c,flush=True);s,mask=condition(seq,c,half,True);missing=~np.isfinite(s['ball']).all(1)
                if mask is not None:missing=missing|mask
                ball_valid=(original_valid&missing).astype(np.float32)
                for swap in (False,True):seqs.append((observations(s,records[train]['pitch'],mask,swap),np.where(yy>=0,1-yy,-1) if swap else yy,w,t.astype(np.float32),v,ball_target,ball_valid))
        # scaling() reads only each tuple's observation matrix.
        mean,scale=scaling(seqs);seqs=[(normalise(x,mean,scale),y,w,t,v,bt,bv) for x,y,w,t,v,bt,bv in seqs];pos=sum(float((t*v).sum()) for x,y,w,t,v,bt,bv in seqs);neg=sum(float(((1-t)*v).sum()) for x,y,w,t,v,bt,bv in seqs);pos_weight=min(10.,neg/max(pos,1));torch.manual_seed(143);rng=np.random.default_rng(143);model=GatedPossession(cell='gru',layers=2,bidirectional=True,boundary=True,hidden=64,projection=64,dropout=.25,input_features=508);
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(147);aux=torch.nn.Linear(128,2);torch.nn.init.xavier_uniform_(aux.weight);torch.nn.init.zeros_(aux.bias)
        all_parameters=list(model.parameters())+list(aux.parameters());opt=torch.optim.AdamW(all_parameters,lr=.0005,weight_decay=.001);trace=[]
        for epoch in range(1,25):
            model.train();aux.train();losses=[];control_losses=[];ball_losses=[];norms=[];start=time.monotonic()
            for step in range(16):
                windows=[]
                for _ in range(32):
                    x,y,w,t,v,bt,bv=seqs[rng.integers(len(seqs))];a=int(rng.integers(len(x)));windows.append(weighted_window(x,y,w,t,v,a,bt,bv))
                xx,yy,ww,tt,vv,bt,bv=[torch.from_numpy(np.stack([z[k] for z in windows])) for k in range(7)];opt.zero_grad(set_to_none=True);loss,control_loss,ball_loss=auxiliary_losses(model,aux,xx,yy,ww,tt,vv,bt,bv,pos_weight)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(all_parameters,1.);opt.step();losses.append(float(loss.detach()));control_losses.append(float(control_loss.detach()));ball_losses.append(float(ball_loss.detach()));norms.append(float(norm))
            row={'iteration':epoch,'mean_loss':float(np.mean(losses)),'mean_control_boundary_loss':float(np.mean(control_losses)),'mean_masked_ball_mse':float(np.mean(ball_losses)),'mean_preclip_gradient_norm':float(np.mean(norms)),'seconds':time.monotonic()-start};trace.append(row);(fd/'learning.json').write_text(json.dumps(trace,indent=2)+'\n');print('GRU',fi,row,flush=True)
        save_checkpoint(fd/'gru.npz',model,mean,scale,12);np.savez_compressed(fd/'ball-auxiliary-head.npz',weight=aux.weight.detach().numpy(),bias=aux.bias.detach().numpy());gated=load_checkpoint(fd/'gru.npz');old_gru=load_checkpoint(Path(roots['motion_gru_root'])/f'fold-{fi}/gru.npz');reference={'base':bundle,'gru':old_gru,'mix':.5};del seqs,opt,model,aux,all_parameters;rows=[];base=[];preds={k:[] for k in names}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('GRU validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);_,b=predict_variants(bundle,gated,s,records[val]['pitch'],mask);a,_=reference_infer(reference,s,records[val]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});base.append(a)
                for k in names:preds[k].append(b[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=a,**b)
        f={'train':train,'validation':val,'learning':trace,'checkpoint_sha256':digest(fd/'gru.npz'),'reference_hashes':{k:digest(p) for k,p in paths.items()},'current':evaluate_accuracy(rows,base),'candidates':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base
        for k in names:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for name,mix in zip(names,MIXES):
        score=evaluate_accuracy(all_rows,all_preds[name],all_base);guards=eligible(folds,name,current,score);trials.append({'name':name,'gru_mix':mix,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'checkpoint_hashes':[digest(out/f'fold-{i}/gru.npz') for i in (0,1)],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            bundle,_=reference_bundle(fi,roots,control_root);gated=load_checkpoint(out/f'fold-{fi}/gru.npz');reference={'base':bundle,'gru':load_checkpoint(Path(roots['motion_gru_root'])/f'fold-{fi}/gru.npz'),'mix':.5}
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('GRU frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),2914+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),3114+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,3518+half)
                    # Compute only the frozen selected blend, retaining exact common evidence.
                    a,p=infer(bundle,s,records[mid]['pitch'],mask);g,_=symmetric(gated['model'],s,records[mid]['pitch'],gated['mean'],gated['scale'],mask);g=g[:,:2]/np.maximum(g[:,:2].sum(1,keepdims=True),1e-12);x=state_features(s,records[mid]['pitch'],mask);aa=tree_model.predict(bundle['models']['control'],x);bb=tree_model.predict(bundle['models']['control'],swap_state_columns(x));tree=(aa+bb[:,[1,0,2]])[:,:2]*.5;mix=selected['gru_mix'];live=(1-mix)*(p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12))+mix*g;d=p[:,2];q=np.column_stack([live*(1-d[:,None]),d]);event=probabilities(bundle['models']['boundary'],boundary_observations(s,records[mid]['pitch'],q,mask))[:,1];a_reference,_=reference_infer(reference,s,records[mid]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});base.append(a_reference);preds.append(adaptive_decode(q,event,1.))
        audit={'current':evaluate_accuracy(rows,base),'candidate':evaluate_accuracy(rows,preds,base),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','roots','control-root','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.control_root,a.out)
if __name__=='__main__':main()
