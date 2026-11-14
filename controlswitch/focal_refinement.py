"""046: matched extra-CE versus focal-loss continuation from frozen043."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from .narrow_boundary import bundle040,narrow_targets
from .joint_state import joint_features,swap_joint
from .spell_training import spell_weights
from .motion_gru import weighted_window
from .gated_rnn import load_checkpoint,save_checkpoint,normalise,predict
from .ranked_pipeline import infer as base_infer
from .boundary_had import boundary_observations,adaptive_decode
from .model import probabilities
from .boundary_alignment import eligible
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest

KINDS=('extra_ce','focal2')
def refinement_loss(logits,y,w,head,t,v,pos_weight,kind):
    if kind not in KINDS:raise ValueError('Unknown continuation loss')
    known=y>=0
    if known.any():
        ce=F.cross_entropy(logits[known],y[known],reduction='none');factor=(1-torch.exp(-ce))**2 if kind=='focal2' else torch.ones_like(ce);control=(ce*factor*w[known]).sum()/w[known].sum().clamp_min(1)
    else:control=logits.sum()*0
    event=F.binary_cross_entropy_with_logits(head,t,reduction='none',pos_weight=head.new_tensor(pos_weight));return control+.1*(event*v).sum()/v.sum().clamp_min(1)

def infer_models(base,grus,s,pitch,mask=None):
    _,p=base_infer(base,s,pitch,mask);h=p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12);x=joint_features(s,pitch,mask);sx=swap_joint(x);d=p[:,2];out={}
    for name,m in grus.items():
        a,_=predict(m['model'],x,m['mean'],m['scale']);b,_=predict(m['model'],sx,m['mean'],m['scale']);g=(a+b[:,[1,0,2]])*.5;g=g[:,:2]/np.maximum(g[:,:2].sum(1,keepdims=True),1e-12);live=.5*h+.5*g;q=np.column_stack([live*(1-d[:,None]),d]);event=probabilities(base['models']['boundary'],boundary_observations(s,pitch,q,mask))[:,1];out[name]=adaptive_decode(q,event,1.)
    return out

def run(cache_root,roots,run_root,out):
    run_root=Path(run_root);out=Path(out)
    if out.exists():raise ValueError('Choose a new output folder')
    prereq=json.loads((run_root/'motion-gru-043/report.json').read_text())
    if not prereq['recommended_for_freeze']:raise ValueError('Expected passed043 audit')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);protocol={'version':'focal-continuation-046','reference':'frozen04350percentGRU','hypothesis':'Focus extra recurrent training on hard labelled frames, with matched ordinary-CE continuation as the control.',
    'initialisation':'Each branch starts from the same corresponding043final checkpoint/scaler, never validation weights. Both use seed146 for identical windows/dropout draws.',
    'training':'8fixed iterations x512windows,batch32,core256/context64,AdamWlr.0001/weightdecay.001,clip1. Same508features,spell weights,four augmented conditions/team swaps. CE or focalgamma2; same auxiliary boundary BCE.1 and narrow+-2target/known-live+-12valid neighbourhood.',
    'variants':list(KINDS),'selection':'Only50percentGRU+50percent040live fusion. Mean official four-condition spellF1 gain>=.003 over043; clean/gap fold decline<=.01,noise fold no decline,deadF1 decline<=.005. Highest eligible mean; FP reported.',
    'audit':'Freeze/hash one winner before new draws seeds3014/3214/3618. Winner/043 only. Audit mean must improve or retain043 without runner-up selection.',
    'history':'Two heavily reused first-division Bundesliga matches, not independent tests. No challenge data/video/labels/public score used. No percentile claim.'};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
    folds=[];all_rows=[];all_base=[];all_preds={k:[] for k in KINDS}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();base,paths=bundle040(fi,roots,run_root/'joint-040');checkpoint=run_root/f'motion-gru-043/fold-{fi}/gru.npz';old=load_checkpoint(checkpoint);mean,scale=old['mean'],old['scale'];seqs=[]
        for half in (0,1):
            seq=period_arrays(records[train],half);y=np.where(seq['y']<2,seq['y'],-1);w=spell_weights(seq['y']).astype(np.float32);t,v=narrow_targets(seq['y'],2)
            for c in CONDITIONS:
                print('Features',fi,half,c,flush=True);s,mask=condition(seq,c,half,True);x=joint_features(s,records[train]['pitch'],mask)
                for swap in (False,True):seqs.append((normalise(swap_joint(x) if swap else x,mean,scale),np.where(y>=0,1-y,-1) if swap else y,w,t.astype(np.float32),v.astype(np.float32)))
        pos=sum(float((t*v).sum()) for x,y,w,t,v in seqs);neg=sum(float(((1-t)*v).sum()) for x,y,w,t,v in seqs);pw=min(10.,neg/max(pos,1));grus={'reference':old};traces={}
        for kind in KINDS:
            m=load_checkpoint(checkpoint)['model'];torch.manual_seed(146);rng=np.random.default_rng(146);opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.001);trace=[]
            for epoch in range(1,9):
                m.train();losses=[];norms=[];start=time.monotonic()
                for step in range(16):
                    windows=[]
                    for _ in range(32):
                        x,y,w,t,v=seqs[rng.integers(len(seqs))];a=int(rng.integers(len(x)));windows.append(weighted_window(x,y,w,t,v,a))
                    xx,yy,ww,tt,vv=[torch.from_numpy(np.stack([z[k] for z in windows])) for k in range(5)];opt.zero_grad(set_to_none=True);logits,head=m(xx);loss=refinement_loss(logits,yy,ww,head,tt,vv,pw,kind)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite continuation loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
                row={'iteration':epoch,'mean_loss':float(np.mean(losses)),'mean_preclip_gradient_norm':float(np.mean(norms)),'seconds':time.monotonic()-start};trace.append(row);print(kind,fi,row,flush=True);(fd/(kind+'-learning.json')).write_text(json.dumps(trace,indent=2)+'\n')
            save_checkpoint(fd/(kind+'.npz'),m,mean,scale,12);grus[kind]=load_checkpoint(fd/(kind+'.npz'));traces[kind]=trace;del m,opt
        del seqs;rows=[];current=[];preds={k:[] for k in KINDS}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);p=infer_models(base,grus,s,records[val]['pitch'],mask)
                with np.load(run_root/f'motion-gru-043/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as a:np.testing.assert_array_equal(p['reference'],a['gru0.5'])
                rows.append({'y':seq['y'],'condition':c});current.append(p['reference'])
                for k in KINDS:preds[k].append(p[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=p['reference'],**{k:p[k] for k in KINDS})
        f={'train':train,'validation':val,'initial_checkpoint_sha256':digest(checkpoint),'learning':traces,'current':evaluate_accuracy(rows,current),'candidates':{k:evaluate_accuracy(rows,p,current) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=current
        for k in KINDS:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for kind in KINDS:
        score=evaluate_accuracy(all_rows,all_preds[kind],all_base);guards=eligible(folds,kind,current,score);trials.append({'name':kind,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'checkpoint_hashes':[{k:digest(out/f'fold-{i}'/(k+'.npz')) for k in KINDS} for i in (0,1)],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];current=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            base,_=bundle040(fi,roots,run_root/'joint-040');grus={'reference':load_checkpoint(run_root/f'motion-gru-043/fold-{fi}/gru.npz'),selected['name']:load_checkpoint(out/f'fold-{fi}'/(selected['name']+'.npz'))}
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),3014+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),3214+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,3618+half)
                    p=infer_models(base,grus,s,records[mid]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});current.append(p['reference']);preds.append(p[selected['name']])
        audit={'current':evaluate_accuracy(rows,current),'candidate':evaluate_accuracy(rows,preds,current),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','run-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.run_root,a.out)
if __name__=='__main__':main()
