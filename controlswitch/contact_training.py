"""057: matched recurrent continuation with per-player contact persistence."""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from .contact_state import contact_features,swap_contact
from .joint_state import joint_features,swap_joint
from .gated_rnn import GatedPossession,load_checkpoint,save_checkpoint,normalise,predict,CORE,CONTEXT
from .spatial_symmetry import augmented_window,reflection_signs
from .short_spell_training import open_entry,training_anchor
from .narrow_boundary import bundle040
from .motion_gru import loss_value
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

BRANCHES=['matched508','contact572']

def warm_contact(old,extra_mean,extra_scale):
    config={**old['model'].config,'input_features':572};m=GatedPossession(**config);state=old['model'].state_dict();new=m.state_dict()
    for k in new:
        if k=='embedding.0.weight':new[k].zero_();new[k][:,:508].copy_(state[k])
        else:new[k].copy_(state[k])
    m.load_state_dict(new);return m,np.r_[old['mean'],extra_mean].astype(np.float32),np.r_[old['scale'],extra_scale].astype(np.float32)

def contact_window(e,start,mean,scale,code):
    n=len(e['x']);a=max(0,start-CONTEXT);b=min(n,start+CORE+CONTEXT);offset=a-(start-CONTEXT);raw=np.column_stack([e['x'][a:b],e['extra'][a:b]])
    if code&1:raw=swap_contact(raw)
    raw=raw*np.r_[reflection_signs(bool(code&2),bool(code&4)),np.ones(64,np.float32)]
    xx=np.zeros((CORE+2*CONTEXT,572),np.float32);xx[offset:offset+b-a]=normalise(raw,mean,scale);yy=np.full(len(xx),-1,np.int64);ww=np.zeros(len(xx),np.float32);tt=ww.copy();vv=ww.copy();size=min(CORE,n-start);sl=slice(CONTEXT,CONTEXT+size);yy[sl]=e['y'][start:start+size];ww[sl]=e['focus_w'][start:start+size];tt[sl]=e['t'][start:start+size];vv[sl]=e['v'][start:start+size]
    if code&1:yy=np.where(yy>=0,1-yy,-1)
    ww[yy<0]=0;return xx,yy,ww,tt,vv

def extra_scaling(entries):
    count=0;total=np.zeros(64,np.float64);squares=total.copy();order=np.r_[np.arange(32,64),np.arange(32)]
    for e in entries:
        x=e['extra'];s=x.sum(0,dtype=np.float64);q=np.einsum('ij,ij->j',x.astype(np.float64),x.astype(np.float64));total+=s+s[order];squares+=q+q[order];count+=2*len(x)
    mean=total/count;scale=np.maximum(np.sqrt(np.maximum(0,squares/count-mean*mean)),.05);return mean.astype(np.float32),scale.astype(np.float32)

def infer_models(base,grus,s,pitch,mask=None):
    _,p=base_infer(base,s,pitch,mask);h=p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12);x=joint_features(s,pitch,mask);sx=swap_joint(x);extended=None;out={};emissions={}
    for name,m in grus.items():
        if m['model'].config['input_features']==572:
            if extended is None:extended=np.column_stack([x,contact_features(s,mask)])
            a,b=extended,swap_contact(extended)
        else:a,b=x,sx
        pa,_=predict(m['model'],a,m['mean'],m['scale']);pb,_=predict(m['model'],b,m['mean'],m['scale']);g=(pa+pb[:,[1,0,2]])*.5;g=g[:,:2]/np.maximum(g[:,:2].sum(1,keepdims=True),1e-12);live=.5*h+.5*g;d=p[:,2];q=np.column_stack([live*(1-d[:,None]),d]);event=probabilities(base['models']['boundary'],boundary_observations(s,pitch,q,mask))[:,1];out[name]=adaptive_decode(q,event,1.);emissions[name]=q
    return out,emissions

def run(cache_root,roots,run_root,corpus_root,out):
    run_root=Path(run_root);corpus_root=Path(corpus_root).resolve();out=Path(out)
    if out.exists():raise ValueError('Choose new output directory')
    assert json.loads((run_root/'short-spell-053/report.json').read_text())['recommended_for_freeze'];manifest=json.loads((corpus_root/'manifest.json').read_text());out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol={'version':'contact-persistence-057','reference':'frozen053','hypothesis':'Teamwise independent maxima lose per-player ball association. Add64 scalar per-player contact/persistence features pooled without identities; contrast matched continuation of old508features.',
    'branches':BRANCHES,'training':'Both start corresponding053short_focus weights. Contact572 copies all old parameters and zero-initialises64 new embedding columns; extra mean/scale fitted to training-only observations plus team swaps.16fixed iterations x512windows,batch32,AdamWlr.0001,weightdecay.001,clip1,seed157. Same two64unit bidirectional GRU layers,projection64,dropout.25. No validation early stopping.',
    'sampling':'Same053half-short<=3s anchors and2xexistingCEweights onshortspells, spatial reflection/team swaps, four original training conditions. Auxiliary narrow boundaryBCE.1 unchanged. ForwardWMXonly/reverseWN1only. No weak-game windows.',
    'features':'508old plus32perteam: top1/top2/top3means of8 per-player proximity,coupling,interception,trajectory-change,persistence,alignment scores;6 same-player weighted past/future memories,acceleration and visibility. Scalars invariant to pitch reflection; no labels,identities,timestamps. Current ball missing =>extra contact evidence0. Velocities use+-2frames,turn evidence+-4frames; fixed plausible-speed masks match existing feature recipe.',
    'selection':'Same50percent newGRU+50percent040conditional live,dead/quality/031boundary unchanged. Mean four-condition spellF1 gain>=.003 vs053; per-fold clean/gap decline<=.01,noise no decline,deadF1 decline<=.005; aggregate false positives must not increase. Highest eligible original mean.',
    'audit':'Freeze/hash one winner before new observation draws7514/7714/8118. Only winner/053audited;mean must improve andFPmustnotincrease,else retain053 without runner-up retuning.',
    'history':'Two heavily reused external first-division Bundesliga games, not independent matches. No challenge inputs/video/labels/public scores used. Goal.70 not guaranteed. No new downloads.'};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        files=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(files)!=1:raise ValueError('Ambiguous external cache')
        with np.load(files[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
    extra_root=out/'extra';extra_root.mkdir();folds=[];all_rows=[];all_base=[];all_preds={k:[] for k in BRANCHES}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();strong=[]
        for entry in manifest['entries']:
            if entry['source']!='strong' or entry['match_id']!=train:continue
            e=open_entry(corpus_root,entry);half=entry['half']-1;seq=period_arrays(records[train],half);s,mask=condition(seq,entry['condition'],half,True);p=extra_root/(Path(entry['features']).stem+'-contact.npy');np.save(p,contact_features(s,mask),allow_pickle=False);e['extra']=np.load(p,mmap_mode='r',allow_pickle=False);strong.append(e)
        assert len(strong)==8 and all(e['match_id']==train for e in strong);extra_mean,extra_scale=extra_scaling(strong);np.savez_compressed(fd/'extra-scaler.npz',mean=extra_mean,scale=extra_scale);checkpoint=run_root/f'short-spell-053/fold-{fi}/short_focus.npz';old=load_checkpoint(checkpoint);base,paths=bundle040(fi,roots,run_root/'joint-040');grus={'reference':old};learning={};pos=sum(float((e['t']*e['v']).sum()) for e in strong);neg=sum(float(((1-e['t'])*e['v']).sum()) for e in strong);pw=min(10.,neg/max(pos,1))
        for name in BRANCHES:
            if name=='contact572':m,mean,scale=warm_contact(old,extra_mean,extra_scale)
            else:m=load_checkpoint(checkpoint)['model'];mean,scale=old['mean'],old['scale']
            torch.manual_seed(157);rng=np.random.default_rng(157);opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.001);trace=[]
            for iteration in range(1,17):
                start=time.monotonic();m.train();losses=[];norms=[]
                for _ in range(16):
                    windows=[]
                    for _ in range(32):
                        e=strong[int(rng.integers(len(strong)))];anchor=training_anchor(e,rng,True);a=max(0,anchor-int(rng.integers(CORE)));code=int(rng.integers(8));windows.append(contact_window(e,a,mean,scale,code) if name=='contact572' else augmented_window(e['x'],e['y'],e['focus_w'],e['t'],e['v'],a,mean,scale,code=code,spatial=True))
                    xx,yy,ww,tt,vv=[torch.from_numpy(np.stack([z[k] for z in windows])) for k in range(5)];opt.zero_grad(set_to_none=True);logits,head=m(xx);loss=loss_value(logits,yy,ww,head,tt,vv,pw)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
                row={'iteration':iteration,'mean_loss':float(np.mean(losses)),'mean_preclip_gradient_norm':float(np.mean(norms)),'seconds':time.monotonic()-start};trace.append(row);print(name,fi,row,flush=True);(fd/(name+'-learning.json')).write_text(json.dumps(trace,indent=2)+'\n')
            save_checkpoint(fd/(name+'.npz'),m,mean,scale,12);grus[name]=load_checkpoint(fd/(name+'.npz'));learning[name]=trace;del m,opt
        del strong;rows=[];base_preds=[];preds={k:[] for k in BRANCHES}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);p,q=infer_models(base,grus,{k:v for k,v in s.items() if k!='y'},records[val]['pitch'],mask)
                with np.load(run_root/f'short-spell-053/fold-{fi}/validation-{half*4+ci}.npz') as a:np.testing.assert_array_equal(p['reference'],a['short_focus'])
                rows.append({'y':seq['y'],'condition':c});base_preds.append(p['reference'])
                for k in BRANCHES:preds[k].append(p[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=p['reference'],**{k:p[k] for k in BRANCHES},**{'emissions_'+k:v for k,v in q.items()})
        f={'train':train,'validation':val,'training_checkpoint_sha256':digest(checkpoint),'extra_scaler_sha256':digest(fd/'extra-scaler.npz'),'learning':learning,'current':evaluate_accuracy(rows,base_preds),'candidates':{k:evaluate_accuracy(rows,p,base_preds) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base_preds
        for k in BRANCHES:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for name in BRANCHES:
        score=evaluate_accuracy(all_rows,all_preds[name],all_base);guards=eligible(folds,name,current,score);guards['false_positive_no_increase']=score['totals']['false_positive']<=current['totals']['false_positive'];trials.append({'name':name,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'checkpoint_hashes':[{k:digest(out/f'fold-{fi}'/(k+'.npz')) for k in BRANCHES} for fi in (0,1)],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base_preds=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            base,_=bundle040(fi,roots,run_root/'joint-040');grus={'reference':load_checkpoint(run_root/f'short-spell-053/fold-{fi}/short_focus.npz'),selected['name']:load_checkpoint(out/f'fold-{fi}'/(selected['name']+'.npz'))}
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for ci,c in enumerate(CONDITIONS):
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),7514+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),7714+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,8118+half)
                    p,q=infer_models(base,grus,s,records[mid]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});base_preds.append(p['reference']);preds.append(p[selected['name']]);np.savez_compressed(out/f'fold-{fi}'/f'audit-{half*4+ci}.npz',current=p['reference'],candidate=p[selected['name']])
        a=evaluate_accuracy(rows,base_preds);b=evaluate_accuracy(rows,preds,base_preds);audit={'current':a,'candidate':b,'mean_improves':b['mean_spell_f1']>a['mean_spell_f1'],'false_positive_no_increase':b['totals']['false_positive']<=a['totals']['false_positive'],'same_matches_new_noise_only':True}
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves'] and audit['false_positive_no_increase']),'promoted_to_default':False,'independent_test':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials:print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],audit['false_positive_no_increase'],flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','run-root','corpus-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.run_root,a.corpus_root,a.out)
