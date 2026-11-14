"""052: memory-mapped spatial and weak-team transfer from frozen043."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from .transfer_corpus import WEAK_IDS
from .spatial_symmetry import augmented_window
from .narrow_boundary import bundle040
from .gated_rnn import load_checkpoint,save_checkpoint,CORE
from .motion_gru import loss_value
from .focal_refinement import infer_models
from .boundary_alignment import eligible
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest

BRANCHES=[{'name':'strong_more','weak_fraction':0.,'spatial':False},{'name':'spatial_more','weak_fraction':0.,'spatial':True},{'name':'weak_transfer','weak_fraction':.25,'spatial':False},{'name':'spatial_transfer','weak_fraction':.25,'spatial':True}]

def open_entry(root,e):
    xp=(root/e['features']).resolve();yp=(root/e['labels']).resolve()
    if not xp.is_relative_to(root) or not yp.is_relative_to(root):raise ValueError('Escaping corpus path')
    x=np.load(xp,mmap_mode='r',allow_pickle=False)
    if x.shape!=(e['rows'],508) or x.dtype!=np.float32:raise ValueError('Unexpected corpus features')
    with np.load(yp,allow_pickle=False) as a:y,w,t,v=(a[k].copy() for k in ('y','w','t','v'))
    known=np.flatnonzero(y>=0)
    if not len(known):raise ValueError('Corpus sequence has no H/A supervision')
    return {**e,'x':x,'y':y,'w':w,'t':t,'v':v,'known':known}

def run(cache_root,roots,run_root,corpus_root,out):
    out=Path(out);run_root=Path(run_root);corpus_root=Path(corpus_root).resolve()
    if out.exists():raise ValueError('Choose new output directory')
    corpus=json.loads((corpus_root/'manifest.json').read_text());prior=json.loads((run_root/'motion-gru-043/report.json').read_text())
    if not corpus['complete'] or not prior['recommended_for_freeze']:raise ValueError('Need complete external corpus and passed043 audit')
    if set(corpus['protocol']['weak_matches'])!=set(WEAK_IDS):raise ValueError('Unexpected weak source pool')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out.mkdir(parents=True);protocol={'version':'disk-backed-transfer-052','reference':'frozen043','hypothesis':'Add existing first-division SkillCorner team labels to the current GRU without treating weak ownership as live/dead truth; compare spatial reflections and matched strong-only continuation.',
    'branches':BRANCHES,'training':'Each branch starts from corresponding043weights/scaler.16fixed iterations x512windows,batch32,AdamWlr.0001/weightdecay.001,clip1,seed152. Fixed508layout/64unit2layerbidirectional architecture. Rawfloat32 corpus read by memory-mapped window; normalisation/padding identical to043. No whole-corpus RAM load.',
    'sampling':'Strong pool contains only the fold training match. Weak pool contains only4039/3749/3518/2068/2440. Draw known-frame-anchored cores with random within-core offset. Team swap in all branches; spatial branches add coherent x/y reflections, verified against raw geometry. Same draws within matched branches.',
    'labels':'Strong H/A spell-weighted CE and narrow boundaryBCE.1. Weak H/A known-frame weight.5, unknown-1; weak boundary loss disabled. No weak dead labels. Normaliser remains043training-only. Window source probabilities are distinct from independent label counts.',
    'selection':'Only50percent newGRU+50percent040live fusion. Official four-condition mean spellF1 gain>=.003 over043; clean/gap per-fold decline<=.01,noise fold no decline,deadF1 decline<=.005. Highest eligible mean; FP reported.',
    'audit':'Freeze/hash one winner before different observation draws seeds3514/3714/4118. Winner/043 only. Audit mean must improve or retain043, no runner-up selection.',
    'history':'All seven corpus matches previously viewed; strong validation games extensively reused. Weak2269/2841/3442/2417 excluded from fitting. No independent-test,0.7orpercentile claim. No challenge observations/video/labels/public score used.'};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        fs=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(fs)!=1:raise ValueError('Ambiguous validation cache')
        with np.load(fs[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong validation match')
    weak=[open_entry(corpus_root,e) for e in corpus['entries'] if e['source']=='weak' and e['match_id'] in WEAK_IDS];names=[b['name'] for b in BRANCHES];folds=[];all_rows=[];all_base=[];all_preds={k:[] for k in names}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();strong=[open_entry(corpus_root,e) for e in corpus['entries'] if e['source']=='strong' and e['match_id']==train]
        if len(strong)!=8 or any(e['match_id']!=train for e in strong):raise ValueError('Training fold contamination')
        base,paths=bundle040(fi,roots,run_root/'joint-040');checkpoint=run_root/f'motion-gru-043/fold-{fi}/gru.npz';old=load_checkpoint(checkpoint);mean,scale=old['mean'],old['scale'];pos=sum(float((e['t']*e['v']).sum()) for e in strong);neg=sum(float(((1-e['t'])*e['v']).sum()) for e in strong);pw=min(10.,neg/max(pos,1));grus={'reference':old};learning=[]
        for branch in BRANCHES:
            name=branch['name'];m=load_checkpoint(checkpoint)['model'];torch.manual_seed(152);rng=np.random.default_rng(152);opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.001);trace=[];source_counts={}
            for epoch in range(1,17):
                m.train();losses=[];norms=[];start=time.monotonic()
                for step in range(16):
                    windows=[]
                    for _ in range(32):
                        u=rng.random();wi=int(rng.integers(len(weak)));si=int(rng.integers(len(strong)));e=weak[wi] if u<branch['weak_fraction'] else strong[si];anchor=int(e['known'][rng.integers(len(e['known']))]);a=max(0,anchor-int(rng.integers(CORE)));code=int(rng.integers(8));windows.append(augmented_window(e['x'],e['y'],e['w'],e['t'],e['v'],a,mean,scale,code=code,spatial=branch['spatial']));source_counts[e['match_id']]=source_counts.get(e['match_id'],0)+1
                    xx,yy,ww,tt,vv=[torch.from_numpy(np.stack([z[k] for z in windows])) for k in range(5)];opt.zero_grad(set_to_none=True);logits,head=m(xx);loss=loss_value(logits,yy,ww,head,tt,vv,pw)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite transfer loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
                row={'iteration':epoch,'mean_loss':float(np.mean(losses)),'mean_preclip_gradient_norm':float(np.mean(norms)),'seconds':time.monotonic()-start};trace.append(row);print(name,fi,row,flush=True);(fd/(name+'-learning.json')).write_text(json.dumps({'trace':trace,'sampled_windows_by_match':source_counts},indent=2)+'\n')
            save_checkpoint(fd/(name+'.npz'),m,mean,scale,12);grus[name]=load_checkpoint(fd/(name+'.npz'));learning.append({'branch':branch,'trace':trace,'sampled_windows_by_match':source_counts});del m,opt
        del strong;rows=[];current=[];preds={k:[] for k in names}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);p=infer_models(base,grus,s,records[val]['pitch'],mask)
                with np.load(run_root/f'motion-gru-043/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as ref:np.testing.assert_array_equal(p['reference'],ref['gru0.5'])
                rows.append({'y':seq['y'],'condition':c});current.append(p['reference'])
                for k in names:preds[k].append(p[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=p['reference'],**{k:p[k] for k in names})
        f={'train':train,'validation':val,'weak_training_ids':list(WEAK_IDS),'initial_checkpoint_sha256':digest(checkpoint),'learning':learning,'current':evaluate_accuracy(rows,current),'candidates':{k:evaluate_accuracy(rows,p,current) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=current
        for k in names:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);trials=[]
    for branch in BRANCHES:
        score=evaluate_accuracy(all_rows,all_preds[branch['name']],all_base);guards=eligible(folds,branch['name'],current,score);trials.append({**branch,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'checkpoint_hashes':[{k:digest(out/f'fold-{i}'/(k+'.npz')) for k in names} for i in (0,1)],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];current=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            base,_=bundle040(fi,roots,run_root/'joint-040');grus={'reference':load_checkpoint(run_root/f'motion-gru-043/fold-{fi}/gru.npz'),selected['name']:load_checkpoint(out/f'fold-{fi}'/(selected['name']+'.npz'))}
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for c in CONDITIONS:
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),3514+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),3714+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,4118+half)
                    p=infer_models(base,grus,s,records[mid]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});current.append(p['reference']);preds.append(p[selected['name']])
        audit={'current':evaluate_accuracy(rows,current),'candidate':evaluate_accuracy(rows,preds,current),'same_matches_new_noise_only':True};audit['mean_improves']=audit['candidate']['mean_spell_f1']>audit['current']['mean_spell_f1']
    report={'protocol':protocol,'corpus_summary':{'feature_bytes':corpus['feature_bytes'],'previously_viewed_matches':True,'source_audit':corpus['source_audit']},'folds':folds,'current':evaluate_accuracy(all_rows,all_base),'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',report['current']['mean_spell_f1'],report['current']['totals'],flush=True)
    for t in trials:print(t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','run-root','corpus-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.run_root,a.corpus_root,a.out)
if __name__=='__main__':main()
