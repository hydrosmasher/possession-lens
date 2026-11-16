"""058: matched CE versus local spell-IoU continuation, external data only."""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from .contact_training import contact_window,infer_models
from .short_spell_training import open_entry,training_anchor
from .spatial_symmetry import augmented_window
from .gated_rnn import load_checkpoint,save_checkpoint,CORE
from .motion_gru import loss_value
from .spell_iou_loss import local_spell_iou
from .narrow_boundary import bundle040
from .boundary_alignment import eligible
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest
BRANCHES=('matched_ce','spell_iou')

def run(cache_root,roots,run_root,corpus_root,out):
    run_root=Path(run_root);corpus_root=Path(corpus_root).resolve();out=Path(out);previous=json.loads((run_root/'contact-057/report.json').read_text());source='contact-057' if previous['recommended_for_freeze'] else 'short-spell-053';initial=previous if source=='contact-057' else json.loads((run_root/'short-spell-053/report.json').read_text());initial_name=initial['selected']['name']
    if initial['selected']['scores']['mean_spell_f1']>=.7:print('Verified preceding candidate reaches0.70; no new experiment started.',flush=True);return
    if out.exists():raise ValueError('Choose new output directory')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=json.loads((corpus_root/'manifest.json').read_text())
    protocol={'version':'local-spell-overlap-058','reference':source+'/'+initial_name,'hypothesis':'Frame CE does not directly penalise possession-boundary overhang or fragmentation. Add a differentiable local spell-IoU surrogate, with a matched continuation control.',
    'initialisation':'Use the best verified057winner if it passed its frozen audit; otherwise053. If preceding candidate alreadyreaches0.70,do not start. Preserve architecture/features/scaler exactly. Each branch starts the corresponding external training-fold checkpoint.',
    'branches':list(BRANCHES),'training':'16fixed iterations x512windows,batch32,AdamWlr.0001,weightdecay.001,clip1,seed158; same053half-short anchor sampling/2xshortCEweights,spatial/teamaugmentation and narrow boundaryBCE.1. Only spell_iou adds0.25*local_soft_IoU_loss. No early stopping or parameter search.',
    'surrogate':'Conditional H/A probabilities only. Equally average complete known-live GTspells inside loss core with opposite-team known-live neighbours. Intersection is correct-class soft mass inside spell; union is GTlength plus correct-class mass outside spell up to neighbouring-spell midpoints. Context,padding,D/unknown and edge-truncated spells excluded. This is a surrogate, not exact differentiable official greedy matching. Labels only training targets,never inputs.',
    'selection':'Same50percentGRU+50percent040live,dead/quality/boundary unchanged. Original four-condition mean gain>=.003 vsreference; clean/gap fold decline<=.01,noise no decline,deadF1 decline<=.005,aggregateFPnoincrease. Highest eligible mean.',
    'audit':'Freeze one winner/hashes before new draws8514/8714/9118. Only winner/reference audited. Audit mean must improve andFPmustnotincrease;otherwise retainreference,no runner-up retuning.',
    'history':'Existing two heavily reused first-division Bundesliga matches. ForwardWMX/reverseWN1 fitting,opposite-match development; not independent testing. No challenge data/video/labels/public scores or newdownloads.'};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records={}
    for mid in ('J03WMX','J03WN1'):
        files=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(files)!=1:raise ValueError('Ambiguous external cache')
        with np.load(files[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
    folds=[];all_rows=[];all_base=[];all_preds={k:[] for k in BRANCHES}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();checkpoint=run_root/source/f'fold-{fi}'/(initial_name+'.npz');old=load_checkpoint(checkpoint);width=old['model'].config['input_features'];strong=[]
        for entry in manifest['entries']:
            if entry['source']!='strong' or entry['match_id']!=train:continue
            e=open_entry(corpus_root,entry)
            if width==572:e['extra']=np.load(run_root/'contact-057/extra'/(Path(entry['features']).stem+'-contact.npy'),mmap_mode='r',allow_pickle=False)
            strong.append(e)
        if len(strong)!=8 or any(e['match_id']!=train for e in strong):raise ValueError('Wrong training fold')
        base,_=bundle040(fi,roots,run_root/'joint-040');mean,scale=old['mean'],old['scale'];pos=sum(float((e['t']*e['v']).sum()) for e in strong);neg=sum(float(((1-e['t'])*e['v']).sum()) for e in strong);pw=min(10.,neg/max(pos,1));grus={'reference':old};learning={}
        for name in BRANCHES:
            m=load_checkpoint(checkpoint)['model'];torch.manual_seed(158);rng=np.random.default_rng(158);opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.001);trace=[]
            for iteration in range(1,17):
                start=time.monotonic();m.train();losses=[];overlaps=[];norms=[]
                for _ in range(16):
                    windows=[]
                    for _ in range(32):
                        e=strong[int(rng.integers(len(strong)))];anchor=training_anchor(e,rng,True);a=max(0,anchor-int(rng.integers(CORE)));code=int(rng.integers(8));windows.append(contact_window(e,a,mean,scale,code) if width==572 else augmented_window(e['x'],e['y'],e['focus_w'],e['t'],e['v'],a,mean,scale,code=code,spatial=True))
                    xx,yy,ww,tt,vv=[torch.from_numpy(np.stack([z[k] for z in windows])) for k in range(5)];opt.zero_grad(set_to_none=True);logits,head=m(xx);base_loss=loss_value(logits,yy,ww,head,tt,vv,pw);overlap=local_spell_iou(logits,yy) if name=='spell_iou' else logits.sum()*0;loss=base_loss+.25*overlap
                    if not torch.isfinite(loss):raise ValueError('Nonfinite overlap training loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(base_loss.detach()));overlaps.append(float(overlap.detach()));norms.append(float(norm))
                row={'iteration':iteration,'mean_base_loss':float(np.mean(losses)),'mean_overlap_loss':float(np.mean(overlaps)),'mean_preclip_gradient_norm':float(np.mean(norms)),'seconds':time.monotonic()-start};trace.append(row);print(name,fi,row,flush=True);(fd/(name+'-learning.json')).write_text(json.dumps(trace,indent=2)+'\n')
            save_checkpoint(fd/(name+'.npz'),m,mean,scale,12);grus[name]=load_checkpoint(fd/(name+'.npz'));learning[name]=trace;del m,opt
        del strong;rows=[];base_preds=[];preds={k:[] for k in BRANCHES}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);p,q=infer_models(base,grus,{k:v for k,v in s.items() if k!='y'},records[val]['pitch'],mask)
                with np.load(run_root/source/f'fold-{fi}/validation-{half*4+ci}.npz') as a:np.testing.assert_array_equal(p['reference'],a[initial_name])
                rows.append({'y':seq['y'],'condition':c});base_preds.append(p['reference'])
                for k in BRANCHES:preds[k].append(p[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=seq['y'],current=p['reference'],**{k:p[k] for k in BRANCHES},**{'emissions_'+k:v for k,v in q.items()})
        f={'train':train,'validation':val,'initial_checkpoint_sha256':digest(checkpoint),'learning':learning,'current':evaluate_accuracy(rows,base_preds),'candidates':{k:evaluate_accuracy(rows,p,base_preds) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base_preds
        for k in BRANCHES:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);np.testing.assert_allclose(current['mean_spell_f1'],initial['selected']['scores']['mean_spell_f1'],atol=1e-14,rtol=0);trials=[]
    for name in BRANCHES:
        score=evaluate_accuracy(all_rows,all_preds[name],all_base);guards=eligible(folds,name,current,score);guards['false_positive_no_increase']=score['totals']['false_positive']<=current['totals']['false_positive'];trials.append({'name':name,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'initial_source':source,'initial_name':initial_name,'checkpoint_hashes':[{k:digest(out/f'fold-{fi}'/(k+'.npz')) for k in BRANCHES} for fi in (0,1)],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base_preds=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            base,_=bundle040(fi,roots,run_root/'joint-040');grus={'reference':load_checkpoint(run_root/source/f'fold-{fi}'/(initial_name+'.npz')),selected['name']:load_checkpoint(out/f'fold-{fi}'/(selected['name']+'.npz'))}
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for ci,c in enumerate(CONDITIONS):
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),8514+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),8714+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,9118+half)
                    p,q=infer_models(base,grus,s,records[mid]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});base_preds.append(p['reference']);preds.append(p[selected['name']]);np.savez_compressed(out/f'fold-{fi}'/f'audit-{half*4+ci}.npz',current=p['reference'],candidate=p[selected['name']])
        a=evaluate_accuracy(rows,base_preds);b=evaluate_accuracy(rows,preds,base_preds);audit={'current':a,'candidate':b,'mean_improves':b['mean_spell_f1']>a['mean_spell_f1'],'false_positive_no_increase':b['totals']['false_positive']<=a['totals']['false_positive'],'same_matches_new_noise_only':True}
    report={'protocol':protocol,'initial_source':source,'initial_name':initial_name,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves'] and audit['false_positive_no_increase']),'promoted_to_default':False,'independent_test':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials:print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],audit['false_positive_no_increase'],flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','run-root','corpus-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.run_root,a.corpus_root,a.out)
