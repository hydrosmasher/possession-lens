"""060: learned identity-free player association before the frozen GRU core."""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from .player_set_state import player_nodes,transform_nodes
from .player_set_rnn import warm_start,node_window,predict as set_predict,save_checkpoint as save_set,load_checkpoint as load_set
from .gated_rnn import load_checkpoint,save_checkpoint,predict,CORE
from .joint_state import joint_features,swap_joint
from .spatial_symmetry import augmented_window
from .short_spell_training import open_entry,training_anchor
from .motion_gru import loss_value
from .research_snapshot import best_descriptor,make_snapshot,cached_case
from .ranked_pipeline import infer as base_infer
from .robust_dead_state import robust_dead_features,blend_dead
from .deadball import observation_features
from .boundary_had import boundary_observations,adaptive_decode
from .model import probabilities
from .boundary_alignment import eligible
from .control_state import evaluate_accuracy
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .pipeline import digest
from . import tree_model
BRANCHES=('matched_ce','player_set')

def infer_variants(snapshot,models,s,pitch,mask=None):
    _,base_p=base_infer(snapshot['base'],s,pitch,mask);h=base_p[:,:2]/np.maximum(base_p[:,:2].sum(1,keepdims=True),1e-12);x=joint_features(s,pitch,mask);sx=swap_joint(x);nodes=None;dead=None
    if 'dead_refinement' in snapshot:
        d=snapshot['descriptor'];f=robust_dead_features(s,pitch,mask) if d['dead_features']==487 else observation_features(s,pitch,mask);dead=tree_model.predict(snapshot['dead_refinement'],f)[:,1]
    labels={};emissions={}
    for name,b in models.items():
        if b.get('player_set'):
            if nodes is None:nodes=player_nodes(s,pitch,mask)
            a,_=set_predict(b['model'],x,nodes,b['mean'],b['scale']);z,_=set_predict(b['model'],sx,transform_nodes(nodes,1),b['mean'],b['scale'])
        else:a,_=predict(b['model'],x,b['mean'],b['scale']);z,_=predict(b['model'],sx,b['mean'],b['scale'])
        g=(a+z[:,[1,0,2]])*.5;g=g[:,:2]/np.maximum(g[:,:2].sum(1,keepdims=True),1e-12);live=.5*h+.5*g;d=base_p[:,2];q=np.column_stack([live*(1-d[:,None]),d])
        if dead is not None:q=blend_dead(q,dead,snapshot['descriptor']['dead_mix'])
        event=probabilities(snapshot['base']['models']['boundary'],boundary_observations(s,pitch,q,mask))[:,1];labels[name]=adaptive_decode(q,event,1.);emissions[name]=q
    return labels,emissions

def run(cache_root,roots,run_root,corpus_root,out):
    run_root=Path(run_root).resolve();corpus_root=Path(corpus_root).resolve();out=Path(out);descriptor=best_descriptor(run_root)
    if descriptor['mean_spell_f1']>=.7:print('Verified previous candidate reaches0.70; no new experiment.',flush=True);return
    if out.exists():raise ValueError('Choose new output directory')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);protocol={'version':'player-set-gru-060','reference':descriptor,'hypothesis':'Fixed teamwise summaries lose which raw player/ball motion cues belong together. A shared small player encoder learns association before the existing GRU; compare matched ordinary continuation.',
    'architecture':'Old508feature64projection/2bidirectional64unitGRU retained. Each team has16nearest valid nodes (padded/masked);20physical relative-motion/contact-history/centroid-relative features plus validity. Shared20->32->32MLP+learned masked attention,32pooled perteam ->64projection added to old embedding. New projection starts0; old weights copied exactly. No names/identities/labels/timestamps. Fixed physical scaling,translation-invariant node features. Inference averages team swaps.',
    'training':'16fixed iterations x512windows,batch32,AdamWlr.0001/weightdecay.001,clip1,seed160. Same053half-short anchors/2xshortCEweights and auxiliary boundaryBCE.1. Same four corruption conditions,team swaps/coherent pitch reflections. Locked0.1per-node training dropout and0.25new-embedding dropout; old dropout.25 retained. No new scalers or early stopping.',
    'sampling':'Only fold-trainingWMX orWN1; opposite match validation, both extensively reused. Node feature caches are separate float32 memory-mapped arrays and not source artifacts. No weak-game inputs or new downloads.',
    'selection':'Fixed50percent newGRU+50percent040conditional live; current verified dead expert/031head/decoder unchanged. Original mean spellF1 gain>=.003,per-fold clean/gap decline<=.01,noise no decline,deadF1 decline<=.005,aggregateFPnonincrease. Highest eligible mean.',
    'audit':'Freeze/hash one winner before observation seeds10514/10714/11118. Winner/reference only. Mean improves+FPnonincrease orretainreference without runner-up tuning.',
    'history':'Development on two heavily reused first-division Bundesliga matches, not independent tests. No challenge inputs/video/labels/publicscore used. Target0.70is not a promise.'};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');manifest=json.loads((corpus_root/'manifest.json').read_text());records={}
    for mid in ('J03WMX','J03WN1'):
        files=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(files)!=1:raise ValueError('Ambiguous external cache')
        with np.load(files[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
    node_root=out/'nodes';node_root.mkdir();folds=[];all_rows=[];all_base=[];all_preds={k:[] for k in BRANCHES}
    for fi,(train,val) in enumerate([('J03WMX','J03WN1'),('J03WN1','J03WMX')]):
        print('FOLD',fi,train,val,flush=True);fd=out/f'fold-{fi}';fd.mkdir();snapshot=make_snapshot(descriptor,fi,roots,run_root);old=snapshot['gru']
        if old['model'].config['input_features']!=508:raise ValueError('Set-encoder warm-start requires508feature reference')
        strong=[]
        for entry in manifest['entries']:
            if entry['source']!='strong' or entry['match_id']!=train:continue
            e=open_entry(corpus_root,entry);half=entry['half']-1;seq=period_arrays(records[train],half);s,mask=condition(seq,entry['condition'],half,True);p=node_root/(Path(entry['features']).stem+'-nodes.npy');print('Node cache',train,half,entry['condition'],flush=True);np.save(p,player_nodes(s,records[train]['pitch'],mask),allow_pickle=False);e['nodes']=np.load(p,mmap_mode='r',allow_pickle=False);strong.append(e)
        if len(strong)!=8:raise ValueError('Wrong external training fold')
        mean,scale=old['mean'],old['scale'];pos=sum(float((e['t']*e['v']).sum()) for e in strong);neg=sum(float(((1-e['t'])*e['v']).sum()) for e in strong);pw=min(10.,neg/max(pos,1));models={'reference':old};learning={}
        for name in BRANCHES:
            m=warm_start(old) if name=='player_set' else load_checkpoint(run_root/descriptor['control_source']/f'fold-{fi}'/(descriptor['control_name']+'.npz'))['model'];torch.manual_seed(160);rng=np.random.default_rng(160);opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.001);trace=[]
            for iteration in range(1,17):
                start=time.monotonic();m.train();losses=[];norms=[]
                for _ in range(16):
                    windows=[];node_windows=[]
                    for _ in range(32):
                        e=strong[int(rng.integers(len(strong)))];anchor=training_anchor(e,rng,True);a=max(0,anchor-int(rng.integers(CORE)));code=int(rng.integers(8));windows.append(augmented_window(e['x'],e['y'],e['focus_w'],e['t'],e['v'],a,mean,scale,code=code,spatial=True))
                        if name=='player_set':node_windows.append(transform_nodes(node_window(e['nodes'],a),code))
                    xx,yy,ww,tt,vv=[torch.from_numpy(np.stack([z[k] for z in windows])) for k in range(5)];opt.zero_grad(set_to_none=True);logits,head=m(xx,torch.from_numpy(np.stack(node_windows))) if name=='player_set' else m(xx);loss=loss_value(logits,yy,ww,head,tt,vv,pw)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite set-RNN loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
                row={'iteration':iteration,'mean_loss':float(np.mean(losses)),'mean_preclip_gradient_norm':float(np.mean(norms)),'seconds':time.monotonic()-start};trace.append(row);print(name,fi,row,flush=True);(fd/(name+'-learning.json')).write_text(json.dumps(trace,indent=2)+'\n')
            if name=='player_set':save_set(fd/(name+'.npz'),m,mean,scale);models[name]=load_set(fd/(name+'.npz'))
            else:save_checkpoint(fd/(name+'.npz'),m,mean,scale,12);models[name]=load_checkpoint(fd/(name+'.npz'))
            learning[name]=trace;del m,opt
        del strong;rows=[];base_preds=[];preds={k:[] for k in BRANCHES}
        for half in (0,1):
            seq=period_arrays(records[val],half)
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fi,half,c,flush=True);s,mask=condition(seq,c,half,False);p,q=infer_variants(snapshot,models,{k:v for k,v in s.items() if k!='y'},records[val]['pitch'],mask);y,expected,_=cached_case(descriptor,fi,half*4+ci,run_root);np.testing.assert_array_equal(y,seq['y']);np.testing.assert_array_equal(p['reference'],expected);rows.append({'y':seq['y'],'condition':c});base_preds.append(expected)
                for k in BRANCHES:preds[k].append(p[k])
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=y,current=expected,**{k:p[k] for k in BRANCHES},**{'emissions_'+k:v for k,v in q.items()})
        f={'train':train,'validation':val,'learning':learning,'current':evaluate_accuracy(rows,base_preds),'candidates':{k:evaluate_accuracy(rows,p,base_preds) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');all_rows+=rows;all_base+=base_preds
        for k in BRANCHES:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);np.testing.assert_allclose(current['mean_spell_f1'],descriptor['mean_spell_f1'],atol=1e-14,rtol=0);trials=[]
    for name in BRANCHES:
        score=evaluate_accuracy(all_rows,all_preds[name],all_base);guards=eligible(folds,name,current,score);guards['false_positive_no_increase']=score['totals']['false_positive']<=current['totals']['false_positive'];trials.append({'name':name,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'reference':descriptor,'checkpoint_hashes':[{k:digest(out/f'fold-{fi}'/(k+'.npz')) for k in BRANCHES} for fi in (0,1)],'before_new_draw_audit':True},indent=2)+'\n');print('FROZEN',selected,flush=True);audit=None
    if selected:
        rows=[];base_preds=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            snapshot=make_snapshot(descriptor,fi,roots,run_root);model=load_set(out/f'fold-{fi}/player_set.npz') if selected['name']=='player_set' else load_checkpoint(out/f'fold-{fi}/matched_ce.npz');models={'reference':snapshot['gru'],selected['name']:model}
            for half in (0,1):
                seq=period_arrays(records[mid],half)
                for ci,c in enumerate(CONDITIONS):
                    print('Frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),10514+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),10714+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,11118+half)
                    p,q=infer_variants(snapshot,models,s,records[mid]['pitch'],mask);rows.append({'y':seq['y'],'condition':c});base_preds.append(p['reference']);preds.append(p[selected['name']]);np.savez_compressed(out/f'fold-{fi}'/f'audit-{half*4+ci}.npz',current=p['reference'],candidate=p[selected['name']])
        a=evaluate_accuracy(rows,base_preds);b=evaluate_accuracy(rows,preds,base_preds);audit={'current':a,'candidate':b,'mean_improves':b['mean_spell_f1']>a['mean_spell_f1'],'false_positive_no_increase':b['totals']['false_positive']<=a['totals']['false_positive'],'same_matches_new_noise_only':True}
    report={'protocol':protocol,'reference':descriptor,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves'] and audit['false_positive_no_increase']),'promoted_to_default':False,'independent_test':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials:print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['guards'],flush=True)
    if audit:print('AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],audit['false_positive_no_increase'],flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache-root','roots','run-root','corpus-root','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,json.loads(a.roots.read_text()),a.run_root,a.corpus_root,a.out)
