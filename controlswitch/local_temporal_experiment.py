"""069: local temporal evidence for short possessions; external-only study."""
from pathlib import Path
import json,time
import numpy as np,torch
from torch.nn import functional as F
from .local_temporal import LocalTemporal,save,load,symmetric,locked_dead_decode
from .short_spell_training import open_entry,training_anchor
from .spatial_symmetry import augmented_window
from .gated_rnn import load_checkpoint,CORE
from .research_snapshot import cached_case
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition
from .control_state import evaluate_accuracy
from .pipeline import pm,digest


def short_matches(y,p,max_frames=75):
    runs=pm.valid_runs(y);gt=pm.possession_spells(y,runs);flags=pm.matched_gt_flags(pm.spell_overlaps(gt,pm.possession_spells(p,runs)),len(gt),.5)
    return sum(bool(h) for h,(a,b,k) in zip(flags,gt) if b-a<=max_frames)


def run(root,cache,corpus):
    root=Path(root).resolve();corpus=Path(corpus).resolve();out=root/'local/local-temporal-069';out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol={'version':'local-temporal-069','diagnosis':'068findsonly142of1201missedshortspells recovered byremovingH/Asmoothing;weak/fragmentedtimeevidence dominates. Do not lowerglobalpenalty.',
      'architecture':'508->64linear/LayerNorm/GELU;3residual64channel5tapconvolutions,dilations1/4/16,LayerNorm/GELU/lockeddropout.2;2classhead. Finite42frame(1.68s) learned radius on top offixed observation features. Offline futurecontext,notliveforecasting.',
      'fitting':'Fromscratch,20fixediterationsx512windows,batch32;AdamWlr.0003/weightdecay.001/gradientclip1/seed169+fold. Samecorresponding053training-only508scaler. Fittingmatchstrongcorpusonly,all4trainingcorruptions,halfshortanchors<=3s,2xshortCEweights,coherentteam/pitchreflections. No boundaryloss or earlystopping. Existinglabels aretargets/sampling only.',
      'folds':[{'fit':'J03WMX','evaluate':'J03WN1'},{'fit':'J03WN1','evaluate':'J03WMX'}],
      'variants':['neutral_blend','local_expert'],
      'frozen':'053emissions/031boundaryevent/Dmask remainreference. NewconditionalHA=.75reference+.25expert;neutralcomparison uses.5/.5expert. Locked-Ddecoderusesexistingper-frameadaptivecost12andcachedboundaryevents;referenceexactreplayasserted. No globalcostchange,nominimumduration,newDbridgesorchangedhalfboundaries.',
      'selection':'Onlylocal_expert:mean spellF1 gain>=.002;matchedshort<=3sspellsstrictincrease;totalmisses/FPnonincrease;eachnoise foldnondecline;eachclean/masked/longgapfolddecline<=.01;meanmustexceedneutralcontrol. Dmaskexact. Freezeweights beforeaudit.',
      'audit':'Selectedwinneronlyvs053,samegamesnewdrawseeds10518/10718/11122;mustimprovemean+shortmatches,misses/FPnonincrease,Dmaskexact. No tuning/runnerupsafteraudit. These are notnewmatches.',
      'corpus_manifest_sha256':digest(corpus/'manifest.json'),'source_sha256':{k:digest(root/'controlswitch'/k) for k in ('local_temporal.py','local_temporal_experiment.py')},
      'history':'Twoheavilyreusedfirst-divisionBundesligagames;developmentonly. Nochallengeinputs/video/labels/publicscore,newdownloadsorverifiedreceiptclaims.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');manifest=json.loads((corpus/'manifest.json').read_text());names=('neutral_blend','local_expert');allrows=[];allbase=[];allpred={k:[] for k in names};folds=[]
    for fi,(train,val) in enumerate((('J03WMX','J03WN1'),('J03WN1','J03WMX'))):
      fd=out/f'fold-{fi}';fd.mkdir();entries=[open_entry(corpus,e) for e in manifest['entries'] if e['source']=='strong' and e['match_id']==train]
      if len(entries)!=8:raise ValueError('Wrong training fold')
      old=load_checkpoint(root/f'local/short-spell-053/fold-{fi}/short_focus.npz');mean,scale=old['mean'],old['scale'];del old;torch.manual_seed(169+fi);rng=np.random.default_rng(169+fi);m=LocalTemporal();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001);trace=[]
      for iteration in range(1,21):
        start=time.monotonic();m.train();losses=[];norms=[]
        for _ in range(16):
          windows=[]
          for _ in range(32):
            e=entries[int(rng.integers(len(entries)))];anchor=training_anchor(e,rng,True);a=max(0,anchor-int(rng.integers(CORE)));code=int(rng.integers(8));windows.append(augmented_window(e['x'],e['y'],e['focus_w'],e['t'],e['v'],a,mean,scale,code=code,spatial=True))
          xx,yy,ww=[torch.from_numpy(np.stack([z[k] for z in windows])) for k in range(3)];opt.zero_grad(set_to_none=True);logits=m(xx);valid=yy>=0;losses_all=F.cross_entropy(logits.flatten(0,1),yy.clamp_min(0).flatten(),reduction='none').reshape_as(yy);weight=ww*valid;loss=(losses_all*weight).sum()/weight.sum().clamp_min(1)
          if not torch.isfinite(loss):raise ValueError('Nonfinite temporal objective')
          loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
        row={'iteration':iteration,'training_ce':float(np.mean(losses)),'preclip_gradient_norm':float(np.mean(norms)),'seconds':time.monotonic()-start};trace.append(row);print('Train local expert',fi,row,flush=True);(fd/'learning.json').write_text(json.dumps(trace,indent=2)+'\n')
      save(fd/'local-expert.npz',m,mean,scale);bundle=load(fd/'local-expert.npz');del entries,m,opt
      files=list(Path(cache).glob(val+'-*.npz'))
      if len(files)!=1:raise ValueError('Ambiguous external cache')
      with np.load(files[0],allow_pickle=False) as a:record={k:a[k] for k in a.files}
      rows=[];base=[];preds={k:[] for k in names}
      for half in (0,1):
        seq=period_arrays(record,half)
        for ci,kind in enumerate(CONDITIONS):
          print('Evaluate local expert',fi,half,kind,flush=True);s,mask=condition(seq,kind,half,False);g=symmetric(bundle,s,record['pitch'],mask);y,current,q=cached_case({'source':'short-spell-053','name':'short_focus'},fi,half*4+ci,root/'local');np.testing.assert_array_equal(y,seq['y'])
          with np.load(root/f'local/precision-055-cache/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as a:event=a['event'].copy()
          h=q[:,:2]/np.maximum(q[:,:2].sum(1,keepdims=True),1e-12);np.testing.assert_array_equal(locked_dead_decode(q[:,:2],event,current),current);local={}
          for name,expert in [('neutral_blend',np.full_like(g,.5)),('local_expert',g)]:
            p=locked_dead_decode(.75*h+.25*expert,event,current);np.testing.assert_array_equal(p==2,current==2);preds[name].append(p);local[name]=p
          rows.append({'y':y,'condition':kind});base.append(current);np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=y,current=current,expert=g,**local)
      f={'train':train,'evaluate':val,'learning':trace,'model_sha256':digest(fd/'local-expert.npz'),'reference':evaluate_accuracy(rows,base),'variants':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');allrows+=rows;allbase+=base
      for k in names:allpred[k]+=preds[k]
    reference=evaluate_accuracy(allrows,allbase);short=sum(short_matches(r['y'],p) for r,p in zip(allrows,allbase));scores={k:evaluate_accuracy(allrows,p,allbase) for k,p in allpred.items()};trials=[]
    for name,p in allpred.items():
      score=scores[name];sh=sum(short_matches(r['y'],v) for r,v in zip(allrows,p));guards={'spell_gain':score['mean_spell_f1']>=reference['mean_spell_f1']+.002,'short_recovery':sh>short,'misses':score['totals']['missed']<=reference['totals']['missed'],'false_positives':score['totals']['false_positive']<=reference['totals']['false_positive'],'noise':all(f['variants'][name]['tracker_noise']['spell_f1_mean']>=f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),'clean_gap':all(f['variants'][name][c]['spell_f1_mean']>=f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'beats_neutral':score['mean_spell_f1']>scores['neutral_blend']['mean_spell_f1']};trials.append({'name':name,'scores':score,'short_matched':sh,'guards':guards,'eligible':name=='local_expert' and all(guards.values())})
    selected=next((t for t in trials if t['eligible']),None);report={'protocol':protocol,'folds':folds,'reference':reference,'reference_short_matched':short,'trials':trials,'selected':selected,'frozen_audit_required':bool(selected),'promoted_to_default':False,'independent_test':False};(out/'freeze.json').write_text(json.dumps({'selected':selected,'models':[f['model_sha256'] for f in folds],'before_new_draw_audit':True},indent=2)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials:print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],'short',t['short_matched'],t['guards'],flush=True)
    return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--cache-root',type=Path);parser.add_argument('--corpus-root',type=Path);a=parser.parse_args();run(a.root,a.cache_root or a.root/'data/external/cache',a.corpus_root or a.root/'local/transfer-corpus-052')
