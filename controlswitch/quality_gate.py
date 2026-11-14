"""029: externally trained observation-quality gating of frozen dead experts."""
from pathlib import Path
import argparse,json
import numpy as np
from .model import shifted,rolling,decode
from .external import period_arrays
from .tracker_robustness import CONDITIONS,condition,evaluate
from .deadball import observation_features,fuse
from .pipeline import digest
from . import tree_model

VERSION='observation-quality-gate-029'

def quality_features(s,mask=None):
    ball=s['ball'].copy();bs=s['ball_speed'].copy()
    if mask is not None:ball[mask]=np.nan;bs[mask]=np.nan
    xy=s['xy'];speed=s['speed'];seen=np.isfinite(ball).all(1);valid=np.isfinite(xy).all(2)
    bv=(ball-shifted(ball,-1))*25;pv=(xy-shifted(xy,-1))*25
    bgood=np.isfinite(bv).all(1)&np.isfinite(bs);pgood=np.isfinite(pv).all(2)&np.isfinite(speed)
    bdiff=np.abs(np.log1p(np.linalg.norm(bv,axis=1))-np.log1p(np.clip(bs,0,100)))
    pdiff=np.abs(np.log1p(np.linalg.norm(pv,axis=2))-np.log1p(np.clip(speed,0,100)))
    bj=np.linalg.norm(bv-shifted(bv,-1),axis=1)*25;pj=np.linalg.norm(pv-shifted(pv,-1),axis=2)*25
    def mean_valid(values,good):return np.where(good,values,0).sum(1)/np.maximum(good.sum(1),1)
    duplicate=[]
    for team in (0,1):
        pos=xy[:,s['teams']==team];best=np.zeros(len(ball))
        for i in range(pos.shape[1]):
            for j in range(i):
                d=np.sum((pos[:,i]-pos[:,j])**2,axis=1);best=np.maximum(best,np.exp(-np.nan_to_num(d,nan=np.inf)/(.25**2)))
        duplicate.append(best)
    raw=np.column_stack([seen,np.where(bgood,bdiff,0),np.nan_to_num(np.log1p(np.minimum(bj,1000)),nan=0),pgood.mean(1),mean_valid(pdiff,pgood),mean_valid(np.log1p(np.minimum(pj,1000)),np.isfinite(pj)),valid.sum(1)/22,np.minimum(*duplicate),np.maximum(*duplicate)]).astype(np.float32)
    return np.column_stack([raw,rolling(raw,12),rolling(raw,50)])

def gated_dead(old,new,quality,cap):
    if cap not in (.75,1.):raise ValueError('Unsupported frozen gate cap')
    old=np.asarray(old);new=np.asarray(new);quality=np.asarray(quality)
    if old.shape!=new.shape or old.shape!=quality.shape or old.ndim!=1 or not np.isfinite(old).all() or not np.isfinite(new).all() or not np.isfinite(quality).all() or ((old<0)|(old>1)|(new<0)|(new>1)|(quality<0)|(quality>1)).any():raise ValueError('Invalid gate probabilities')
    mix=.5+(cap-.5)*quality
    a=np.clip(old,1e-12,1-1e-12);b=np.clip(new,1e-12,1-1e-12)
    logit=(1-mix)*np.log(a/(1-a))+mix*np.log(b/(1-b))
    return 1/(1+np.exp(-np.clip(logit,-80,80)))

def score(rows,preds,current=None):
    r=evaluate(rows,preds,current)
    for c in CONDITIONS:
        ix=[i for i,x in enumerate(rows) if x['condition']==c];r[c]['frame_accuracy']=sum(int((rows[i]['y']==preds[i]).sum()) for i in ix)/sum(len(rows[i]['y']) for i in ix)
    return r

def run(cache_root,old_dead_forward,old_dead_reverse,noise_root,emission_root,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    out=Path(out)
    if out.exists():raise ValueError('Choose a new run folder')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'history':'Adaptive research after accuracy027 on same repeatedly used Bundesliga pair, no independent test','competition_data_used':False,'gate_supervision':'Automatic external corruption condition, not possession or BallStatus labels; tracker_noise=1,clean/short/long=0','features':'27 team-invariant quality features: visibility, stored-vs-position-derived speed mismatch, acceleration, finite player velocities, visible counts and same-team duplicate proximity; raw and rolling12/50. No condition flags, identities, time or reference labels as inputs.','quality_training':'Each opposite training match, both halves and four conditions, every5th frame;50 boosting iterations/7leaves/lr.08/minleaf80/l2=10,seed129,no earlystop','frozen':'019 H/A probabilities; original014/015 and noise018 dead experts unchanged','caps':[.75,1.],'gate':'Noise expert weight=.5+(cap-.5)*P(corruption); clean baseline remains50/50. Expert logits mixed before unchanged deadweight.75 and decoder12/3.','selection':'Highest four-condition spellF1 with no total misses/merges/FP increase; prior fold loss<=.01; noise fold no decline; each deadF1 loss<=.005; clean pooled frame accuracy no decline. Report accuracy;95% is a target, not guaranteed.','folds':[{'train':'J03WMX','validation':'J03WN1'},{'train':'J03WN1','validation':'J03WMX'}],'source':'Existing first-division Bundesliga Sportec/IDSSE CC BY4.0, no new data downloads','hashes':{'old_dead':[digest(old_dead_forward),digest(old_dead_reverse)],'noise_dead':[digest(Path(noise_root)/f'fold-{i}/dead-estimator.npz') for i in (0,1)]}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');folds=[];all_rows=[];all_current=[];all_preds={cap:[] for cap in protocol['caps']}
    for fold,(train,val,old_path) in enumerate([('J03WMX','J03WN1',old_dead_forward),('J03WN1','J03WMX',old_dead_reverse)]):
        print('Fold',fold,train,'->',val,flush=True);fd=out/f'fold-{fold}';fd.mkdir();records={}
        for mid in (train,val):
            files=list(Path(cache_root).glob(mid+'-*.npz'))
            if len(files)!=1:raise ValueError('Ambiguous cache')
            with np.load(files[0],allow_pickle=False) as a:records[mid]={k:a[k] for k in a.files}
            if str(records[mid]['match_id'])!=mid:raise ValueError('Wrong match')
        xs=[];ys=[]
        for half in (0,1):
            original=period_arrays(records[train],half)
            for c in CONDITIONS:
                s,mask=condition(original,c,half,True);x=quality_features(s,mask);xs.append(x[::5]);ys.append(np.full(len(x[::5]),int(c=='tracker_noise'),int))
        x=np.concatenate(xs);y=np.concatenate(ys);head=HistGradientBoostingClassifier(max_iter=50,learning_rate=.08,max_leaf_nodes=7,min_samples_leaf=80,l2_regularization=10,early_stopping=False,random_state=129)
        print('Fitting quality detector',x.shape,flush=True)
        with threadpool_limits(limits=1):head.fit(x,y)
        tree_model.save(fd/'quality-estimator.npz',head,'hist_boost');gate=tree_model.load(fd/'quality-estimator.npz')
        if not np.allclose(head.predict_proba(x[:10000]),tree_model.predict(gate,x[:10000])[:,:2],atol=1e-9):raise AssertionError('Export mismatch')
        old=tree_model.load(old_path);new=tree_model.load(Path(noise_root)/f'fold-{fold}/dead-estimator.npz');rows=[];current=[];preds={cap:[] for cap in protocol['caps']};gate_audit=[]
        for half in (0,1):
            original=period_arrays(records[val],half);pitch=records[val]['pitch']
            for ci,c in enumerate(CONDITIONS):
                print('Validation',fold,half,c,flush=True);s,mask=condition(original,c,half,False);x=quality_features(s,mask);q=tree_model.predict(gate,x)[:,1]
                with threadpool_limits(limits=1):native=head.predict_proba(x)[:,1]
                if not np.allclose(q,native,atol=1e-9):raise AssertionError('Gate export mismatch')
                f=Path(emission_root)/f'fold-{fold}/validation-{half*4+ci}.npz'
                with np.load(f,allow_pickle=False) as a:p=a['emissions'];base=a['current'];reference=a['reference']
                np.testing.assert_array_equal(reference,original['y']);rows.append({'y':reference,'condition':c});current.append(base);live=p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12);h=np.column_stack([live,np.zeros(len(live))]);obs=observation_features(s,pitch,mask);a=tree_model.predict(old,obs)[:,1];b=tree_model.predict(new,obs)[:,1]
                for cap in protocol['caps']:preds[cap].append(decode(fuse(h,gated_dead(a,b,q,cap),.75),12,stop_penalty=3))
                gate_audit.append({'half':half+1,'condition':c,'mean_corruption_score':float(q.mean())})
        r={'train':train,'validation':val,'training_examples':len(y),'gate_sha256':digest(fd/'quality-estimator.npz'),'gate_audit':gate_audit,'current':score(rows,current),'variants':{str(cap):score(rows,p,current) for cap,p in preds.items()},'export_matches_native':True};folds.append(r);(fd/'report.json').write_text(json.dumps(r,indent=2)+'\n');all_rows+=rows;all_current+=current
        for cap in preds:all_preds[cap]+=preds[cap]
        for i,row in enumerate(rows):np.savez_compressed(fd/f'validation-{i}.npz',reference=row['y'],current=current[i],**{'cap'+str(cap):p[i] for cap,p in preds.items()})
    current=score(all_rows,all_current);trials=[]
    for cap in protocol['caps']:
        r=score(all_rows,all_preds[cap],all_current);a=r['totals'];b=current['totals'];guards={'spell_f1':r['mean_spell_f1']>current['mean_spell_f1'],'misses':a['missed']<=b['missed'],'merges':a['merges']<=b['merges'],'false_positives':a['false_positive']<=b['false_positive'],'prior_conditions':all(f['variants'][str(cap)][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'tracker_noise':all(f['variants'][str(cap)]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),'dead_f1':all(f['variants'][str(cap)][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS),'clean_accuracy':r['clean']['frame_accuracy']>=current['clean']['frame_accuracy']}
        trials.append({'cap':cap,'scores':r,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None,'fresh_test_results':None,'promoted_to_default':False,'limitations':['Corruption labels identify a synthetic domain, not the actual challenge error process.','Same two reused matches; no independently validated accuracy or top-percentile claim.','Old and noise experts share provider supervision; gate training uses synthetic condition labels only.','No challenge data, public score, labels or video used.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',current['mean_spell_f1'],current['clean']['frame_accuracy'],current['totals'])
    for t in trials:print('CAP',t['cap'],t['scores']['mean_spell_f1'],[(c,t['scores'][c]['frame_accuracy']) for c in CONDITIONS],t['scores']['totals'],t['guards'])
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','old-dead-forward','old-dead-reverse','noise-root','emission-root','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.cache_root,a.old_dead_forward,a.old_dead_reverse,a.noise_root,a.emission_root,a.out)
if __name__=='__main__':main()
