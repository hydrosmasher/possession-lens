"""External-only, partially labelled SkillCorner experiment. Never accepts challenge inputs.

Provider possession is weak supervision, not independent human ground truth.
Unknowns are excluded, never converted to dead ball or bridged across gaps.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from .skillcorner_legacy import SPLIT
from .model import instantaneous_features,decode
from . import rnn_deep as net
from .rnn_benchmark import augmented_sequences,conditions,evaluate
from .rnn_capacity import observation_probabilities
from .rnn_runtime import load_frozen,infer_frozen
from .experiment import baseline_prediction


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def interpolate(a,positions):
    """Interpolate only two adjacent observed native samples; no gap filling."""
    lo=np.floor(positions).astype(int);hi=np.minimum(lo+1,len(a)-1)
    w=positions-lo;w=w.reshape((-1,)+(1,)*(a.ndim-1))
    out=a[lo]*(1-w)+a[hi]*w
    exact=np.isclose(positions,lo);out[exact]=a[lo[exact]]
    return out.astype(np.float32)

def load_match(root,mid,recover_provider_ids=False,id_gap_frames=10,include_owner=False):
    p=Path(root)/mid;m=json.loads((p/'match_data.json').read_text());frames=json.loads((p/'structured_data.json').read_text())
    players=[x for x in m['players'] if x.get('trackable_object') is not None]
    ids={x['trackable_object']:i for i,x in enumerate(players)}
    if recover_provider_ids:
        from .control_recovery import recover_ids
        recover_ids(frames,{x['trackable_object']:0 if x['team_id']==m['home_team']['id'] else 1 for x in players},id_gap_frames)
    team=np.array([0 if x['team_id']==m['home_team']['id'] else 1 for x in players]+[0]*22+[1]*22,np.int8)
    parts=[];audit=[]
    for period in (1,2):
        ff=[f for f in frames if f['period']==period]
        if not ff:raise ValueError('Missing playing period')
        first=ff[0]['frame'];last=ff[-1]['frame'];n=last-first+1
        if n>150000 or any(b['frame']<=a['frame'] for a,b in zip(ff,ff[1:])):raise ValueError('Invalid frame timeline')
        xy=np.full((n,len(team),2),np.nan,np.float32);ball=np.full((n,2),np.nan,np.float32);y=np.full(n,3,np.int8)
        slots={};previous=set();anonymous=0;identified=0;ball_id=m['ball']['trackable_object'];owner_native=np.full(n,-1,np.int32) if include_owner else None
        for f in ff:
            j=f['frame']-first;y[j]={'home team':0,'away team':1}.get((f.get('possession') or {}).get('group'),3)
            current=set()
            for obj in f['data']:
                tid=obj.get('trackable_object');pos=[obj['x'],obj['y']]
                if tid==ball_id or obj.get('group_name')=='ball':ball[j]=pos
                elif tid in ids:xy[j,ids[tid]]=pos;identified+=1
                elif obj.get('group_name') in ('home team','away team','home goalkeeper','away goalkeeper'):
                    t=0 if obj['group_name'].startswith('home') else 1;key=(t,obj['track_id']);current.add(key);anonymous+=1
                    if key not in slots:
                        used={v for k,v in slots.items() if k[0]==t and (k in previous or k in current)}
                        free=[len(players)+22*t+i for i in range(22) if len(players)+22*t+i not in used]
                        if not free:raise ValueError('Anonymous pool overflow')
                        slot=free[0]
                        slots={k:v for k,v in slots.items() if v!=slot};slots[key]=slot
                        # Leave a missing observation on assignment to break false slot velocity.
                        continue
                    xy[j,slots[key]]=pos
            if include_owner:
                owner_id=(f.get('possession') or {}).get('trackable_object');slot=ids.get(owner_id)
                if slot is not None and y[j]<2 and team[slot]==y[j] and np.isfinite(xy[j,slot]).all() and np.isfinite(ball[j]).all() and sum(obj.get('trackable_object')==owner_id for obj in f['data'])==1:owner_native[j]=slot
            previous=current
        # Native 10 Hz -> common 25 Hz for compatibility with frozen baseline/RNN.
        positions=np.arange(int(np.floor((n-1)*2.5))+1)/2.5
        seq={'xy':interpolate(xy,positions),'ball':interpolate(ball,positions),'teams':team,'y':y[np.floor(positions+1e-9).astype(int)],'period':np.array(period)}
        if include_owner:seq['owner']=owner_native[np.floor(positions+1e-9).astype(int)]
        for source,target in [('xy','speed'),('ball','ball_speed')]:
            a=seq[source];v=np.full(a.shape[:-1],np.nan,np.float32);v[1:]=np.linalg.norm(a[1:]-a[:-1],axis=-1)*25;seq[target]=v
        # Discard impossible velocities rather than injecting spikes into either comparator.
        seq['speed'][seq['speed']>15]=np.nan;seq['ball_speed'][seq['ball_speed']>60]=np.nan
        parts.append((seq,np.array([m['pitch_length'],m['pitch_width']],np.float32)))
        valid=y<2;starts=np.flatnonzero(valid & ~np.r_[False,valid[:-1]]);ends=np.flatnonzero(valid & ~np.r_[valid[1:],False])+1
        audit.append({'period':period,'native_frames':n,'labelled_native_frames':int(valid.sum()),'label_coverage':float(valid.mean()),'valid_runs':len(starts),'median_valid_run_seconds':float(np.median(ends-starts)/10),'direct_labelled_team_changes':int(np.sum((y[1:]<2)&(y[:-1]<2)&(y[1:]!=y[:-1]))),'identified_observations':identified,'anonymous_observations':anonymous,'ball_observed_fraction':float(np.isfinite(ball).all(1).mean())})
        if include_owner:audit[-1]['owner_labelled_native_frames']=int((owner_native>=0).sum());audit[-1]['owner_policy']='Exact provider ID, current-frame player+ball observed, group/team consistent; unknown excluded'
    return parts,audit


def run(root,out,reference,iterations=12):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen.json').exists():raise ValueError('Frozen experiment already exists; no post-test refit')
    protocol={'split':SPLIT,'hidden_sizes':[[24],[24,24]],'max_iterations':iterations,'windows_per_iteration':1024,'dropout':.15,'l2':.0001,'seed':42,'decoder_penalties':[3,12,24,48],'checkpoint_selection':'best mean clean/masked validation spell F1 across iteration/depth/penalty','reference_sha256':digest(reference),'evaluation':'conditional provider-labelled H/A spells; NOT full-match independent ground truth','unknown_policy':'X, no dead labels, gaps not bridged','native_fps':10,'common_fps':25,'resampling':'adjacent finite tracking linear interpolation; causal label hold; no labels in features','source':'SkillCorner MIT pinned original release','test_read_before_freeze':False,'competition_data_used':False}
    (out/'training-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    training=[];validation=[];audits={}
    for mid in SPLIT['train']+SPLIT['validation']:
        parts,audits[mid]=load_match(root,mid)
        (training if mid in SPLIT['train'] else validation).extend(parts)
    (out/'adapter-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    sequences=augmented_sequences(training);val=[(instantaneous_features(s,p),np.where(s['y']<2,s['y'],-1)) for s,p in validation]
    sequences=[(x,np.where(y<2,y,-1)) for x,y in sequences]
    rows=conditions(validation,100);best={'score':-1};trials=[]
    # on_epoch accesses an explicit snapshot supplied by the training API.
    for sizes in [(24,),(24,24)]:
        def progress(row,model):
            probs=[observation_probabilities(model,s,p,mask) for s,p,c,mask in rows]
            for penalty in protocol['decoder_penalties']:
                scores=evaluate(rows,[decode(p,penalty) for p in probs]);score=scores['selection_spell_f1']
                trials.append({'layers':len(sizes),'iteration':row['iteration'],'penalty':penalty,'score':score,'train_ce':row['train_probe_cross_entropy'],'val_ce':row['validation_probe_cross_entropy']})
                if score>best['score']:
                    best.update(score=score,layers=len(sizes),iteration=row['iteration'],penalty=penalty)
                    np.savez_compressed(out/'candidate-model.npz',**model,version=np.array(net.VERSION),transition_penalty=np.array(float(penalty)),symmetric=np.array(True))
            (out/'validation-trials.json').write_text(json.dumps(trials,indent=2)+'\n')
            print('Validation best',json.dumps(best),flush=True)
        net.train(sequences,val,sizes,epochs=iterations,patience=iterations,on_epoch_model=progress)
    baseline_trials=[]
    for w in [0,25,50,125]:baseline_trials.append({'window':w,**evaluate(rows,[baseline_prediction(s,w,mask) for s,p,c,mask in rows])})
    base=max(baseline_trials,key=lambda x:x['selection_spell_f1'])
    frozen={**best,'candidate_sha256':digest(out/'candidate-model.npz'),'reference_sha256':digest(reference),'baseline_window':base['window'],'adapter_sha256':digest(__file__),'test_labels_read':False}
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    # Only now open the sealed tracking contents and possession labels.
    test,audit=load_match(root,SPLIT['test'][0]);rows=conditions(test,300)
    result={}
    for name,path in [('candidate',out/'candidate-model.npz'),('current',reference)]:
        state=load_frozen(path);result[name]=evaluate(rows,[infer_frozen(state,s,p,mask)[0] for s,p,c,mask in rows])
    result['baseline']=evaluate(rows,[baseline_prediction(s,base['window'],mask) for s,p,c,mask in rows])
    report={'protocol':protocol,'frozen':frozen,'test_audit':audit,'test_results':result,'candidate_beats_current':result['candidate']['selection_spell_f1']>result['current']['selection_spell_f1'],'promoted_to_default':False,'limitations':['Automatically generated provider labels; this is a weak-supervision transfer experiment.','Only observed H/A-labelled runs are scored; no supervised dead-ball state.','Unknown gaps fragment spells and can inflate F1: these scores do not establish the official challenge target.','Training checkpoint selected repeatedly on one validation match; one test match is limited evidence.','Interpolated 25 Hz frames are not independent observations.','Offline decoder uses future evidence; not live forecasting.','Existing current comparator has second-division training ancestry; new candidate trained from scratch only on selected top-five-league matches.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'test_results':result,'candidate_beats_current':report['candidate_beats_current']},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--iterations',type=int,default=12);a=p.parse_args();run(a.root,a.out,a.reference,a.iterations)
