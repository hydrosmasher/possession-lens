"""Spell recovery on external top-five-league data; sealed test opened after freeze."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from . import rnn_deep as net
from .skillcorner_europe import load_match,digest
from .model import instantaneous_features,boundary_targets,rolling,probabilities,fit_classifier,decode
from .rnn_capacity import observation_probabilities
from .rnn_benchmark import conditions,evaluate
from .rnn_runtime import load_frozen,infer_frozen,DEEP_BOUNDARY_VERSION
from .rnn_boundaries import invariant_boundary_features,spell_diagnostics
from .experiment import missing_mask,baseline_prediction

TRAIN=('4039','3749','3518');VALID='2269';TEST='2417'
PENALTIES=(3,12,24,48);STRENGTHS=(0,.5,1)

def loss_weights(y):
    """Short observed spells matter more; unknown-censored durations are not trusted."""
    y=np.asarray(y);w=np.zeros(len(y),np.float32)
    edges=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
    for a,b in zip(edges[:-1],edges[1:]):
        if y[a]>=2:continue
        censored=a==0 or b==len(y) or y[a-1]>=2 or y[b]>=2
        w[a:b]=1. if censored else np.clip(np.sqrt(250/(b-a)),.5,4)
    w*=1+.5*boundary_targets(y,radius=12)
    return w

def boundary_supervision(y):
    target=boundary_targets(y,radius=12)
    # Entire target/context neighbourhood must have known home/away labels.
    valid=rolling((y>=2).astype(np.float32)[:,None],25)[:,0]==0
    valid[:25]=False;valid[-25:]=False
    return target,valid

def totals(d):
    keys=('missed_reference_spells_at_iou_0_5','false_positive_spells_at_iou_0_5','predictions_overlapping_multiple_reference_spells','predicted_spells_under_0_4_seconds')
    return {k:sum(d[c][k] for c in ('clean','masked')) for k in keys}

def accepted(d,reference):
    a=totals(d);b=totals(reference)
    return a['false_positive_spells_at_iou_0_5']<=1.1*b['false_positive_spells_at_iou_0_5'] and a['predicted_spells_under_0_4_seconds']<=b['predicted_spells_under_0_4_seconds']+5

def save_model(path,model,penalty,head=None,strength=0):
    extra={} if head is None else {**{'event_'+k:v for k,v in head.items()},'boundary_strength':np.array(strength),'min_frames':np.array(1)}
    np.savez_compressed(path,**model,version=np.array(net.VERSION if head is None else DEEP_BOUNDARY_VERSION),transition_penalty=np.array(float(penalty)),symmetric=np.array(True),**extra)

def run(root,out,reference,production,iterations=12):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze-record.json').exists():raise ValueError('Experiment frozen; do not refit using a viewed test')
    required=out/'preregistered-protocol.json'
    if not required.exists():raise ValueError('Write the protocol and reserve the test before running')
    protocol=json.loads(required.read_text())
    if protocol['sealed_test']!=TEST or tuple(protocol['train'])!=TRAIN or protocol['validation']!=[VALID]:raise ValueError('Incorrect split')
    protocol.update(reference_sha256=digest(reference),production_sha256=digest(production),maximum_iterations=iterations)
    (out/'run-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    train=[];audits={}
    for mid in TRAIN:
        parts,audits[mid]=load_match(root,mid);train.extend(parts)
    validation,audits[VALID]=load_match(root,VALID)
    (out/'development-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    valrows=conditions(validation,100);previous=load_frozen(reference)
    refpred=[infer_frozen(previous,s,p,mask)[0] for s,p,c,mask in valrows]
    refscore=evaluate(valrows,refpred);refdiag=spell_diagnostics(valrows,refpred)
    best={'name':'previous_european','selection_spell_f1':refscore['selection_spell_f1'],'penalty':previous['penalty'],'strength':0,'iteration':None}
    shutil.copy2(reference,out/'candidate-model.npz');trials=[]
    def trial(name,model,probs,penalty,iteration=None,head=None,strength=0,events=None):
        pred=[decode(p,penalty,b if strength else None,strength) for p,b in zip(probs,events if events is not None else [None]*len(probs))]
        score=evaluate(valrows,pred);diag=spell_diagnostics(valrows,pred);ok=accepted(diag,refdiag)
        row={'name':name,'iteration':iteration,'penalty':penalty,'strength':strength,**score,'diagnostics':diag,'passes_false_positive_guard':ok};trials.append(row)
        if ok and row['selection_spell_f1']>best['selection_spell_f1']:
            best.clear();best.update({k:row[k] for k in ('name','iteration','penalty','strength','selection_spell_f1')});save_model(out/'candidate-model.npz',model,penalty,head if strength else None,strength)
        (out/'validation-trials.json').write_text(json.dumps(trials,indent=2)+'\n')
        return row
    sequences=[]
    for i,(s,pitch) in enumerate(train):
        w=loss_weights(s['y'])
        for swap in (False,True):
            y=s['y'].copy()
            if swap:y=np.where(y==0,1,np.where(y==1,0,y))
            y=np.where(y<2,y,-1)
            for masked in (False,True):sequences.append((instantaneous_features(s,pitch,swap,missing_mask(len(y),42+i) if masked else None),y.copy(),w.copy()))
    val=[(instantaneous_features(s,p),np.where(s['y']<2,s['y'],-1)) for s,p in validation]
    weighted_best={'score':-1};history=[]
    def progress(row,model):
        history.append(row);probs=[observation_probabilities(model,s,p,mask) for s,p,c,mask in valrows]
        for penalty in PENALTIES:
            r=trial('spell_weighted',model,probs,penalty,row['iteration'])
            if r['selection_spell_f1']>weighted_best['score']:
                weighted_best.update(score=r['selection_spell_f1'],iteration=row['iteration']);save_model(out/'weighted-backbone.npz',model,penalty)
        (out/'learning.json').write_text(json.dumps(history,indent=2)+'\n');print('Validation best:',json.dumps(best),flush=True)
    print('Training spell-weighted RNN using only the three training matches',flush=True)
    net.train(sequences,val,(24,),epochs=iterations,patience=iterations,on_epoch_model=progress,train_probe_sequences=[(instantaneous_features(s,p),np.where(s['y']<2,s['y'],-1)) for s,p in train])
    for name,state in [('previous_boundary',previous),('weighted_boundary',load_frozen(out/'weighted-backbone.npz'))]:
        model=state['model'];xs=[];ys=[]
        for i,(s,pitch) in enumerate(train):
            target,valid=boundary_supervision(s['y'])
            for masked in (False,True):
                mask=missing_mask(len(target),42+i) if masked else None
                p=observation_probabilities(model,s,pitch,mask);x=invariant_boundary_features(s,pitch,p,mask);ix=np.flatnonzero(valid)[::5];xs.append(x[ix]);ys.append(target[ix])
        x=np.concatenate(xs);y=np.concatenate(ys);rng=np.random.default_rng(42)
        positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
        if not len(positive) or not len(negative):raise ValueError('Boundary supervision lacks both classes')
        count=min(50000,max(len(positive),len(negative)));ix=np.r_[rng.choice(positive,count,replace=len(positive)<count),rng.choice(negative,count,replace=len(negative)<count)]
        print('Fitting balanced boundary head:',name,'positive native samples',len(positive),'negative',len(negative),flush=True)
        head=fit_classifier(x[ix],y[ix],classes=2,epochs=35)
        probs=[observation_probabilities(model,s,p,mask) for s,p,c,mask in valrows]
        events=[probabilities(head,invariant_boundary_features(s,p,pb,mask))[:,1] for (s,p,c,mask),pb in zip(valrows,probs)]
        for penalty in PENALTIES:
            for strength in STRENGTHS:trial(name,model,probs,penalty,weighted_best.get('iteration') if name.startswith('weighted') else None,head,strength,events)
        print('Boundary selection best:',json.dumps(best),flush=True)
    baselines=[{'window':w,**evaluate(valrows,[baseline_prediction(s,w,mask) for s,p,c,mask in valrows])} for w in (0,25,50,125)]
    baseline=max(baselines,key=lambda x:x['selection_spell_f1'])
    frozen={**best,'candidate_sha256':digest(out/'candidate-model.npz'),'reference_sha256':digest(reference),'production_sha256':digest(production),'baseline_window':baseline['window'],'test_labels_read_at_freeze':False,'source_hashes':{p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('rnn_deep.py'),Path(__file__).with_name('rnn_runtime.py'),Path(__file__).with_name('skillcorner_europe.py')]}}
    (out/'freeze-record.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Frozen. Opening reserved 2417 test for one evaluation.',flush=True)
    test,audit=load_match(root,TEST);rows=conditions(test,400);results={};diagnostics={}
    for name,path in [('candidate',out/'candidate-model.npz'),('previous_european',reference),('production_current',production)]:
        state=load_frozen(path);pred=[infer_frozen(state,s,p,mask)[0] for s,p,c,mask in rows];results[name]=evaluate(rows,pred);diagnostics[name]=spell_diagnostics(rows,pred)
    pred=[baseline_prediction(s,baseline['window'],mask) for s,p,c,mask in rows];results['baseline']=evaluate(rows,pred);diagnostics['baseline']=spell_diagnostics(rows,pred)
    report={'protocol':protocol,'frozen':frozen,'validation_reference':refscore,'validation_reference_diagnostics':refdiag,'validation_selected':max((t for t in trials if all(t[k]==best[k] for k in ('name','iteration','penalty','strength'))),key=lambda t:t['selection_spell_f1']) if best['name']!='previous_european' else {'scores':refscore,'diagnostics':refdiag},'test_audit':audit,'test_results':results,'test_diagnostics':diagnostics,'candidate_beats_previous_european':results['candidate']['selection_spell_f1']>results['previous_european']['selection_spell_f1'],'promoted_to_default':False,'test_status':'viewed_once_after_freeze','limitations':['Automatic provider labels, partial observed H/A runs, no supervised dead-ball state.','Unknown gaps censor spells and may inflate scores; this is not official challenge accuracy.','Fresh test shares both clubs with training; this is a different match, not unseen-team generalisation.','Validation PSG–Marseille has been reused repeatedly; selection may overfit it.','Merge counts are geometric overlap diagnostics, not confirmed event annotations.','Offline boundary context and Viterbi use future evidence.','The fresh test must never be used for subsequent tuning.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('Test summary:',json.dumps({k:v['selection_spell_f1'] for k,v in results.items()}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','out','reference','production'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iterations',type=int,default=12);a=p.parse_args();run(a.root,a.out,a.reference,a.production,a.iterations)
