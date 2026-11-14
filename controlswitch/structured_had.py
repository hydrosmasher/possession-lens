"""Follow-up021: evidence-gated boundaries and factorised play-state decoding."""
from pathlib import Path
import argparse,json
import numpy as np
from .boundary_had import adaptive_decode
from .tracker_robustness import CONDITIONS,evaluate
from .model import shifted,rolling
from .pipeline import digest

VERSION='structured-had-021'
VARIANTS=('supported_boundaries','hierarchical','hierarchical_supported')

def supported_score(p,q):
    p=np.asarray(p);q=np.asarray(q)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or q.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(q).all() or (p<0).any() or ((q<0)|(q>1)).any():raise ValueError('Invalid boundary evidence')
    live=p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12)
    local=rolling(live,6)
    def flip(k):
        before=shifted(local,-k);after=shifted(local,k)
        return before[:,0]*after[:,1]+before[:,1]*after[:,0]
    return q*np.clip(np.maximum(flip(12),flip(25)),0,1)

def decode_two(p,change_cost):
    p=np.asarray(p);cost=np.broadcast_to(np.asarray(change_cost,float),(len(p),))
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or not np.isfinite(p).all() or (p<0).any() or (p.sum(1)<=0).any() or not np.isfinite(cost).all() or (cost<0).any():raise ValueError('Invalid binary decoding inputs')
    logp=np.log(np.maximum(p,1e-8));value=logp[0].copy();back=np.zeros((len(p),2),np.int8)
    for i in range(1,len(p)):
        choices=value[:,None]-np.array([[0,cost[i]],[cost[i],0]])
        back[i]=choices.argmax(0);value=choices[back[i],np.arange(2)]+logp[i]
    labels=np.empty(len(p),np.int8);labels[-1]=value.argmax()
    for i in range(len(p)-1,0,-1):labels[i-1]=back[i,labels[i]]
    return labels

def hierarchical_decode(p,score=None):
    p=np.asarray(p)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all() or (p<0).any() or (p.sum(1)<=0).any():raise ValueError('Invalid H/A/D probabilities')
    p=p/p.sum(1,keepdims=True)
    play=decode_two(np.column_stack([1-p[:,2],p[:,2]]),3.)
    labels=np.full(len(p),2,np.int8);live=play==0
    changes=np.flatnonzero(np.r_[False,live,False][1:]!=np.r_[False,live,False][:-1])
    if score is not None:
        score=np.asarray(score)
        if score.shape!=(len(p),) or not np.isfinite(score).all() or ((score<0)|(score>1)).any():raise ValueError('Invalid event score')
    for a,b in zip(changes[::2],changes[1::2]):
        hp=p[a:b,:2];hp=hp/np.maximum(hp.sum(1,keepdims=True),1e-12)
        cost=12. if score is None else 12*np.clip(1+.5*(1-2*score[a:b]),.25,2)
        labels[a:b]=decode_two(hp,cost)
    return labels

def predict(p,q,variant):
    if variant=='supported_boundaries':return adaptive_decode(p,supported_score(p,q),.5,'balanced')
    if variant=='hierarchical':return hierarchical_decode(p)
    if variant=='hierarchical_supported':return hierarchical_decode(p,supported_score(p,q))
    raise ValueError('Unknown structured variant')

def run(previous,out):
    previous=Path(previous);out=Path(out)
    if out.exists():raise ValueError('Choose a new directory')
    out.mkdir(parents=True)
    protocol={'version':VERSION,'history':'Follow-up after020 variants failed false-positive guard; same extensively reused external Bundesliga labels, not independent testing','competition_data_used':False,'input_protocol_sha256':digest(previous/'protocol.json'),'frozen_emissions':'019;020 event heads unchanged; no new fitting','variants':list(VARIANTS),'supported_boundaries':'multiply event score by max opposing-team support between+-12 and+-25frame means, radius6; no hard threshold or frame-specific correction','hierarchical':'Binary live/dead Viterbi cost3 based on summed live probability, then H/A Viterbi12 independently within live runs; no forced minimum duration; H/A uncertainty cannot change binary play likelihood','hierarchical_supported':'same hierarchy, with supported020 score adapting H/A costs at strength.5','selection':'Best four-condition mean spellF1 with no total misses/merges/FP increase; each clean/ball-gap fold loss<=.01; each tracker_noise fold no loss; each deadF1 loss<=.005. No post-result tuning.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[];current=[];preds={v:[] for v in VARIANTS};folds=[]
    for fold in (0,1):
        fr=[];fc=[];fp={v:[] for v in VARIANTS};fd=out/f'fold-{fold}';fd.mkdir()
        for half in (0,1):
            for ci,c in enumerate(CONDITIONS):
                i=half*4+ci
                with np.load(previous/f'fold-{fold}/validation-{i}.npz',allow_pickle=False) as a:y=a['reference'];p=a['emissions'];q=a['boundary_score'];base=a['current']
                fr.append({'y':y,'condition':c});fc.append(base)
                for v in VARIANTS:fp[v].append(predict(p,q,v))
                np.savez_compressed(fd/f'validation-{i}.npz',reference=y,current=base,**{v:p[-1] for v,p in fp.items()})
        r={'validation':'J03WN1' if fold==0 else 'J03WMX','current':evaluate(fr,fc),'variants':{v:evaluate(fr,p,fc) for v,p in fp.items()}};folds.append(r);(fd/'report.json').write_text(json.dumps(r,indent=2)+'\n');rows+=fr;current+=fc
        for v in preds:preds[v]+=fp[v]
    base=evaluate(rows,current);trials=[]
    for v in VARIANTS:
        score=evaluate(rows,preds[v],current);a=score['totals'];b=base['totals'];guards={'mean_f1':score['mean_spell_f1']>base['mean_spell_f1'],'misses':a['missed']<=b['missed'],'merges':a['merges']<=b['merges'],'false_positives':a['false_positive']<=b['false_positive'],'prior_conditions':all(f['variants'][v][c]['spell_f1_mean']>=f['current'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'tracker_noise':all(f['variants'][v]['tracker_noise']['spell_f1_mean']>=f['current']['tracker_noise']['spell_f1_mean'] for f in folds),'dead_f1':all(f['variants'][v][c]['dead_f1']>=f['current'][c]['dead_f1']-.005 for f in folds for c in CONDITIONS)}
        trials.append({'variant':v,'scores':score,'guards':guards,'eligible':all(guards.values())})
    eligible=[t for t in trials if t['eligible']];report={'protocol':protocol,'folds':folds,'current':base,'trials':trials,'selected':max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None,'best_unrestricted':max(trials,key=lambda t:t['scores']['mean_spell_f1']),'fresh_test_results':None,'promoted_to_default':False,'limitations':['Adaptive follow-up on the same reused two-match labels; no unbiased test.','Synthetic errors differ from actual challenge noise.','Binary play-state hierarchy can remove valid short stoppages, and must pass the guards.','Same possession instances repeat across conditions.','No challenge data, video, public labels or score used.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('CURRENT',base['mean_spell_f1'],base['totals'])
    for t in trials:print(t['variant'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'])
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.previous,a.out)
if __name__=='__main__':main()
