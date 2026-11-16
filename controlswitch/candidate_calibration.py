"""073: score genuine/false short candidates, including already-correct spells."""
import numpy as np
from .proposal_calibration import event_features
from .pipeline import pm
COLUMNS=np.array([0,5,6,7,8,9,10,11,12,13,14])


def candidate_pool(case,threshold):
    p=case['proposal'];seen=case['seen'];motion=case['motion'];valid=case['receipt_valid'];receipt=case['receipt'];events=[];edges=np.r_[0,np.flatnonzero(p[1:]!=p[:-1])+1,len(p)]
    for a,b in zip(edges[:-1],edges[1:]):
        k=int(p[a])
        if k==2 or a==0 or b==len(p) or b-a>75:continue
        if p[a-1]!=1-k or p[b]!=1-k or not seen[a-1:b+1].all():continue
        support=valid[a:b,k]&(motion[a:b,k,0]<=2.5)&(receipt[a:b,k]>=threshold)
        if support.any():events.append({'start':int(a),'end':int(b),'team':k})
    x=event_features(case['current'],case['q'],case['g'],motion,valid,receipt,threshold,events)
    return events,x[:,COLUMNS]


def pool_targets(y,events):
    y=np.asarray(y);gt=pm.possession_spells(y,pm.valid_runs(y));labels=[];known=[]
    for e in events:
        a,b,k=e['start'],e['end'],e['team'];known.append(bool((y[a-1:b+1]<3).all()));hit=False
        for u,v,team in gt:
            if team!=k:continue
            inter=max(0,min(b,v)-max(a,u));union=(b-a)+(v-u)-inter
            if inter/max(union,1)>=.5:hit=True;break
        labels.append(int(hit))
    return np.asarray(labels,np.int8),np.asarray(known,bool)


def positive_retention_threshold(scores,labels):
    scores=np.asarray(scores);labels=np.asarray(labels)
    if scores.shape!=labels.shape or not np.isfinite(scores).all():raise ValueError('Invalid calibration candidates')
    positive=scores[labels==1]
    # No positive calibration evidence gives no authority to filter.
    return float(positive.min()) if len(positive) else 0.
