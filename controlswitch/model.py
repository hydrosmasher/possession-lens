"""Observation-only features, learned control-change evidence and structured decoding.

No feature function accepts possession labels. No fit operation occurs at inference time.
"""
import warnings
import numpy as np

FEATURE_VERSION='geometry-context-v1'
def rolling(a,radius):
    n=len(a);idx=np.arange(n);lo=np.maximum(0,idx-radius);hi=np.minimum(n,idx+radius+1)
    c=np.vstack([np.zeros((1,a.shape[1])),np.cumsum(a,axis=0,dtype=np.float64)])
    return ((c[hi]-c[lo])/(hi-lo)[:,None]).astype(np.float32)

def shifted(a,offset):
    return a[np.clip(np.arange(len(a))+offset,0,len(a)-1)]

def instantaneous_features(sequence,pitch,swap=False,ball_missing=None):
    xy=sequence['xy'];speed=sequence['speed'];teams=sequence['teams'];ball=sequence['ball'].copy();bs=sequence['ball_speed'].copy()
    if ball_missing is not None:ball[ball_missing]=np.nan;bs[ball_missing]=np.nan
    seen=np.isfinite(ball).all(axis=1);n=len(ball)
    dist=np.linalg.norm(xy-ball[:,None,:],axis=2);dist[~np.isfinite(dist)]=100
    delta=np.linalg.norm(ball-shifted(ball,-1),axis=1)*25
    base=[seen.astype(float),np.nan_to_num(np.clip(bs,0,50),nan=0)/25,np.nan_to_num(np.clip(delta,0,50),nan=0)/25,
          np.nan_to_num(ball[:,0],nan=0)/pitch[0],np.nan_to_num(ball[:,1],nan=0)/pitch[1],
          np.nan_to_num(np.minimum(pitch[0]/2-np.abs(ball[:,0]),pitch[1]/2-np.abs(ball[:,1])),nan=0)/10]
    for t in ([1,0] if swap else [0,1]):
        mask=teams==t;pos=xy[:,mask];sp=speed[:,mask];ds=np.sort(dist[:,mask],axis=1)
        if ds.shape[1]==0:raise ValueError('Both teams must have tracking slots')
        valid=np.isfinite(pos).all(axis=2);count=valid.sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            centroid=np.nanmean(pos,axis=1);spread=np.nanstd(pos,axis=1)
            mean_speed=np.nanmean(sp,axis=1);max_speed=np.nanmax(sp,axis=1)
        base += [np.log1p(ds[:,0]),np.log1p(ds[:,min(1,ds.shape[1]-1)]),
                 (ds<2).sum(axis=1)/3,(ds<5).sum(axis=1)/5,count/11,
                 np.nan_to_num(mean_speed,nan=0)/10,np.nan_to_num(max_speed,nan=0)/12,
                 np.nan_to_num(centroid[:,0],nan=0)/pitch[0],np.nan_to_num(centroid[:,1],nan=0)/pitch[1],
                 np.nan_to_num(spread[:,0],nan=0)/pitch[0],np.nan_to_num(spread[:,1],nan=0)/pitch[1],
                 np.nan_to_num(np.linalg.norm(centroid-ball,axis=1),nan=0)/50]
    return np.column_stack(base).astype(np.float32)

def interaction_features(sequence,swap=False,ball_missing=None):
    """Causal 2-D interaction evidence, not confirmed physical contacts.

    Joint proximity / trajectory / player movement features are pooled by team.
    Unreliable track-slot displacement is masked rather than treated as acceleration.
    """
    xy=sequence['xy'];teams=sequence['teams'];ball=sequence['ball'].copy()
    if ball_missing is not None:ball[ball_missing]=np.nan
    bv=(ball-shifted(ball,-1))*25
    pv=(xy-shifted(xy,-1))*25
    goodball=np.isfinite(bv).all(axis=1)&(np.linalg.norm(bv,axis=1)<60)
    goodplayer=np.isfinite(pv).all(axis=2)&(np.linalg.norm(pv,axis=2)<12)
    previous_bv=shifted(bv,-1)
    turn=1-np.sum(bv*previous_bv,axis=1)/(np.linalg.norm(bv,axis=1)*np.linalg.norm(previous_bv,axis=1)+1e-6)
    turn=np.clip(np.nan_to_num(turn,nan=0),0,2)/2
    output=[]
    for team in ([1,0] if swap else [0,1]):
        pos=xy[:,teams==team];velocity=pv[:,teams==team];valid=goodplayer[:,teams==team]&goodball[:,None]
        dist=np.linalg.norm(pos-ball[:,None,:],axis=2)
        previous_dist=np.linalg.norm(shifted(pos,-1)-shifted(ball,-1)[:,None,:],axis=2)
        closeness=np.exp(-np.nan_to_num(dist,nan=100)**2/8)
        relative=np.linalg.norm(velocity-bv[:,None,:],axis=2)
        alignment=np.sum(velocity*bv[:,None,:],axis=2)/(np.linalg.norm(velocity,axis=2)*np.linalg.norm(bv,axis=1)[:,None]+1e-6)
        closing=np.clip((previous_dist-dist)*25,-20,20)/20
        soft_association=np.where(valid,closeness*np.exp(-np.nan_to_num(relative,nan=100)/5),0)
        contact_candidate=np.where(valid,closeness*turn[:,None],0)
        output.extend([np.max(soft_association,axis=1),np.max(contact_candidate,axis=1),
                       np.max(np.where(valid,closeness*np.nan_to_num(alignment,nan=0),0),axis=1),
                       np.max(np.where(valid,closeness*np.nan_to_num(closing,nan=0),0),axis=1),
                       np.sum(valid&(dist<3),axis=1)/3])
    return np.nan_to_num(np.column_stack(output),nan=0).astype(np.float32)

def features(sequence,pitch,swap=False,ball_missing=None,include_interactions=False):
    a=instantaneous_features(sequence,pitch,swap,ball_missing)
    if include_interactions:a=np.column_stack([a,interaction_features(sequence,swap,ball_missing)])
    # Offline context, always computed within a single half. No label or status inputs.
    return np.column_stack([a,rolling(a,12),rolling(a,50),shifted(rolling(a,12),25)-shifted(rolling(a,12),-25)]).astype(np.float32)

def trailing(a,width):
    idx=np.arange(len(a));lo=np.maximum(0,idx-width+1);hi=idx+1
    c=np.vstack([np.zeros((1,a.shape[1])),np.cumsum(a,axis=0,dtype=np.float64)])
    return ((c[hi]-c[lo])/(hi-lo)[:,None]).astype(np.float32)

def causal_features(sequence,pitch,swap=False,ball_missing=None):
    a=np.column_stack([instantaneous_features(sequence,pitch,swap,ball_missing),
                       interaction_features(sequence,swap,ball_missing)])
    return np.column_stack([a,trailing(a,25),trailing(a,50),a-shifted(a,-25)]).astype(np.float32)

def turnover_targets(y,horizon=25):
    """Next direct H↔A turnover within horizon; do not count a restart as a turnover.

    Current dead/unknown frames, incomplete horizons and horizons containing unknown
    reference labels are excluded (-1). Labels are only used to make targets.
    """
    n=len(y);indices=np.arange(n);events=np.flatnonzero(y[1:]!=y[:-1])+1
    event_index=np.searchsorted(events,indices,side='right')
    upcoming=np.r_[events,n][event_index]
    after=np.r_[y,np.int8(3)][upcoming]
    target=((upcoming<=indices+horizon)&(after<2)&(after!=y)).astype(np.int8)
    unknown=np.r_[0,np.cumsum(y==3)]
    incomplete=(indices+horizon>=n)|((unknown[np.minimum(indices+horizon+1,n)]-unknown[indices])>0)
    target[(y>=2)|incomplete]=-1
    return target

def softmax(z):
    z=z-z.max(axis=1,keepdims=True);e=np.exp(z);return e/e.sum(axis=1,keepdims=True)

def fit_classifier(x,y,classes=3,epochs=45,seed=42):
    """Small standardised multinomial logistic model, mini-batch Adam, fixed seed."""
    valid=(y>=0)&(y<classes);x=x[valid];y=y[valid]
    if len(np.unique(y))!=classes:raise ValueError('Training split does not contain all required classes')
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),.05)
    x=np.clip((x-mean)/scale,-8,8);x=np.column_stack([x,np.ones(len(x))]).astype(np.float32)
    w=np.zeros((x.shape[1],classes),dtype=np.float32);m=np.zeros_like(w);v=np.zeros_like(w);rng=np.random.default_rng(seed);step=0
    for epoch in range(epochs):
        order=rng.permutation(len(x))
        for start in range(0,len(x),1024):
            ix=order[start:start+1024];xb=x[ix];p=softmax(xb@w);p[np.arange(len(ix)),y[ix]]-=1
            gradient=xb.T@p/len(ix)+.0005*w;gradient[-1]-=.0005*w[-1]
            step+=1;m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
            w-=.01*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    return {'weights':w,'mean':mean,'scale':scale}

def probabilities(model,x):
    z=np.clip((x-model['mean'])/model['scale'],-8,8)
    return softmax(z@model['weights'][:-1]+model['weights'][-1])

def boundary_features(x,p):
    # Probability contrasts and observation changes on both sides of a possible switch.
    return np.column_stack([x,np.abs(shifted(x,12)-shifted(x,-12)),p,
                            shifted(p,12)-shifted(p,-12),np.abs(shifted(p,25)-shifted(p,-25))]).astype(np.float32)

def boundary_targets(y,radius=12):
    valid=(y[:-1]<3)&(y[1:]<3)
    changes=np.r_[False,(y[1:]!=y[:-1])&valid]
    return (rolling(changes[:,None].astype(float),radius)[:,0]>0).astype(np.int8)

def decode(p,penalty,boundary=None,strength=0):
    """Viterbi over stay / H↔A / stop / restart explanations at every frame.

    No hard candidate pruning: a missed candidate cannot forbid a genuine transition.
    The learned boundary verifier changes transition costs, never the observed labels.
    """
    n=len(p);logp=np.log(np.maximum(p,1e-8));back=np.zeros((n,3),np.int8);score=logp[0].copy()
    costs=np.full(n,penalty,dtype=float)
    if boundary is not None:costs=penalty*np.clip(1+strength*(1-2*boundary),.15,3)
    eye=np.eye(3,dtype=bool)
    for i in range(1,n):
        candidates=score[:,None]-np.where(eye,0,costs[i])
        back[i]=np.argmax(candidates,axis=0);score=candidates[back[i],np.arange(3)]+logp[i]
    labels=np.empty(n,dtype=np.int8);labels[-1]=np.argmax(score)
    for i in range(n-1,0,-1):labels[i-1]=back[i,labels[i]]
    return labels

def nearest_codes(sequence,ball_missing=None):
    ball=sequence['ball'].copy()
    if ball_missing is not None:ball[ball_missing]=np.nan
    d=np.linalg.norm(sequence['xy']-ball[:,None,:],axis=2);d[~np.isfinite(d)]=np.inf
    codes=sequence['teams'][np.argmin(d,axis=1)].astype(np.int8)
    codes[~np.isfinite(d).any(axis=1)]=-1
    return codes
