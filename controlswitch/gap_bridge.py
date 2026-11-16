"""066: player-motion specialist for bracketed missing-ball intervals.

Endpoint ball locations remain observations; no invented ball path is generated.
"""
import numpy as np
from .collective_state import collective_features
from .model import trailing
TEAM_WIDTH = 90
WIDTH = 188


def bridge_features(sequence, pitch, mask=None):
    ball=np.array(sequence['ball'],dtype=float,copy=True);xy=np.asarray(sequence['xy']);teams=np.asarray(sequence['teams']);n=len(ball)
    if mask is not None:
        mask=np.asarray(mask,bool)
        if mask.shape!=(n,):raise ValueError('Invalid bridge ball mask')
        ball[mask]=np.nan
    if n==0:raise ValueError('Empty bridge sequence')
    seen=np.isfinite(ball).all(1);t=np.arange(n);previous=np.maximum.accumulate(np.where(seen,t,-1));following=np.minimum.accumulate(np.where(seen,t,n)[::-1])[::-1];bracketed=(previous>=0)&(following<n);left=np.maximum(previous,0);right=np.minimum(following,n-1);length=following-previous-1;eligible=(~seen)&bracketed&(length<=300)
    c=collective_features(sequence,pitch);past=trailing(c,25);future=trailing(c[::-1],25)[::-1];anchors=[]
    # Derivatives require the ball actually observed; gaps never become velocity0.
    bv=np.full_like(ball,np.nan);bv[1:]=(ball[1:]-ball[:-1])*25
    pv=np.full_like(xy,np.nan,dtype=float);pv[1:]=(xy[1:]-xy[:-1])*25
    for team in (0,1):
        pos=xy[:,teams==team];v=pv[:,teams==team];distance=np.linalg.norm(pos-ball[:,None],axis=2);valid=np.isfinite(distance);ds=np.sort(np.where(valid,distance,100),axis=1);prox=np.where(valid,np.exp(-np.nan_to_num(distance,nan=100)/2),0);rel=np.linalg.norm(v-bv[:,None],axis=2);good=np.isfinite(rel)&(np.linalg.norm(v,axis=2)<15)&(np.linalg.norm(bv,axis=1)[:,None]<60)
        contact=(prox*np.where(good,np.exp(-np.nan_to_num(rel,nan=100)/5),0)).max(1)
        a=np.column_stack([np.log1p(ds[:,0]),np.log1p(ds[:,min(1,ds.shape[1]-1)]),(ds<2).sum(1)/3,(ds<5).sum(1)/5,prox.sum(1)/3,contact]);a[~seen]=0
        anchors.append(a)
    pair=[]
    for team in (0,1):
        sl=slice(26*team,26*(team+1));a=anchors[team];before=np.where(bracketed[:,None],a[left],0);after=np.where(bracketed[:,None],a[right],0);pair.append(np.column_stack([c[:,sl],past[:,sl],future[:,sl],before,after]))
    denom=np.maximum(following-previous,1);common=np.column_stack([np.clip(t-previous,0,625)/300,np.clip(following-t,0,625)/300,np.clip(length,0,625)/300,(t-previous)/denom,previous>=0,following<n,seen,eligible])
    x=np.nan_to_num(np.column_stack([*pair,common]),nan=0,posinf=0,neginf=0).astype(np.float32)
    if x.shape!=(n,WIDTH) or not np.isfinite(x).all():raise ValueError('Invalid gap bridge features')
    return x,eligible,seen


def swap_bridge(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=WIDTH:raise ValueError('Invalid bridge width')
    return np.column_stack([x[:,TEAM_WIDTH:2*TEAM_WIDTH],x[:,:TEAM_WIDTH],x[:,2*TEAM_WIDTH:]])


def anchored_gap_decode(frozen,prior,expert,eligible,mix=.5,penalty=12.):
    """Revise H/A only inside eligible gaps; lock D and all observed frames."""
    frozen=np.asarray(frozen);prior=np.asarray(prior,float);expert=np.asarray(expert,float);eligible=np.asarray(eligible,bool)
    n=len(frozen)
    if prior.shape!=(n,2) or expert.shape!=prior.shape or eligible.shape!=(n,) or not np.isin(frozen,[0,1,2]).all() or not np.isfinite(prior).all() or not np.isfinite(expert).all() or (prior<0).any() or (expert<0).any() or not 0<=mix<=1:raise ValueError('Invalid bridge decoding input')
    # A zero mix must be the exact frozen baseline, rather than a new decoder.
    if mix==0:return frozen.copy()
    p=prior/np.maximum(prior.sum(1,keepdims=True),1e-12);g=expert/np.maximum(expert.sum(1,keepdims=True),1e-12);logp=np.log(np.maximum((1-mix)*p+mix*g,1e-8));out=frozen.copy();active=eligible&(frozen<2);bounds=np.flatnonzero(np.diff(np.r_[False,active,False]))
    for a,b in bounds.reshape(-1,2):
        score=logp[a].copy();back=np.zeros((b-a,2),np.int8)
        if a>0 and frozen[a-1]<2:score[1-frozen[a-1]]-=penalty
        for t in range(a+1,b):
            choices=score[:,None]-np.array([[0,penalty],[penalty,0]]);back[t-a]=choices.argmax(0);score=choices.max(0)+logp[t]
        if b<n and frozen[b]<2:score[1-frozen[b]]-=penalty
        state=int(score.argmax())
        for t in range(b-1,a-1,-1):
            out[t]=state
            if t>a:state=int(back[t-a,state])
    return out
