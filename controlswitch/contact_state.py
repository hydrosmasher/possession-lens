"""057: per-player contact persistence, pooled without player identities."""
import numpy as np
from .model import shifted,rolling,trailing
from .joint_state import joint_features,swap_joint

EXTRA=64
WIDTH=572

def contact_features(s,mask=None):
    xy=np.asarray(s['xy'],np.float32);ball=np.asarray(s['ball'],np.float32).copy();teams=np.asarray(s['teams']);n=len(ball)
    if mask is not None:ball[np.asarray(mask,bool)]=np.nan
    seen=np.isfinite(ball).all(1);validp=np.isfinite(xy).all(2);valid=validp&seen[:,None]
    pv=(shifted(xy,2)-shifted(xy,-2))*6.25;bv=(shifted(ball,2)-shifted(ball,-2))*6.25;ps=np.linalg.norm(pv,axis=2);bs=np.linalg.norm(bv,axis=1)
    velocity_valid=valid&np.isfinite(pv).all(2)&np.isfinite(bv).all(1)[:,None]&(ps<15)&(bs<60)[:,None]
    rel=xy-ball[:,None,:];rv=pv-bv[:,None,:];dist=np.linalg.norm(rel,axis=2);prox=np.where(valid,np.exp(-np.nan_to_num(dist,nan=100)/1.5),0).astype(np.float32)
    relative_speed=np.linalg.norm(rv,axis=2);coupling=np.where(velocity_valid,prox*np.exp(-np.nan_to_num(relative_speed,nan=100)/5),0).astype(np.float32)
    tau=np.clip(-np.sum(rel*rv,axis=2)/np.maximum(np.sum(rv*rv,axis=2),.1),0,.4);closest=np.linalg.norm(rel+tau[:,:,None]*rv,axis=2);intercept=np.where(velocity_valid,np.exp(-np.nan_to_num(closest,nan=100)/1.5),0)
    before=(ball-shifted(ball,-4))*6.25;after=(shifted(ball,4)-ball)*6.25;ib=np.linalg.norm(before,axis=1);ob=np.linalg.norm(after,axis=1);good=seen&np.isfinite(before).all(1)&np.isfinite(after).all(1)&(ib>.5)&(ob>.5)&(ib<60)&(ob<60)
    turn=np.where(good,np.clip(1-np.sum(before*after,axis=1)/np.maximum(ib*ob,.1),0,2)/2,0);impulse=np.where(good,np.clip(np.linalg.norm(after-before,axis=1)/20,0,1),0)
    alignment=np.where(velocity_valid,prox*np.clip(np.sum(pv*bv[:,None,:],axis=2)/np.maximum(ps*bs[:,None],.1),0,1)*np.minimum(ps/2,1),0)
    scores=[prox,coupling,intercept,prox*turn[:,None],prox*impulse[:,None],prox*rolling(prox,5),prox*rolling(prox,12),alignment]
    scores=[np.nan_to_num(z,nan=0,posinf=0,neginf=0).astype(np.float32) for z in scores]
    memories=[trailing(prox,w) for w in (5,12)]+[trailing(prox[::-1],w)[::-1] for w in (5,12)]+[trailing(coupling,5),trailing(coupling[::-1],5)[::-1]]
    acceleration=np.where(velocity_valid&shifted(velocity_valid,-2),np.clip(np.linalg.norm(pv-shifted(pv,-2),axis=2)*12.5/20,0,1),0);validity=rolling(validp.astype(np.float32),12);columns=[]
    for team in (0,1):
        ix=teams==team
        if not ix.any():raise ValueError('Both teams require tracking slots')
        pool=[np.sort(z[:,ix],axis=1)[:,::-1] for z in scores]
        cols=[z[:,0] for z in pool]+[z[:,min(1,z.shape[1]-1)] for z in pool]+[z[:,:min(3,z.shape[1])].mean(1) for z in pool]
        weights=prox[:,ix]*.2+coupling[:,ix]*.8;den=weights.sum(1);weights=weights/np.maximum(den[:,None],1e-8)
        cols += [(weights*z[:,ix]).sum(1) for z in memories]
        cols += [(weights*np.nan_to_num(acceleration[:,ix],nan=0)).sum(1),(weights*validity[:,ix]).sum(1)]
        columns.extend(cols)
    result=np.nan_to_num(np.column_stack(columns),nan=0,posinf=0,neginf=0).astype(np.float32)
    if result.shape!=(n,EXTRA):raise ValueError('Unexpected contact feature layout')
    return result

def contact_state(s,pitch,mask=None):return np.column_stack([joint_features(s,pitch,mask),contact_features(s,mask)]).astype(np.float32)
def swap_contact(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=WIDTH:raise ValueError('Expected572 contact-state features')
    return np.column_stack([swap_joint(x[:,:508]),x[:,540:572],x[:,508:540]])
