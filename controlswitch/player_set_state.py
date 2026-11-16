"""060: identity-free raw player/ball nodes for a shared learned encoder."""
import numpy as np
from .model import shifted,trailing
NODES=16
FEATURES=21

def player_nodes(s,pitch,mask=None,return_indices=False):
    xy=np.asarray(s['xy'],np.float32);ball=np.asarray(s['ball'],np.float32).copy();teams=np.asarray(s['teams']);speed=np.asarray(s['speed']);n=len(ball)
    if mask is not None:ball[np.asarray(mask,bool)]=np.nan
    seen=np.isfinite(ball).all(1);valid=np.isfinite(xy).all(2);pv=(shifted(xy,2)-shifted(xy,-2))*6.25;bv=(shifted(ball,2)-shifted(ball,-2))*6.25;ps=np.linalg.norm(pv,axis=2);bs=np.linalg.norm(bv,axis=1);pgood=valid&np.isfinite(pv).all(2)&(ps<15);both=pgood&seen[:,None]&np.isfinite(bv).all(1)[:,None]&(bs<60)[:,None]
    rel=xy-ball[:,None,:];rv=pv-bv[:,None,:];distance=np.linalg.norm(rel,axis=2);prox=np.where(valid&seen[:,None],np.exp(-np.nan_to_num(distance,nan=100)/1.5),0).astype(np.float32)
    before=(ball-shifted(ball,-4))*6.25;after=(shifted(ball,4)-ball)*6.25;ib=np.linalg.norm(before,axis=1);ob=np.linalg.norm(after,axis=1);good=seen&np.isfinite(before).all(1)&np.isfinite(after).all(1)&(ib>.5)&(ob>.5)&(ib<60)&(ob<60);turn=np.where(good,np.clip(1-np.sum(before*after,axis=1)/np.maximum(ib*ob,.1),0,2)/2,0);impulse=np.where(good,np.clip(np.linalg.norm(after-before,axis=1)/20,0,1),0)
    memories=[trailing(prox,5),trailing(prox,25),trailing(prox[::-1],5)[::-1],trailing(prox[::-1],25)[::-1]];out=np.zeros((n,2,NODES,FEATURES),np.float32);selected=np.full((n,2,NODES),-1,np.int32)
    for team in (0,1):
        ix=teams==team
        if not ix.any():raise ValueError('Both teams require slots')
        p=xy[:,ix];ok=valid[:,ix];den=np.maximum(ok.sum(1),1);centroid=np.where(ok[:,:,None],p,0).sum(1)/den[:,None];relative_center=p-centroid[:,None,:];r=rel[:,ix];v=pv[:,ix];dv=rv[:,ix];vg=both[:,ix];vp=pgood[:,ix]
        columns=[np.where(seen[:,None],r[:,:,0]/30,0),np.where(seen[:,None],r[:,:,1]/30,0),np.where(vp,v[:,:,0]/10,0),np.where(vp,v[:,:,1]/10,0),np.where(vg,dv[:,:,0]/20,0),np.where(vg,dv[:,:,1]/20,0),np.where(vp,ps[:,ix]/12,0),np.where(vg,np.linalg.norm(dv,axis=2)/30,0),np.where(seen[:,None],np.log1p(distance[:,ix])/5,0),vg,seen[:,None]*np.ones(ok.shape),relative_center[:,:,0]/30,relative_center[:,:,1]/30,np.clip(np.nan_to_num(speed[:,ix],nan=0),0,30)/12,*[z[:,ix] for z in memories],prox[:,ix]*turn[:,None],prox[:,ix]*impulse[:,None],ok]
        nodes=np.nan_to_num(np.stack(columns,axis=-1),nan=0,posinf=0,neginf=0).astype(np.float32);nodes=np.clip(nodes,-4,4);nodes[~ok]=0;key=np.where(seen[:,None],np.nan_to_num(distance[:,ix],nan=np.inf),np.linalg.norm(relative_center,axis=2));key=np.where(ok,key,np.inf);take=min(NODES,nodes.shape[1]);order=np.argsort(key,axis=1,kind='stable')[:,:take];out[:,team,:take]=np.take_along_axis(nodes,order[:,:,None],axis=1);selected[:,team,:take]=np.flatnonzero(ix)[order]
    return (out,selected) if return_indices else out

def transform_nodes(nodes,code):
    if not isinstance(code,(int,np.integer)) or not 0<=code<8:raise ValueError('Expected augmentation code0..7')
    x=np.asarray(nodes).copy()
    if x.shape[-3:]!=(2,NODES,FEATURES):raise ValueError('Unexpected player-node shape')
    if code&1:x=x[...,::-1,:,:]
    if code&2:x[...,[0,2,4,11]]*=-1
    if code&4:x[...,[1,3,5,12]]*=-1
    return x
