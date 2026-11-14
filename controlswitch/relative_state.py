"""Physics-derived416-feature control evidence; no label or identity inputs."""
import numpy as np
from .model import shifted,rolling,trailing
from .control_state import state_features,swap_state_columns

WIDTH=24

def motion_features(s,mask=None):
    xy=np.asarray(s['xy']);teams=np.asarray(s['teams']);ball=np.asarray(s['ball']).copy()
    if mask is not None:ball[mask]=np.nan
    bv=(shifted(ball,5)-shifted(ball,-5))*2.5;pv=(shifted(xy,5)-shifted(xy,-5))*2.5
    bs=np.linalg.norm(bv,axis=1);ps=np.linalg.norm(pv,axis=2);goodb=np.isfinite(bv).all(1)&(bs<60)&np.isfinite(ball).all(1);goodp=np.isfinite(pv).all(2)&(ps<15)&np.isfinite(xy).all(2)
    oldv=shifted(bv,-5);oldbs=np.linalg.norm(oldv,axis=1);turn=1-np.sum(bv*oldv,axis=1)/np.maximum(bs*oldbs,.1);impulse=np.linalg.norm(bv-oldv,axis=1)*5;goodturn=goodb&shifted(goodb,-5)&(bs>1e-3)&(oldbs>1e-3)
    turn=np.where(goodturn,np.clip(turn,0,2)/2,0);impulse=np.where(goodturn,np.clip(impulse,0,100)/50,0);pa=np.linalg.norm(pv-shifted(pv,-5),axis=2)*5
    columns=[]
    for team in (0,1):
        ix=teams==team
        if not ix.any():raise ValueError('Both teams require tracking slots')
        rel=xy[:,ix]-ball[:,None,:];rv=pv[:,ix]-bv[:,None,:];valid=goodp[:,ix]&goodb[:,None];dist=np.linalg.norm(rel,axis=2);prox=np.where(valid,np.exp(-np.nan_to_num(dist,nan=100)/2),0)
        future=np.linalg.norm(rel+.25*rv,axis=2);past=np.linalg.norm(rel-.25*rv,axis=2);tau=np.clip(-np.sum(rel*rv,axis=2)/np.maximum(np.sum(rv*rv,axis=2),.1),0,.5);closest=np.linalg.norm(rel+tau[:,:,None]*rv,axis=2)
        alignment=np.sum(pv[:,ix]*bv[:,None,:],axis=2)/np.maximum(ps[:,ix]*bs[:,None],.1);direction=np.sum(rel*bv[:,None,:],axis=2)/np.maximum(dist*bs[:,None],.1)
        def minimum(a):return np.log1p(np.min(np.where(valid,a,100),axis=1))/5
        def maximum(a):return np.max(np.where(valid,np.nan_to_num(a,nan=0,posinf=0,neginf=0),0),axis=1)
        columns.extend([minimum(dist),minimum(past),minimum(future),maximum(prox*np.clip(alignment,-1,1)),maximum(prox*turn[:,None]),maximum(prox*impulse[:,None]),minimum(closest),maximum(np.exp(-np.nan_to_num(closest,nan=100)/2)),maximum(prox*np.clip(direction,-1,1)),maximum(prox*np.clip(pa[:,ix],0,20)/10),maximum(prox/(1+bs[:,None])),np.sum(valid&(dist<1),axis=1)/2])
    return np.nan_to_num(np.column_stack(columns),nan=0,posinf=0,neginf=0).astype(np.float32)

def relative_features(s,pitch,mask=None):
    x=motion_features(s,mask);extra=np.column_stack([x,rolling(x,5),rolling(x,25),trailing(x,50),trailing(x[::-1],50)[::-1]])
    return np.column_stack([state_features(s,pitch,mask),extra]).astype(np.float32)

def swap_relative(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=416:raise ValueError('Expected416 motion-control features')
    pair=np.r_[np.arange(12,24),np.arange(12)];order=np.r_[*(pair+24*i for i in range(5))]
    return np.column_stack([swap_state_columns(x[:,:296]),x[:,296:][:,order]])
