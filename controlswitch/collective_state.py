"""Ball-independent team shape/motion evidence for observation gaps."""
import warnings
import numpy as np
from .model import shifted,rolling
from .control_state import state_features,swap_state_columns

def attack_directions(xy,teams):
    x=np.asarray(xy)[:,:,0];teams=np.asarray(teams);seen=np.isfinite(x);left=teams[np.argmin(np.where(seen,x,np.inf),axis=1)];right=teams[np.argmax(np.where(seen,x,-np.inf),axis=1)];valid=seen.any(1)&(left!=right);vote=np.where(left[valid]==0,1,-1).sum();d=float(np.sign(vote));return np.array([d,-d])

def collective_features(s,pitch):
    xy=np.asarray(s['xy']);teams=np.asarray(s['teams']);directions=attack_directions(xy,teams);pv=(shifted(xy,5)-shifted(xy,-5))*2.5;speed=np.linalg.norm(pv,axis=2);parts=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for team in (0,1):
            ix=teams==team
            if not ix.any():raise ValueError('Both teams require slots')
            p=xy[:,ix];valid=np.isfinite(p).all(2);n=valid.sum(1);attacked=p[:,:,0]*directions[team];y=p[:,:,1];v=pv[:,ix];vs=speed[:,ix];good=valid&np.isfinite(v).all(2)&(vs<15);vx=np.where(good,v[:,:,0]*directions[team],np.nan);vy=np.where(good,v[:,:,1],np.nan);sp=np.where(good,vs,np.nan)
            cx=np.nanmean(attacked,axis=1);cy=np.nanmean(y,axis=1);sx=np.nanstd(attacked,axis=1);sy=np.nanstd(y,axis=1);q=np.nanquantile(attacked,[.1,.25,.75,.9],axis=1);dx=(shifted(cx,12)-shifted(cx,-12))*25/24;dy=(shifted(cy,12)-shifted(cy,-12))*25/24
            denom=np.maximum(n,1);vg=np.maximum(good.sum(1),1)
            parts.extend([n/11,np.nanmedian(attacked,axis=1)/pitch[0],cx/pitch[0],sx/pitch[0],sy/pitch[1],*(z/pitch[0] for z in q),(np.nanmax(attacked,axis=1)-np.nanmin(attacked,axis=1))/pitch[0],(np.nanmax(y,axis=1)-np.nanmin(y,axis=1))/pitch[1],sx*sy/(pitch[0]*pitch[1]),np.nanmean(vx,axis=1)/10,np.nanstd(vx,axis=1)/10,np.nanmean(vy,axis=1)/10,np.nanstd(vy,axis=1)/10,np.nanmean(sp,axis=1)/10,(vx>.5).sum(1)/vg,(vx<-.5).sum(1)/vg,good.sum(1)/11,dx/10,dy/10,((attacked>pitch[0]*.25)&valid).sum(1)/denom,((attacked<-pitch[0]*.25)&valid).sum(1)/denom,((np.abs(y)>pitch[1]*.25)&valid).sum(1)/denom,((np.abs(y)<pitch[1]*.15)&valid).sum(1)/denom])
    return np.nan_to_num(np.column_stack(parts),nan=0,posinf=0,neginf=0).astype(np.float32)

def collective_state(s,pitch,mask=None):
    x=collective_features(s,pitch);short=rolling(x,12);return np.column_stack([state_features(s,pitch,mask),x,short,rolling(x,50),shifted(short,25)-shifted(short,-25)]).astype(np.float32)

def swap_collective(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=504:raise ValueError('Expected504 collective-control features')
    pair=np.r_[np.arange(26,52),np.arange(26)];order=np.r_[*(pair+52*i for i in range(4))];return np.column_stack([swap_state_columns(x[:,:296]),x[:,296:][:,order]])
