"""Separate bracketed ball-location hypotheses; original observations stay intact."""
import numpy as np
from .joint_state import joint_features,swap_joint

def ball_hypothesis(s,mask=None,max_gap=300):
    if max_gap!=300:raise ValueError('Expected frozen300frame hypothesis limit')
    ball=np.asarray(s['ball']).copy();n=len(ball)
    if mask is not None:
        if np.asarray(mask).shape!=(n,):raise ValueError('Invalid ball mask')
        ball[mask]=np.nan
    seen=np.isfinite(ball).all(1);filled=np.zeros(n,bool);bounds=np.flatnonzero(np.r_[False,~seen,False][1:]!=np.r_[False,~seen,False][:-1])
    for a,b in zip(bounds[::2],bounds[1::2]):
        if a==0 or b==n or b-a>300:continue
        speed=np.linalg.norm(ball[b]-ball[a-1])*25/(b-a+1)
        if not np.isfinite(speed) or speed>50:continue
        f=np.arange(1,b-a+1)/(b-a+1);ball[a:b]=ball[a-1]+f[:,None]*(ball[b]-ball[a-1]);filled[a:b]=True
    return ball,seen,filled

def latent_features(s,pitch,mask=None):
    base=joint_features(s,pitch,mask);ball,_,_=ball_hypothesis(s,mask);dist=np.linalg.norm(np.asarray(s['xy'])-ball[:,None,:],axis=2);dist[~np.isfinite(dist)]=100;out=[]
    for t in (0,1):
        ds=np.sort(dist[:,np.asarray(s['teams'])==t],axis=1)
        if ds.shape[1]==0:raise ValueError('Both teams need slots')
        out.extend([np.log1p(ds[:,0]),np.log1p(ds[:,min(1,ds.shape[1]-1)])])
    return np.column_stack([base,*out]).astype(np.float32)

def swap_latent(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=512:raise ValueError('Expected512 latent-gap features')
    return np.column_stack([swap_joint(x[:,:508]),x[:,[510,511,508,509]]])
