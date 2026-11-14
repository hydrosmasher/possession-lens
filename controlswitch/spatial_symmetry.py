"""Exact spatial-reflection transforms for the508-feature motion layout."""
import numpy as np
from .joint_state import swap_joint
from .gated_rnn import normalise,make_window,CORE,CONTEXT

def reflection_signs(flip_x=False,flip_y=False):
    signs=np.ones(508,np.float32)
    for offset in range(0,240,30):
        if flip_x:signs[offset+np.array([3,13,25])]=-1
        if flip_y:signs[offset+np.array([4,14,26])]=-1
    if flip_y:
        signs[416+np.array([14,21,40,47])]=-1
        signs[468+np.array([2,9,12,19])]=-1
        signs[488+np.array([2,9,12,19])]=-1
    return signs

def reflect_sequence(s,flip_x=False,flip_y=False):
    out={k:v for k,v in s.items() if k!='y'};sign=np.array([-1 if flip_x else 1,-1 if flip_y else 1],dtype=np.float32);out['xy']=s['xy']*sign;out['ball']=s['ball']*sign;return out

def augmented_window(x,y,w,t,v,start,mean,scale,code=0,spatial=True):
    """Raw feature cache; transform/normalise only the actual bounded window."""
    if not isinstance(code,(int,np.integer)) or not 0<=code<8:raise ValueError('Expected augmentation code0..7')
    a=max(0,start-CONTEXT);b=min(len(x),start+CORE+CONTEXT);offset=a-(start-CONTEXT);local=x[a:b]
    if code&1:local=swap_joint(local)
    if spatial:local=local*reflection_signs(bool(code&2),bool(code&4))
    xx,yy,tt,vv=make_window(x,y,start,t,v);xx[:]=0;xx[offset:offset+b-a]=normalise(local,mean,scale);ww=np.zeros(len(yy),np.float32);n=min(CORE,len(x)-start);ww[CONTEXT:CONTEXT+n]=w[start:start+n]
    if code&1:yy=np.where(yy>=0,1-yy,-1)
    ww[yy<0]=0;return xx,yy,ww,tt,vv
