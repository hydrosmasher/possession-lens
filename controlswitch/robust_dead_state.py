"""059: team-invariant dead-ball evidence from contact and collective motion."""
import numpy as np
from .deadball import observation_features
from .contact_state import contact_features
from .collective_state import collective_features
from .quality_gate import quality_features
from .model import rolling
WIDTH=487

def invariant_pair(x,half):
    return np.column_stack([(x[:,:half]+x[:,half:])*.5,np.abs(x[:,:half]-x[:,half:])])

def robust_dead_features(s,pitch,mask=None):
    old=observation_features(s,pitch,mask);contact=invariant_pair(contact_features(s,mask),32);collective=invariant_pair(collective_features(s,pitch),26);q=quality_features(s,mask);x=np.column_stack([old,contact,collective,rolling(collective,12),rolling(collective,50),q]).astype(np.float32)
    if x.shape!=(len(s['ball']),WIDTH) or not np.isfinite(x).all():raise ValueError('Invalid487feature dead-ball layout')
    return x

def blend_dead(q,new_dead,mix):
    q=np.asarray(q);nd=np.asarray(new_dead)
    if mix not in (.5,1.) or q.ndim!=2 or q.shape[1]!=3 or nd.shape!=(len(q),) or not np.isfinite(q).all() or not np.isfinite(nd).all() or ((nd<0)|(nd>1)).any() or (q<0).any() or (q.sum(1)<=0).any():raise ValueError('Invalid dead-expert fusion')
    old=np.clip(q[:,2],1e-12,1-1e-12);nd=np.clip(nd,1e-12,1-1e-12);logit=(1-mix)*np.log(old/(1-old))+mix*.75*np.log(nd/(1-nd));d=1/(1+np.exp(-np.clip(logit,-80,80)));live=q[:,:2]/np.maximum(q[:,:2].sum(1,keepdims=True),1e-12);live=np.where(q[:,:2].sum(1,keepdims=True)>0,live,.5);return np.column_stack([live*(1-d[:,None]),d])
