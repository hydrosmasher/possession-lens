"""External-only boundary features aligned to frozen052 recurrent emissions."""
import numpy as np
from .gated_rnn import symmetric
from .ranked_pipeline import infer as base_infer
from .boundary_had import boundary_observations,adaptive_decode
from .model import probabilities,rolling,shifted

def evidence(bundle,s,pitch,mask=None):
    _,p=base_infer(bundle['base'],s,pitch,mask);m=bundle['gru'];g,e=symmetric(m['model'],s,pitch,m['mean'],m['scale'],mask)
    g=g[:,:2]/np.maximum(g[:,:2].sum(1,keepdims=True),1e-12);h=p[:,:2]/np.maximum(p[:,:2].sum(1,keepdims=True),1e-12);live=.5*h+.5*g;d=p[:,2];q=np.column_stack([live*(1-d[:,None]),d]);x=boundary_observations(s,pitch,q,mask)
    return q,x,e

def event_features(x,event):
    x=np.asarray(x);event=np.asarray(event)
    if x.ndim!=2 or x.shape[1]!=116 or event.shape!=(len(x),) or not len(x) or not np.isfinite(x).all() or not np.isfinite(event).all() or ((event<0)|(event>1)).any():raise ValueError('Invalid recurrent boundary evidence')
    e=event[:,None];return np.column_stack([x,event,rolling(e,12)[:,0],rolling(e,25)[:,0],np.abs(shifted(event,12)-shifted(event,-12))]).astype(np.float32)

def decode_head(head,q,x,event):
    width=len(head['mean']);features=x if width==116 else event_features(x,event) if width==120 else None
    if features is None:raise ValueError('Unsupported aligned boundary layout')
    return adaptive_decode(q,probabilities(head,features)[:,1],1.)
