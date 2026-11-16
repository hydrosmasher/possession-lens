"""069: finite-context temporal control expert; not the deployed model."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from .gated_rnn import LockedDropout,normalise,make_window,CORE,CONTEXT
from .joint_state import joint_features,swap_joint
VERSION='local-temporal-control-069-v1'

class LocalTemporal(nn.Module):
    def __init__(self):
        super().__init__();self.project=nn.Sequential(nn.Linear(508,64),nn.LayerNorm(64),nn.GELU());self.convs=nn.ModuleList([nn.Conv1d(64,64,5,padding=2*d,dilation=d) for d in (1,4,16)]);self.norms=nn.ModuleList([nn.LayerNorm(64) for _ in range(3)]);self.dropout=LockedDropout(.2);self.classifier=nn.Linear(64,2)
        for m in self.modules():
            if isinstance(m,(nn.Linear,nn.Conv1d)):nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x):
        h=self.project(x)
        for conv,norm in zip(self.convs,self.norms):
            residual=torch.nn.functional.gelu(norm(conv(h.transpose(1,2)).transpose(1,2)));h=h+self.dropout(residual)
        return self.classifier(h)


def predict(model,x,mean,scale,batch=16):
    model.eval();x=normalise(x,mean,scale);unknown=np.full(len(x),-1,np.int64);result=[]
    with torch.inference_mode():
        for first in range(0,len(x),CORE*batch):
            starts=list(range(first,min(len(x),first+CORE*batch),CORE));windows=np.stack([make_window(x,unknown,a)[0] for a in starts]);p=model(torch.from_numpy(windows)).softmax(-1).numpy()
            for i,a in enumerate(starts):result.append(p[i,CONTEXT:CONTEXT+min(CORE,len(x)-a)])
    return np.concatenate(result)


def symmetric(bundle,s,pitch,mask=None):
    x=joint_features(s,pitch,mask);a=predict(bundle['model'],x,bundle['mean'],bundle['scale']);b=predict(bundle['model'],swap_joint(x),bundle['mean'],bundle['scale']);return (a+b[:,::-1])*.5


def save(path,model,mean,scale):
    meta={'version':VERSION,'features':508,'width':64,'kernel':5,'dilations':[1,4,16],'radius_frames':42,'dropout':.2,'core':CORE,'context':CONTEXT,'classes':2}
    np.savez_compressed(path,meta=np.array(json.dumps(meta)),mean=mean,scale=scale,**{'weight__'+k:v.detach().numpy() for k,v in model.state_dict().items()})


def load(path):
    if Path(path).stat().st_size>5_000_000:raise ValueError('Oversized temporal checkpoint')
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(str(a['meta']));expected={'version':VERSION,'features':508,'width':64,'kernel':5,'dilations':[1,4,16],'radius_frames':42,'dropout':.2,'core':CORE,'context':CONTEXT,'classes':2}
        if meta!=expected:raise ValueError('Unsupported temporal checkpoint')
        mean=a['mean'].copy();scale=a['scale'].copy()
        if mean.shape!=(508,) or scale.shape!=(508,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid temporal scaling')
        model=LocalTemporal();state={}
        if set(a.files)!={'meta','mean','scale'}|{'weight__'+k for k in model.state_dict()}:raise ValueError('Unexpected temporal weights')
        for k,v in model.state_dict().items():
            w=a['weight__'+k]
            if w.shape!=tuple(v.shape) or not np.isfinite(w).all():raise ValueError('Invalid temporal weights')
            state[k]=torch.from_numpy(w.copy())
        model.load_state_dict(state);model.eval()
    return {'model':model,'mean':mean,'scale':scale}


def locked_dead_decode(live,event,frozen,penalty=12.):
    live=np.asarray(live,float);event=np.asarray(event,float);frozen=np.asarray(frozen);n=len(frozen)
    if live.shape!=(n,2) or event.shape!=(n,) or not np.isfinite(live).all() or not np.isfinite(event).all() or (live<0).any() or (live.sum(1)<=0).any() or ((event<0)|(event>1)).any() or not np.isin(frozen,[0,1,2]).all():raise ValueError('Invalid locked-D decoder inputs')
    costs=penalty*np.clip(2-2*event,.25,2);logp=np.log(np.maximum(live,1e-8));out=frozen.copy();edges=np.flatnonzero(np.diff(np.r_[False,frozen<2,False]))
    for a,b in edges.reshape(-1,2):
        values=logp[a].copy();back=np.zeros((b-a,2),np.int8)
        for t in range(a+1,b):
            choices=values[:,None]-np.array([[0,costs[t]],[costs[t],0]]);back[t-a]=choices.argmax(0);values=choices.max(0)+logp[t]
        state=int(values.argmax())
        for t in range(b-1,a-1,-1):
            out[t]=state
            if t>a:state=int(back[t-a,state])
    return out
