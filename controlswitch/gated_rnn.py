"""Optional PyTorch gated sequence labelling; no labels enter inference.

Inference uses overlap-save windows with exactly the same finite context as
training. Bidirectional candidates are offline, not live possession forecasts.
Checkpoints contain NumPy arrays and JSON metadata, never pickle objects.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .interaction_context import observations
from .model import decode
VERSION='torch-gated-possession-v1'
CORE=256
CONTEXT=64

class LockedDropout(nn.Module):
    def __init__(self,p):
        super().__init__();self.p=p
    def forward(self,x):
        if not self.training or not self.p:return x
        mask=x.new_empty(x.shape[0],1,x.shape[2]).bernoulli_(1-self.p)/(1-self.p)
        return x*mask

class GatedPossession(nn.Module):
    def __init__(self,cell='gru',layers=1,bidirectional=False,hidden=32,projection=48,dropout=.15,boundary=False,input_features=130):
        super().__init__()
        if cell not in ('gru','lstm') or layers not in (1,2) or hidden not in (32,64) or projection not in (48,64) or input_features not in (130,416,504,508) or not 0<=dropout<1:raise ValueError('Unsupported gated architecture')
        self.config=dict(cell=cell,layers=layers,bidirectional=bool(bidirectional),hidden=hidden,projection=projection,dropout=dropout,boundary=bool(boundary),input_features=input_features)
        self.embedding=nn.Sequential(nn.Linear(input_features,projection),nn.LayerNorm(projection),nn.GELU(),LockedDropout(dropout))
        cls=nn.GRU if cell=='gru' else nn.LSTM
        self.recurrent=cls(projection,hidden,num_layers=layers,batch_first=True,bidirectional=bidirectional,dropout=dropout if layers>1 else 0)
        width=hidden*(2 if bidirectional else 1)
        self.norm=nn.LayerNorm(width);self.dropout=LockedDropout(dropout);self.classifier=nn.Linear(width,3)
        self.boundary_head=nn.Linear(width,1) if boundary else None
        self.reset_parameters()
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module,nn.Linear):nn.init.xavier_uniform_(module.weight);nn.init.zeros_(module.bias)
        for name,p in self.recurrent.named_parameters():
            if 'weight_ih' in name:
                for gate in p.chunk(3 if self.config['cell']=='gru' else 4,0):nn.init.xavier_uniform_(gate)
            elif 'weight_hh' in name:
                for gate in p.chunk(3 if self.config['cell']=='gru' else 4,0):nn.init.orthogonal_(gate)
            else:
                nn.init.zeros_(p)
                if self.config['cell']=='lstm' and 'bias_ih' in name:p.data[self.config['hidden']:2*self.config['hidden']]=1
    def representation(self,x):
        h,_=self.recurrent(self.embedding(x));return self.dropout(self.norm(h))
    def forward(self,x):
        h,_=self.recurrent(self.embedding(x));h=self.dropout(self.norm(h))
        return self.classifier(h),None if self.boundary_head is None else self.boundary_head(h).squeeze(-1)

def scaling(sequences):
    # Streaming moments from training observations only; no validation statistics.
    count=0;features=sequences[0][0].shape[1];total=np.zeros(features,np.float64);sq=np.zeros(features,np.float64)
    for item in sequences:
        x=item[0];count+=len(x);total+=x.sum(0,dtype=np.float64);sq+=np.einsum('ij,ij->j',x.astype(np.float64),x.astype(np.float64))
    mean=total/count;scale=np.maximum(np.sqrt(np.maximum(0,sq/count-mean*mean)),.05)
    return mean.astype(np.float32),scale.astype(np.float32)

def normalise(x,mean,scale):return np.clip((x-mean)/scale,-8,8).astype(np.float32)

def make_window(x,y,start,boundary=None,valid_boundary=None):
    """Fixed windows, central core loss only; context and padding are unlabelled."""
    n=len(x);a=max(0,start-CONTEXT);b=min(n,start+CORE+CONTEXT);offset=a-(start-CONTEXT)
    xx=np.zeros((CORE+2*CONTEXT,x.shape[1]),np.float32);yy=np.full(len(xx),-1,np.int64);xx[offset:offset+b-a]=x[a:b]
    size=min(CORE,n-start);yy[CONTEXT:CONTEXT+size]=y[start:start+size]
    target=np.zeros(len(xx),np.float32);weight=np.zeros(len(xx),np.float32)
    if boundary is not None:
        target[CONTEXT:CONTEXT+size]=boundary[start:start+size];weight[CONTEXT:CONTEXT+size]=valid_boundary[start:start+size]
    return xx,yy,target,weight

def predict(model,x,mean,scale,batch=32):
    model.eval();probs=[];events=[];x=normalise(x,mean,scale);unknown=np.full(len(x),-1,np.int64)
    with torch.inference_mode():
        for a in range(0,len(x),CORE*batch):
            starts=list(range(a,min(len(x),a+CORE*batch),CORE));windows=[make_window(x,unknown,s)[0] for s in starts]
            logits,head=model(torch.from_numpy(np.stack(windows)));p=logits.softmax(-1).cpu().numpy();e=None if head is None else head.sigmoid().cpu().numpy()
            for i,start in enumerate(starts):
                size=min(CORE,len(x)-start);probs.append(p[i,CONTEXT:CONTEXT+size]);events.append(np.full(size,.5,np.float32) if e is None else e[i,CONTEXT:CONTEXT+size])
    return np.concatenate(probs),np.concatenate(events)

def gated_observations(model,sequence,pitch,mask=None,swap=False):
    width=model.config.get('input_features',130)
    if width==130:return observations(sequence,pitch,mask,swap)
    if width==416:
        from .relative_state import relative_features,swap_relative
        x=relative_features(sequence,pitch,mask);return swap_relative(x) if swap else x
    if width==504:
        from .collective_state import collective_state,swap_collective
        x=collective_state(sequence,pitch,mask);return swap_collective(x) if swap else x
    if width==508:
        from .joint_state import joint_features,swap_joint
        x=joint_features(sequence,pitch,mask);return swap_joint(x) if swap else x
    raise ValueError('Unsupported gated feature layout')

def symmetric(model,sequence,pitch,mean,scale,mask=None):
    a,ea=predict(model,gated_observations(model,sequence,pitch,mask),mean,scale)
    b,eb=predict(model,gated_observations(model,sequence,pitch,mask,True),mean,scale)
    return (a+b[:,[1,0,2]])*.5,(ea+eb)*.5

def save_checkpoint(path,model,mean,scale,penalty,strength=0):
    meta={'version':VERSION,'config':model.config,'core':CORE,'context':CONTEXT,'penalty':float(penalty),'boundary_strength':float(strength),'symmetric':True}
    np.savez_compressed(path,version=np.array(VERSION),meta=np.array(json.dumps(meta)),mean=mean,scale=scale,**{'weight__'+k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})

def load_checkpoint(path):
    if Path(path).stat().st_size>5_000_000:raise ValueError('Oversized gated checkpoint')
    with np.load(path,allow_pickle=False) as d:
        meta=json.loads(str(d['meta']));mean=d['mean'];scale=d['scale']
        if meta['version']!=VERSION or meta['core']!=CORE or meta['context']!=CONTEXT or meta['symmetric'] is not True:raise ValueError('Unsupported gated inference policy')
        features=meta['config'].get('input_features',130)
        if mean.shape!=(features,) or scale.shape!=(features,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid gated scaling')
        if not np.isfinite(meta['penalty']) or meta['penalty']<0 or not np.isfinite(meta['boundary_strength']) or not 0<=meta['boundary_strength']<=1:raise ValueError('Invalid gated decoder')
        model=GatedPossession(**meta['config']);weights={}
        for key,v in model.state_dict().items():
            arr=d['weight__'+key]
            if arr.shape!=tuple(v.shape) or not np.isfinite(arr).all():raise ValueError('Invalid gated weights')
            weights[key]=torch.from_numpy(arr.copy())
        model.load_state_dict(weights,strict=True);model.eval()
    return {'model':model,'mean':mean.copy(),'scale':scale.copy(),**meta,'gated':True}

def infer(frozen,sequence,pitch,mask=None):
    p,b=symmetric(frozen['model'],sequence,pitch,frozen['mean'],frozen['scale'],mask)
    return decode(p,frozen['penalty'],b if frozen['boundary_strength'] else None,frozen['boundary_strength']),p
