"""060: small shared player encoder feeding the retained bidirectional GRU."""
from pathlib import Path
import json,numpy as np,torch
from torch import nn
from .gated_rnn import GatedPossession,LockedDropout,normalise,make_window,CORE,CONTEXT
from .joint_state import joint_features,swap_joint
from .player_set_state import player_nodes,transform_nodes,NODES,FEATURES
VERSION='player-set-gru-060-v1'

class PlayerSetPossession(nn.Module):
    def __init__(self,base_config,pooling='per_team'):
        super().__init__()
        expected={'cell':'gru','layers':2,'bidirectional':True,'hidden':64,'projection':64,'dropout':.25,'boundary':True,'input_features':508}
        if base_config!=expected:raise ValueError('Expected frozen508feature64unit GRU architecture')
        if pooling not in ('per_team','global'):raise ValueError('Unsupported player pooling')
        self.pooling=pooling
        self.base=GatedPossession(**base_config);self.encoder=nn.Sequential(nn.Linear(20,32),nn.LayerNorm(32),nn.GELU(),nn.Linear(32,32),nn.GELU());self.attention=nn.Linear(32,1);self.node_projection=nn.Linear(64,64);self.node_dropout=LockedDropout(.25);self.base_config=base_config;self.node_drop=.1
        for layer in self.encoder:
            if isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.attention.weight);nn.init.zeros_(self.attention.bias);nn.init.zeros_(self.node_projection.weight);nn.init.zeros_(self.node_projection.bias)
    def forward(self,x,nodes):
        if nodes.shape!=(*x.shape[:2],2,NODES,FEATURES):raise ValueError('Invalid batched node shape')
        valid=nodes[...,20]>.5
        if self.training and self.node_drop:
            keep=torch.rand((len(x),1,2,NODES),device=x.device)>=self.node_drop;valid=valid&keep
        e=self.encoder(nodes[...,:20]);scores=self.attention(e).squeeze(-1).masked_fill(~valid,-1e4)
        if self.pooling=='global':
            weight=scores.flatten(-2).softmax(-1).reshape_as(scores)*valid;weight=weight/weight.sum((-1,-2),keepdim=True).clamp_min(1e-8)
        else:weight=scores.softmax(-1)*valid;weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
        pool=(e*weight.unsqueeze(-1)).sum(-2).reshape(*x.shape[:2],64);embedding=self.base.embedding(x)+self.node_dropout(self.node_projection(pool));h,_=self.base.recurrent(embedding);h=self.base.dropout(self.base.norm(h));return self.base.classifier(h),self.base.boundary_head(h).squeeze(-1)

def warm_start(old,pooling='per_team'):
    m=PlayerSetPossession(old['model'].config.copy(),pooling);m.base.load_state_dict(old['model'].state_dict());return m

def node_window(nodes,start):
    a=max(0,start-CONTEXT);b=min(len(nodes),start+CORE+CONTEXT);offset=a-(start-CONTEXT);out=np.zeros((CORE+2*CONTEXT,2,NODES,FEATURES),np.float32);out[offset:offset+b-a]=nodes[a:b];return out

def predict(model,x,nodes,mean,scale,batch=16):
    model.eval();x=normalise(x,mean,scale);unknown=np.full(len(x),-1,np.int64);probs=[];events=[]
    with torch.inference_mode():
        for a in range(0,len(x),CORE*batch):
            starts=list(range(a,min(len(x),a+CORE*batch),CORE));xx=np.stack([make_window(x,unknown,z)[0] for z in starts]);nnodes=np.stack([node_window(nodes,z) for z in starts]);logits,head=model(torch.from_numpy(xx),torch.from_numpy(nnodes));p=logits.softmax(-1).numpy();e=head.sigmoid().numpy()
            for i,z in enumerate(starts):
                size=min(CORE,len(x)-z);probs.append(p[i,CONTEXT:CONTEXT+size]);events.append(e[i,CONTEXT:CONTEXT+size])
    return np.concatenate(probs),np.concatenate(events)

def symmetric(model,s,pitch,mean,scale,mask=None):
    x=joint_features(s,pitch,mask);nodes=player_nodes(s,pitch,mask);a,ea=predict(model,x,nodes,mean,scale);b,eb=predict(model,swap_joint(x),transform_nodes(nodes,1),mean,scale);return (a+b[:,[1,0,2]])*.5,(ea+eb)*.5

def save_checkpoint(path,model,mean,scale):
    meta={'version':VERSION,'base_config':model.base_config,'core':CORE,'context':CONTEXT,'nodes_per_team':NODES,'node_features':FEATURES,'node_width':32,'node_drop':.1,'symmetric':True,'pooling':model.pooling};np.savez_compressed(path,meta=np.array(json.dumps(meta)),mean=mean,scale=scale,**{'weight__'+k:v.detach().numpy() for k,v in model.state_dict().items()})

def load_checkpoint(path):
    if Path(path).stat().st_size>5_000_000:raise ValueError('Oversized set-RNN checkpoint')
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(str(a['meta']));expected={'version':VERSION,'core':CORE,'context':CONTEXT,'nodes_per_team':NODES,'node_features':FEATURES,'node_width':32,'node_drop':.1,'symmetric':True}
        if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Unsupported set-RNN policy')
        mean=a['mean'].copy();scale=a['scale'].copy()
        if mean.shape!=(508,) or scale.shape!=(508,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid set-RNN scaling')
        model=PlayerSetPossession(meta['base_config'],meta.get('pooling','per_team'));weights={}
        for k,v in model.state_dict().items():
            w=a['weight__'+k]
            if w.shape!=tuple(v.shape) or not np.isfinite(w).all():raise ValueError('Invalid set-RNN weights')
            weights[k]=torch.from_numpy(w.copy())
        model.load_state_dict(weights);model.eval()
    return {'model':model,'mean':mean,'scale':scale,'player_set':True,'meta':meta}
