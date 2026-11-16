"""Differentiable local spell-overlap surrogate for external-only training."""
import numpy as np,torch
from torch.nn import functional as F

def local_spell_iou(logits,y):
    if logits.ndim!=3 or logits.shape[-1]!=3 or y.shape!=logits.shape[:2]:raise ValueError('Unexpected spell-loss tensors')
    labels=y.detach().cpu().numpy();indices=[]
    for batch,row in enumerate(labels):
        edges=np.r_[0,np.flatnonzero(row[1:]!=row[:-1])+1,len(row)]
        for j in range(1,len(edges)-2):
            a,b=int(edges[j]),int(edges[j+1]);k=int(row[a]);leftk=int(row[edges[j-1]]);rightk=int(row[edges[j+1]])
            if k not in (0,1) or leftk!=1-k or rightk!=1-k:continue
            left=(int(edges[j-1])+a)//2;right=(b+int(edges[j+2])+1)//2;indices.append((batch,k,left,a,b,right))
    if not indices:return logits.sum()*0
    index=torch.tensor(indices,dtype=torch.long,device=logits.device);batch,k,left,a,b,right=index.T;prob=logits[:,:,:2].softmax(-1);prefix=F.pad(prob.cumsum(1),(0,0,1,0));intersection=prefix[batch,b,k]-prefix[batch,a,k];mass=prefix[batch,right,k]-prefix[batch,left,k];union=(b-a).to(logits.dtype)+mass-intersection
    return (1-(intersection+1e-6)/(union+1e-6)).mean()
