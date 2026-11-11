"""A small, actual recurrent network with truncated backpropagation through time.

NumPy only. External supervision only. No labels are fed into the recurrent state.
"""
import numpy as np
from .model import instantaneous_features,softmax

RNN_VERSION='numpy-tanh-rnn-v1'
RNN_REFINED_VERSION='numpy-tanh-rnn-v1-split-decoder-v1'
RNN_SYMMETRIC_VERSION='numpy-tanh-rnn-v1-team-symmetric-v1'
RNN_DWELL_VERSION='numpy-tanh-rnn-v1-spell-balanced-dwell-v1'

def rnn_features(sequence,pitch):
    # Current observation and one backward difference only; no future context.
    return instantaneous_features(sequence,pitch).astype(np.float32)

def initialise(x,hidden=24,seed=42):
    rng=np.random.default_rng(seed);mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),.05);d=x.shape[1]
    return {'mean':mean,'scale':scale,'wx':(rng.normal(size=(d,hidden))/np.sqrt(d)).astype(np.float32),
            'wh':(np.eye(hidden)*.5).astype(np.float32),'bh':np.zeros(hidden,dtype=np.float32),
            'wy':(rng.normal(size=(hidden,3))*.05).astype(np.float32),'by':np.zeros(3,dtype=np.float32)}

def normalise(model,x):return np.clip((x-model['mean'])/model['scale'],-8,8)

def forward(model,x,h0=None):
    h=np.zeros(len(model['bh']),np.float32) if h0 is None else h0.copy();states=[]
    for row in x:
        h=np.tanh(row@model['wx']+h@model['wh']+model['bh']);states.append(h)
    hs=np.asarray(states);p=softmax(hs@model['wy']+model['by'])
    return p,hs

def loss_gradients(model,x,y,h0=None,weights=None,normalise_weights=True):
    p,hs=forward(model,x,h0);valid=(y>=0)&(y<3)
    indices=np.flatnonzero(valid)
    if weights is None:
        # Preserve the original arithmetic for reproducible existing checkpoints.
        n=max(1,int(valid.sum()))
        loss=float(-np.log(np.maximum(p[indices,y[valid]],1e-8)).sum()/n)
        dy=p.copy();dy[~valid]=0;dy[indices,y[valid]]-=1;dy/=n
    else:
        weights=np.asarray(weights,dtype=np.float32)
        if weights.shape!=y.shape or not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Invalid sample weights')
        w=weights*valid;n=max(1.0,float(w.sum()) if normalise_weights else float(valid.sum()))
        loss=float((-np.log(np.maximum(p[indices,y[valid]],1e-8))*w[valid]).sum()/n)
        dy=p.copy();dy[~valid]=0;dy[indices,y[valid]]-=1;dy*=w[:,None]/n
    grads={k:np.zeros_like(model[k]) for k in ('wx','wh','bh','wy','by')}
    grads['wy']=hs.T@dy;grads['by']=dy.sum(axis=0);dh_next=np.zeros_like(model['bh'])
    zero=np.zeros_like(model['bh']) if h0 is None else h0
    for t in range(len(x)-1,-1,-1):
        dh=dy[t]@model['wy'].T+dh_next;dz=dh*(1-hs[t]**2)
        previous=hs[t-1] if t else zero
        grads['wx']+=np.outer(x[t],dz);grads['wh']+=np.outer(previous,dz);grads['bh']+=dz
        dh_next=dz@model['wh'].T
    return loss,grads,hs[-1]

def train(sequences,epochs=5,hidden=24,seed=42,chunk=128,sample_weights=None,weight_normalisation='chunk'):
    # Standardisation fits training observations only, never validation/test/match inputs.
    if weight_normalisation not in ('chunk','global'):raise ValueError('Invalid sample weight normalisation')
    if sample_weights is not None and (len(sample_weights)!=len(sequences) or any(len(w)!=len(y) for w,(x,y) in zip(sample_weights,sequences))):raise ValueError('Sample weights must match training sequences')
    if sample_weights is not None and weight_normalisation=='global':
        total=sum(float(np.asarray(w)[(y>=0)&(y<3)].sum()) for w,(x,y) in zip(sample_weights,sequences))
        count=sum(int(((y>=0)&(y<3)).sum()) for x,y in sequences)
        mean_weight=total/max(1,count)
        if not np.isfinite(mean_weight) or mean_weight<=0:raise ValueError('Invalid global sample weights')
        sample_weights=[np.asarray(w,dtype=np.float32)/mean_weight for w in sample_weights]
    model=initialise(np.concatenate([x for x,y in sequences]),hidden,seed)
    moments={k:np.zeros_like(model[k]) for k in ('wx','wh','bh','wy','by')};variances={k:v.copy() for k,v in moments.items()}
    history=[];step=0
    for epoch in range(epochs):
        losses=[]
        for si,(raw,y) in enumerate(sequences):
            x=normalise(model,raw);h=None
            weights=None if sample_weights is None else sample_weights[si]
            for a in range(0,len(x),chunk):
                loss,grads,h=loss_gradients(model,x[a:a+chunk],y[a:a+chunk],h,
                                            None if weights is None else weights[a:a+chunk],
                                            normalise_weights=weight_normalisation=='chunk')
                losses.append(loss);norm=np.sqrt(sum(float((g*g).sum()) for g in grads.values()));step+=1
                for k,g in grads.items():
                    g*=min(1,5/max(norm,1e-8));moments[k]=.9*moments[k]+.1*g;variances[k]=.999*variances[k]+.001*g*g
                    model[k]-=.001*(moments[k]/(1-.9**step))/(np.sqrt(variances[k]/(1-.999**step))+1e-8)
        history.append(float(np.mean(losses)));print(f'RNN epoch {epoch+1}/{epochs}: training loss {history[-1]:.4f}',flush=True)
    return model,history

def predict(model,x,chunk=4096):
    x=normalise(model,x);out=[];h=None
    for a in range(0,len(x),chunk):
        p,states=forward(model,x[a:a+chunk],h);h=states[-1];out.append(p)
    return np.concatenate(out)

def observation_probabilities(model,sequence,pitch,symmetric=False,ball_missing=None):
    """Optional exact team-permutation averaging; no labels or fitted parameters."""
    p=predict(model,instantaneous_features(sequence,pitch,ball_missing=ball_missing))
    if not symmetric:return p
    swapped=predict(model,instantaneous_features(sequence,pitch,swap=True,ball_missing=ball_missing))
    return (p+swapped[:,[1,0,2]])*.5
