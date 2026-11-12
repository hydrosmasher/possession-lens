"""Stacked NumPy RNN, minibatch truncated BPTT and explicit regularisation.

Training uses shuffled 128-frame windows with state reset at each window.
Inference carries state across chunks; no labels are accepted at inference.
"""
import numpy as np
from .model import softmax
VERSION='numpy-stacked-tanh-rnn-v1'

def initialise(x,sizes=(24,24),seed=42):
 if not sizes or len(sizes)>3 or any(h<1 or h>128 for h in sizes):raise ValueError('Invalid hidden sizes')
 rng=np.random.default_rng(seed);m={'mean':x.mean(0).astype(np.float32),'scale':np.maximum(x.std(0),.05).astype(np.float32),'sizes':np.array(sizes)};d=x.shape[-1]
 for l,h in enumerate(sizes):
  m[f'wx{l}']=(rng.normal(size=(d,h))/np.sqrt(d)).astype(np.float32);m[f'wh{l}']=(np.eye(h)*.5).astype(np.float32);m[f'b{l}']=np.zeros(h,np.float32);d=h
 m['wo']=(rng.normal(size=(d,3))*.05).astype(np.float32);m['bo']=np.zeros(3,np.float32)
 return m

def parameters(m):return [k for k in m if k not in ('mean','scale','sizes')]
def normalise(m,x):return np.clip((x-m['mean'])/m['scale'],-8,8)

def forward(m,x,h0=None,dropout=0.,rng=None):
 if not 0<=dropout<1:raise ValueError('Invalid dropout')
 sizes=m['sizes'];b=x.shape[0];states=[np.zeros((b,int(h)),np.float32) for h in sizes] if h0 is None else [h.copy() for h in h0]
 masks=[np.ones_like(h) if not dropout else (rng.random(h.shape)>=dropout).astype(np.float32)/(1-dropout) for h in states]
 hs=[[] for _ in sizes];ins=[[] for _ in sizes];outputs=[]
 for t in range(x.shape[1]):
  z=x[:,t]
  for l in range(len(sizes)):
   ins[l].append(z);states[l]=np.tanh(z@m[f'wx{l}']+states[l]@m[f'wh{l}']+m[f'b{l}']);hs[l].append(states[l]);z=states[l]*masks[l]
  outputs.append(z)
 out=np.stack(outputs,axis=1);p=softmax((out@m['wo']+m['bo']).reshape(-1,3)).reshape(b,x.shape[1],3)
 return p,([np.stack(h,axis=1) for h in hs],[np.stack(i,axis=1) for i in ins],masks,out,states)

def loss_gradients(m,x,y,dropout=0.,rng=None,l2=0.,sample_weight=None):
 p,(hs,ins,masks,out,states)=forward(m,x,dropout=dropout,rng=rng)
 valid=(y>=0)&(y<3);bi,ti=np.where(valid)
 weight=np.ones(y.shape,np.float32) if sample_weight is None else np.asarray(sample_weight,dtype=np.float32)
 if weight.shape!=y.shape or not np.isfinite(weight).all() or (weight<0).any():raise ValueError('Invalid frame loss weights')
 weight=np.where(valid,weight,0);n=max(1e-8,float(weight.sum()))
 loss=float((-np.log(np.maximum(p[bi,ti,y[valid]],1e-8))*weight[valid]).sum()/n)
 dy=p.copy();dy[~valid]=0;dy[bi,ti,y[valid]]-=1;dy*=weight[:,:,None]/n
 g={k:np.zeros_like(m[k]) for k in parameters(m)};g['wo']=out.reshape(-1,out.shape[-1]).T@dy.reshape(-1,3);g['bo']=dy.sum((0,1))
 future=[np.zeros_like(h) for h in states]
 for t in range(x.shape[1]-1,-1,-1):
  upper=dy[:,t]@m['wo'].T
  for l in range(len(hs)-1,-1,-1):
   dz=(upper*masks[l]+future[l])*(1-hs[l][:,t]**2);previous=hs[l][:,t-1] if t else np.zeros_like(states[l])
   g[f'wx{l}']+=ins[l][:,t].T@dz;g[f'wh{l}']+=previous.T@dz;g[f'b{l}']+=dz.sum(0)
   future[l]=dz@m[f'wh{l}'].T;upper=dz@m[f'wx{l}'].T
 for k in g:
  if k.startswith('w'):loss+=.5*l2*float((m[k]**2).sum());g[k]+=l2*m[k]
 return loss,g

def predict(m,x,chunk=4096):
 x=normalise(m,x);result=[];h=None
 for a in range(0,len(x),chunk):
  p,cache=forward(m,x[None,a:a+chunk],h0=h);h=cache[-1];result.append(p[0])
 return np.concatenate(result)

def windows(sequences,length=128):
 xs=[];ys=[]
 for item in sequences:
  x,y=item[:2]
  for a in range(0,len(x),length):
   b=min(a+length,len(x));z=np.zeros((length,x.shape[1]),np.float32);v=np.full(length,-1,np.int8);z[:b-a]=x[a:b];v[:b-a]=y[a:b];xs.append(z);ys.append(v)
 return np.array(xs),np.array(ys)

def cross_entropy(m,x,y,batch=32):
 total=0.;n=0
 for a in range(0,len(x),batch):
  p,_=forward(m,normalise(m,x[a:a+batch]));valid=(y[a:a+batch]>=0)&(y[a:a+batch]<3);bi,ti=np.where(valid);v=y[a:a+batch][valid]
  total+=float(-np.log(np.maximum(p[bi,ti,v],1e-8)).sum());n+=len(v)
 return total/max(1,n)

def train(sequences,validation,sizes=(24,24),epochs=16,dropout=.15,l2=.0001,patience=3,seed=42,max_windows=1024,on_epoch=None,train_probe_sequences=None,on_epoch_model=None):
 if epochs<1 or patience<1 or max_windows<1:raise ValueError('Invalid training limits')
 m=initialise(np.concatenate([item[0] for item in sequences]),sizes,seed);x,y=windows(sequences);vx,vy=windows(validation)
 weighted=any(len(item)==3 for item in sequences)
 if weighted and not all(len(item)==3 for item in sequences):raise ValueError('Mixed weighted and unweighted training sequences')
 weights=windows([(item[2][:,None],item[1]) for item in sequences])[0][:,:,0] if weighted else None
 px,py=windows(train_probe_sequences) if train_probe_sequences is not None else (x,y)
 rng=np.random.default_rng(seed);probe=np.random.default_rng(101);tr=probe.choice(len(px),min(64,len(px)),replace=False);va=probe.choice(len(vx),min(128,len(vx)),replace=False)
 moments={k:np.zeros_like(m[k]) for k in parameters(m)};variances={k:v.copy() for k,v in moments.items()};history=[];best=None;best_loss=float('inf');stale=0;step=0
 for epoch in range(1,epochs+1):
  order=rng.permutation(len(x))[:min(len(x),max_windows)];losses=[]
  for a in range(0,len(order),16):
   ix=order[a:a+16];loss,g=loss_gradients(m,normalise(m,x[ix]),y[ix],dropout,rng,l2,sample_weight=weights[ix] if weighted else None);losses.append(loss);step+=1;norm=np.sqrt(sum(float((v*v).sum()) for v in g.values()))
   for k,v in g.items():
    v*=min(1,5/max(norm,1e-8));moments[k]=.9*moments[k]+.1*v;variances[k]=.999*variances[k]+.001*v*v
    m[k]-=.001*(moments[k]/(1-.9**step))/(np.sqrt(variances[k]/(1-.999**step))+1e-8)
  train_ce=cross_entropy(m,px[tr],py[tr]);val_ce=cross_entropy(m,vx[va],vy[va]);row={'iteration':epoch,'sampled_training_windows':len(order),'optimiser_steps':step,'training_objective':float(np.mean(losses)),'train_probe_cross_entropy':train_ce,'validation_probe_cross_entropy':val_ce,'generalisation_gap':val_ce-train_ce};history.append(row)
  if val_ce<best_loss-1e-4:best_loss=val_ce;best={k:v.copy() for k,v in m.items()};stale=0;best_epoch=epoch
  else:stale+=1
  if on_epoch:on_epoch(row)
  if on_epoch_model:on_epoch_model(row,{k:v.copy() for k,v in m.items()})
  print(f'Stack {len(sizes)} iteration {epoch}: train CE {train_ce:.4f}; validation CE {val_ce:.4f}; gap {val_ce-train_ce:.4f}',flush=True)
  if stale>=patience:break
 return best,history,best_epoch
