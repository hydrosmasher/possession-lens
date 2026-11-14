"""Pickle-free NumPy inference for the pinned classical-ML experiment."""
from pathlib import Path
import json
import numpy as np

def save(path,model,kind):
    arrays={};count=len(model.estimators_) if kind=='extra_trees' else len(model._predictors)
    meta={'version':'possession-trees-v1','kind':kind,'count':count,'features':int(model.n_features_in_),'classes':[int(c) for c in model.classes_],'baseline':float(model._baseline_prediction[0,0]) if kind=='hist_boost' else 0.}
    if meta['classes']!=[0,1]:raise ValueError('Expected binary classifier')
    for i in range(count):
        if kind=='extra_trees':
            t=model.estimators_[i].tree_;v=t.value[:,0,:];v=v/np.maximum(v.sum(1,keepdims=True),1e-15);left=t.children_left;right=t.children_right;feature=t.feature;threshold=t.threshold;missing=getattr(t,'missing_go_to_left',np.zeros(len(left),np.uint8))
        else:
            t=model._predictors[i][0].nodes
            if np.any(t['is_categorical']):raise ValueError('Categorical trees unsupported')
            left=np.where(t['is_leaf'],-1,t['left'].astype(np.int64));right=np.where(t['is_leaf'],-1,t['right'].astype(np.int64));feature=t['feature_idx'];threshold=t['num_threshold'];v=t['value'];missing=t['missing_go_to_left']
        arrays.update({f'{i}_{k}':np.asarray(a) for k,a in [('left',left),('right',right),('feature',feature),('threshold',threshold),('value',v),('missing',missing)]})
    np.savez_compressed(path,meta=np.array(json.dumps(meta)),**arrays)

def load(path):
    if Path(path).stat().st_size>100_000_000:raise ValueError('Oversized tree model')
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(str(a['meta']))
        if meta['version']!='possession-trees-v1' or meta['kind'] not in ('extra_trees','hist_boost') or meta['classes']!=[0,1] or not 1<=meta['count']<=512 or not 1<=meta['features']<=512:raise ValueError('Invalid tree metadata')
        trees=[]
        for i in range(meta['count']):
            t={k:a[f'{i}_{k}'].copy() for k in ('left','right','feature','threshold','value','missing')};n=len(t['left']);active=t['left']>=0
            if not n or n>100000 or any(t[k].shape!=(n,) for k in ('left','right','feature','threshold','missing')) or t['value'].shape!=((n,2) if meta['kind']=='extra_trees' else (n,)) or not all(np.isfinite(v).all() for v in t.values()):raise ValueError('Invalid tree arrays')
            if np.any(t['left'][active]>=n) or np.any(t['right'][active]<0) or np.any(t['right'][active]>=n) or np.any(t['feature'][active]<0) or np.any(t['feature'][active]>=meta['features']):raise ValueError('Invalid tree structure')
            trees.append(t)
    return {'meta':meta,'trees':trees}

def predict(model,x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=model['meta']['features']:raise ValueError('Invalid ML features')
    # sklearn ExtraTrees casts input tofloat32 before comparing double thresholds.
    if model['meta']['kind']=='extra_trees':x=x.astype(np.float32)
    result=np.zeros((len(x),2)) if model['meta']['kind']=='extra_trees' else np.full(len(x),model['meta']['baseline'])
    for t in model['trees']:
        nodes=np.zeros(len(x),np.int64)
        for depth in range(129):
            active=t['left'][nodes]>=0
            if not active.any():break
            if depth==128:raise ValueError('Cyclic or overdeep tree')
            ix=np.flatnonzero(active);node=nodes[ix];value=x[ix,t['feature'][node]];left=np.where(np.isnan(value),t['missing'][node].astype(bool),value<=t['threshold'][node]);nodes[ix]=np.where(left,t['left'][node],t['right'][node])
        result+=t['value'][nodes]
    if model['meta']['kind']=='extra_trees':result/=len(model['trees'])
    else:
        q=1/(1+np.exp(-np.clip(result,-80,80)));result=np.column_stack([1-q,q])
    return np.column_stack([result,np.zeros(len(x))])

def symmetric(model,s,p,mask=None):
    from .interaction_context import observations
    a=predict(model,observations(s,p,mask));b=predict(model,observations(s,p,mask,True));return (a+b[:,[1,0,2]])*.5
