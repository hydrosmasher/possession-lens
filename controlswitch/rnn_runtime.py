"""Frozen recurrent inference. No fitting, labels or tuning are accepted here."""
import numpy as np
from .rnn import (RNN_VERSION,RNN_REFINED_VERSION,RNN_SYMMETRIC_VERSION,RNN_DWELL_VERSION,observation_probabilities)
from .rnn_boundaries import BOUNDARY_VERSION,dwell_decode,invariant_boundary_features
from .model import probabilities,decode

VERSIONS=(RNN_VERSION,RNN_REFINED_VERSION,RNN_SYMMETRIC_VERSION,RNN_DWELL_VERSION,BOUNDARY_VERSION)

def load_frozen(path):
    with np.load(path,allow_pickle=False) as d:
        version=str(d['version'])
        from .rnn_deep import VERSION as DEEP_VERSION,parameters
        if version==DEEP_VERSION:
            sizes=d['sizes'];model={k:d[k] for k in ('mean','scale','sizes','wo','bo')}
            if sizes.ndim!=1 or not 1<=len(sizes)<=3 or not np.issubdtype(sizes.dtype,np.integer) or ((sizes<1)|(sizes>128)).any():raise ValueError('Invalid stacked recurrent sizes')
            for l in range(len(sizes)):
                for prefix in ('wx','wh','b'):model[f'{prefix}{l}']=d[f'{prefix}{l}']
            width=len(model['mean'])
            if model['scale'].shape!=(width,) or (model['scale']<=0).any():raise ValueError('Invalid stacked scaling')
            for l,h in enumerate(sizes):
                if model[f'wx{l}'].shape!=(width,h) or model[f'wh{l}'].shape!=(h,h) or model[f'b{l}'].shape!=(h,):raise ValueError('Invalid stacked weight shape')
                width=int(h)
            if model['wo'].shape!=(width,3) or model['bo'].shape!=(3,) or not all(np.isfinite(v).all() for v in model.values()):raise ValueError('Invalid stacked weights')
            penalty=float(d['transition_penalty'])
            if not np.isfinite(penalty) or penalty<0 or not bool(d['symmetric']):raise ValueError('Invalid stacked decoder')
            return {'model':model,'version':version,'symmetric':True,'penalty':penalty,'stop_penalty':None,'min_frames':None,'head':None,'strength':0.,'deep':True}
        if version not in VERSIONS:raise ValueError('Unsupported frozen model version')
        model={k:d[k] for k in ('mean','scale','wx','wh','bh','wy','by')}
        if not all(np.isfinite(v).all() for v in model.values()):raise ValueError('Nonfinite recurrent model')
        symmetric=version in (RNN_SYMMETRIC_VERSION,RNN_DWELL_VERSION,BOUNDARY_VERSION)
        if symmetric and ('symmetric' not in d.files or not bool(d['symmetric'])):raise ValueError('Missing symmetry setting')
        penalty=float(d['transition_penalty']) if 'transition_penalty' in d.files else 0.
        stop=float(d['stop_penalty']) if 'stop_penalty' in d.files else None
        if version==RNN_REFINED_VERSION and stop is None:raise ValueError('Missing stop penalty')
        if not np.isfinite(penalty) or penalty<0 or (stop is not None and (not np.isfinite(stop) or stop<0)):raise ValueError('Invalid frozen penalties')
        minimum=int(d['min_frames']) if version in (RNN_DWELL_VERSION,BOUNDARY_VERSION) else None
        if minimum is not None and not 1<=minimum<=100:raise ValueError('Invalid frozen dwell floor')
        head={k:d['event_'+k] for k in ('mean','scale','weights')} if version==BOUNDARY_VERSION else None
        strength=float(d['boundary_strength']) if head is not None else 0.
        if not np.isfinite(strength) or strength<0:raise ValueError('Invalid frozen boundary strength')
    return {'model':model,'version':version,'symmetric':symmetric,'penalty':penalty,'stop_penalty':stop,
            'min_frames':minimum,'head':head,'strength':strength}

def infer_frozen(frozen,sequence,pitch,mask=None):
    if frozen.get('deep'):
        from .rnn_capacity import observation_probabilities as deep_probabilities
        p=deep_probabilities(frozen['model'],sequence,pitch,mask)
    else:p=observation_probabilities(frozen['model'],sequence,pitch,frozen['symmetric'],mask)
    if frozen['head'] is not None:
        b=probabilities(frozen['head'],invariant_boundary_features(sequence,pitch,p,mask))[:,1]
        labels=dwell_decode(p,frozen['penalty'],b,frozen['strength'],frozen['min_frames'])
    elif frozen['min_frames'] is not None:
        labels=dwell_decode(p,frozen['penalty'],np.full(len(p),.5),0,frozen['min_frames'])
    else:labels=decode(p,frozen['penalty'],stop_penalty=frozen['stop_penalty'])
    return labels,p
