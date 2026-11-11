"""Frozen recurrent inference. No fitting, labels or tuning are accepted here."""
import numpy as np
from .rnn import (RNN_VERSION,RNN_REFINED_VERSION,RNN_SYMMETRIC_VERSION,RNN_DWELL_VERSION,observation_probabilities)
from .rnn_boundaries import BOUNDARY_VERSION,dwell_decode,invariant_boundary_features
from .model import probabilities,decode

VERSIONS=(RNN_VERSION,RNN_REFINED_VERSION,RNN_SYMMETRIC_VERSION,RNN_DWELL_VERSION,BOUNDARY_VERSION)

def load_frozen(path):
    with np.load(path,allow_pickle=False) as d:
        version=str(d['version'])
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
    p=observation_probabilities(frozen['model'],sequence,pitch,frozen['symmetric'],mask)
    if frozen['head'] is not None:
        b=probabilities(frozen['head'],invariant_boundary_features(sequence,pitch,p,mask))[:,1]
        labels=dwell_decode(p,frozen['penalty'],b,frozen['strength'],frozen['min_frames'])
    elif frozen['min_frames'] is not None:
        labels=dwell_decode(p,frozen['penalty'],np.full(len(p),.5),0,frozen['min_frames'])
    else:labels=decode(p,frozen['penalty'],stop_penalty=frozen['stop_penalty'])
    return labels,p
