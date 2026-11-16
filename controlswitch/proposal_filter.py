"""Portable proposal filter artifacts; rejected research requires explicit opt-in."""
from pathlib import Path
import json,numpy as np
from .candidate_calibration import COLUMNS
from .proposal_calibration import FEATURE_NAMES,apply_scores
from .model import probabilities
VERSION='short-proposal-filter-073-v1'
NAMES=[FEATURE_NAMES[i] for i in COLUMNS]


def save(path,model,threshold,provenance,approved=False):
    meta={'version':VERSION,'features':NAMES,'threshold':float(threshold),'approved':bool(approved),'provenance':provenance,'meaning':'Proposal match score and retention-calibrated cutoff; not calibrated confidence or verified receipts.'}
    np.savez_compressed(path,meta=np.array(json.dumps(meta)),**model)
    load(path,allow_research=True)


def load(path,allow_research=False):
    if Path(path).stat().st_size>1_000_000:raise ValueError('Oversized proposal filter')
    with np.load(path,allow_pickle=False) as a:
        if set(a.files)!={'meta','weights','mean','scale'}:raise ValueError('Unexpected filter fields')
        meta=json.loads(str(a['meta']));threshold=meta.get('threshold')
        if meta.get('version')!=VERSION or meta.get('features')!=NAMES or not isinstance(meta.get('approved'),bool) or not isinstance(threshold,(int,float)) or not np.isfinite(threshold) or not 0<=threshold<=1.000001:raise ValueError('Invalid filter policy')
        if not meta['approved'] and not allow_research:raise ValueError('Research filter failed release acceptance; explicit research opt-in required')
        model={k:a[k].copy() for k in ('weights','mean','scale')}
        if model['weights'].shape!=(12,2) or model['mean'].shape!=(11,) or model['scale'].shape!=(11,) or not all(np.isfinite(v).all() for v in model.values()) or (model['scale']<=0).any():raise ValueError('Invalid filter weights/scaling')
    return {'model':model,'threshold':float(threshold),'meta':meta}


def filter_proposals(bundle,current,events,features):
    x=np.asarray(features)
    if x.shape!=(len(events),17) or not np.isfinite(x).all():raise ValueError('Invalid raw proposal features')
    score=probabilities(bundle['model'],x[:,COLUMNS])[:,1]
    return apply_scores(current,events,score,bundle['threshold']),score
