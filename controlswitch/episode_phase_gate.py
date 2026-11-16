"""Whole-episode veto: keep unsupported/conflicting automatic evidence unchanged."""
import numpy as np
from .smart_tagging import phase_evidence
from .proposal_calibration import apply_scores


def episode_filter(current,events,intervals):
    evidence=phase_evidence(intervals,len(current));keep=[];reviews=[]
    for event in events:
        a,b,k=event['start'],event['end'],event['team'];opposite=evidence['unopposed_team_evidence'][a:b,1-k]
        # Require opposition over the ENTIRE episode; no partially covered veto.
        veto=bool(len(opposite) and opposite.all());keep.append(0. if veto else 1.);reviews.append({**event,'phase_veto':veto,'opposed_fraction':float(opposite.mean())})
    p=apply_scores(current,events,np.asarray(keep),.5)
    return p,reviews
