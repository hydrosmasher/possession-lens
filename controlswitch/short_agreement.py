"""071: abstain from short proposals when model evidence conflicts."""
import numpy as np
from .short_proposals import recover_short


def recover_with_agreement(frozen,proposal,seen,scores,valid,close,threshold,prior):
    prior=np.asarray(prior,float)
    if prior.shape!=(len(frozen),2) or not np.isfinite(prior).all() or (prior<0).any() or (prior.sum(1)<=0).any():raise ValueError('Invalid prior team evidence')
    prior=prior/prior.sum(1,keepdims=True)
    out,records=recover_short(frozen,proposal,seen,scores,valid,close,threshold)
    for record in records:
        if not record['accepted']:continue
        a,b,team=record['start'],record['end'],record['team'];other=1-team
        original_agrees=bool(prior[a:b,team].mean()>.5)
        support=valid[a:b,team]&close[a:b,team]&(scores[a:b,team]>=threshold)
        opposing_support=valid[a:b,other]&close[a:b,other]&(scores[a:b,other]>=threshold)
        exclusive_receipt=bool((support&~opposing_support).any())
        record.update(prior_mean_agrees=original_agrees,exclusive_receipt_support=exclusive_receipt)
        if not (original_agrees and exclusive_receipt):
            out[a:b]=np.asarray(frozen)[a:b];record['accepted']=False
    return out,records
