"""070: bounded short-control proposals; model support is not verified touches."""
import numpy as np


def recover_short(frozen,proposal,seen,receipt_scores,receipt_valid,close,threshold,require_receipt=True):
    frozen=np.asarray(frozen);proposal=np.asarray(proposal);seen=np.asarray(seen,bool);scores=np.asarray(receipt_scores);valid=np.asarray(receipt_valid,bool);close=np.asarray(close,bool);n=len(frozen)
    if proposal.shape!=(n,) or seen.shape!=(n,) or scores.shape!=(n,2) or valid.shape!=scores.shape or close.shape!=scores.shape or not np.isfinite(scores).all() or not np.isin(frozen,[0,1,2]).all() or not np.isin(proposal,[0,1,2]).all() or not 0<=threshold<=1:raise ValueError('Invalid short proposal inputs')
    out=frozen.copy();records=[];edges=np.r_[0,np.flatnonzero(proposal[1:]!=proposal[:-1])+1,n]
    for a,b in zip(edges[:-1],edges[1:]):
        team=int(proposal[a]);other=1-team
        if team==2 or a==0 or b==n or b-a>75:continue
        if proposal[a-1]!=other or proposal[b]!=other:continue
        # Entire new spell and its flanks must sit inside ONE opposite spell.
        if not (frozen[a-1:b+1]==other).all():continue
        if not seen[a-1:b+1].all():continue
        support=valid[a:b,team]&close[a:b,team]&(scores[a:b,team]>=threshold)
        accepted=bool(support.any()) or not require_receipt
        records.append({'start':int(a),'end':int(b),'team':team,'accepted':accepted,'receipt_supported':bool(support.any())})
        if accepted:out[a:b]=team
    return out,records
