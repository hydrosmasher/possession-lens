"""072: external-fit proposal scoring and exact recovered-spell retention."""
import numpy as np
from .pipeline import pm
from .short_proposals import recover_short
from .receipt_evidence import receipt_features
from .local_temporal import symmetric,locked_dead_decode
from .boundary_had import boundary_observations
from .model import probabilities
from . import tree_model

FEATURE_NAMES=('length','prior_mean','prior_max','prior_std','prior_support_fraction','local_mean','local_min','local_max','local_contrast','receipt_peak','receipt_margin_peak','receipt_support_fraction','motion_after_speed','motion_speed_drop','motion_future_close','flank_prior_mean','fraction_of_old_spell')


def flags(y,p):
    runs=pm.valid_runs(y);gt=pm.possession_spells(y,runs);return pm.matched_gt_flags(pm.spell_overlaps(gt,pm.possession_spells(p,runs)),len(gt),.5)


def apply_scores(current,events,scores,threshold):
    scores=np.asarray(scores)
    if scores.shape!=(len(events),) or not np.isfinite(scores).all() or not np.isfinite(threshold):raise ValueError('Invalid calibrated scores')
    p=np.asarray(current).copy()
    if p.ndim!=1 or not np.isin(p,[0,1,2]).all():raise ValueError('Invalid baseline labels')
    previous_end=0
    for event,score in zip(events,scores):
        a,b,k=event['start'],event['end'],event['team']
        if not all(isinstance(v,(int,np.integer)) for v in (a,b,k)) or k not in (0,1) or not 0<a<b<len(p) or b-a>75 or a<previous_end or not (np.asarray(current)[a-1:b+1]==1-k).all():raise ValueError('Proposal must lie within a single opposite-team spell')
        previous_end=b
        if score>=threshold:p[a:b]=k
    return p


def event_features(current,q,g,motion,valid,receipt,threshold,events):
    h=q[:,:2]/np.maximum(q[:,:2].sum(1,keepdims=True),1e-12);rows=[]
    for event in events:
        a,b,k=event['start'],event['end'],event['team'];other=1-k;sl=slice(a,b);p=h[sl,k];local=g[sl,k];good=valid[sl,k]&(motion[sl,k,0]<=2.5);r=np.where(good,receipt[sl,k],0);opposite=np.where(valid[sl,other]&(motion[sl,other,0]<=2.5),receipt[sl,other],0)
        left=a
        while left>0 and current[left-1]==other:left-=1
        right=b
        while right<len(current) and current[right]==other:right+=1
        flank=np.r_[g[max(0,a-12):a,k],g[b:min(len(g),b+12),k]];best=int(r.argmax());m=motion[a+best,k]
        rows.append([(b-a)/75,p.mean(),p.max(),p.std(),(p>.5).mean(),local.mean(),local.min(),local.max(),local.mean()-flank.mean(),r.max(),np.max(r-opposite),(r>=threshold).mean(),m[8]/30,m[11]/30,m[13],(h[a-1,k]+h[b,k])/2,(b-a)/max(right-left,1)])
    x=np.asarray(rows,dtype=np.float32).reshape(-1,len(FEATURE_NAMES))
    if not np.isfinite(x).all():raise ValueError('Nonfinite proposal features')
    return x


def observation_case(snapshot,local,receipt_model,receipt_threshold,s,pitch,mask,current=None,q=None,event=None):
    if current is None:
        from .research_snapshot import infer_snapshot
        current,q=infer_snapshot(snapshot,s,pitch,mask)
    if event is None:event=probabilities(snapshot['base']['models']['boundary'],boundary_observations(s,pitch,q,mask))[:,1]
    g=symmetric(local,s,pitch,mask);h=q[:,:2]/np.maximum(q[:,:2].sum(1,keepdims=True),1e-12);proposal=locked_dead_decode(.75*h+.25*g,event,current);motion,valid=receipt_features(s,mask);receipt=tree_model.predict(receipt_model,motion.reshape(-1,14))[:,1].reshape(-1,2);seen=np.isfinite(s['ball']).all(1)
    if mask is not None:seen&=~mask
    full,reviews=recover_short(current,proposal,seen,receipt,valid,motion[:,:,0]<=2.5,receipt_threshold);events=[r for r in reviews if r['accepted']];x=event_features(current,q,g,motion,valid,receipt,receipt_threshold,events)
    return {'current':current,'prototype':full,'events':events,'x':x,'seen':seen,'proposal':proposal,'motion':motion,'receipt_valid':valid,'receipt':receipt,'q':q,'g':g}


def targets(y,case):
    base=flags(y,case['current']);labels=[]
    for event in case['events']:
        p=case['current'].copy();p[event['start']:event['end']]=event['team'];labels.append(int((flags(y,p)&~base).any()))
    return np.asarray(labels,np.int8)


def retention(y,current,prototype,candidate):
    old=flags(y,current);recovered=flags(y,prototype)&~old;new=flags(y,candidate)
    return {'prototype_recovered':int(recovered.sum()),'retained_recovered':int((new&recovered).sum()),'lost_recovered':int((~new&recovered).sum())}


def choose_threshold(cases,scores):
    """Search only external calibration cases; lock exact same-spell retention."""
    from .gap_possession import diagnostics
    if not cases or not sum(len(p) for p in scores):return 0.,{'status':'No calibration proposals; abstain from filtering'}
    values=np.unique(np.r_[0.,*[p for p in scores],1.000001]);reference_fp=sum(diagnostics(c['y'],c['prototype'])['false_positive'] for c in cases);best=0.;audit=[]
    for threshold in values:
        lost=0;fp=0;kept=0
        for case,p in zip(cases,scores):
            candidate=apply_scores(case['current'],case['events'],p,float(threshold));lost+=retention(case['y'],case['current'],case['prototype'],candidate)['lost_recovered'];fp+=diagnostics(case['y'],candidate)['false_positive'];kept+=int((p>=threshold).sum())
        eligible=lost==0 and fp<=reference_fp;audit.append({'threshold':float(threshold),'lost_recovered':lost,'false_positive_spells':fp,'kept_proposals':kept,'eligible':eligible})
        # Highest cutoff among the least-FP retention-preserving choices.
    eligible=[a for a in audit if a['eligible']];selected=min(eligible,key=lambda a:(a['false_positive_spells'],-a['threshold'])) if eligible else audit[0]
    return selected['threshold'],{'status':'Exact recovery-retention constrained external calibration','selected':selected,'prototype_fp':reference_fp,'trials':audit}
