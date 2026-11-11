"""Clearly defined possession denominators; no player identity claims from slots."""
import numpy as np

def team_statistics(labels,fps=25):
    labels=np.asarray(labels);counts={k:int((labels==k).sum()) for k in 'HADX'}
    alive=counts['H']+counts['A'];total=len(labels)
    return {'denominator':'predicted live possession frames H+A; dead and unknown excluded',
            'home_possession_percent':100*counts['H']/alive if alive else None,
            'away_possession_percent':100*counts['A']/alive if alive else None,
            'live_seconds':alive/fps,'dead_seconds':counts['D']/fps,'unknown_seconds':counts['X']/fps,
            'known_label_coverage_percent':100*(total-counts['X'])/total if total else None,'counts':counts,
            'status':'model estimates, not ground-truth statistics'}

def tracklet_statistics(labels,xy,ball,teams,fps=25,radius=2):
    """Anonymous proximity ledger; never a confirmed player-possession metric.

    Radius is the original fixed proxy setting, not fitted on competition data.
    Numerical distance ties are withheld instead of choosing the first slot.
    """
    labels=np.asarray(labels);xy=np.asarray(xy);ball=np.asarray(ball);teams=np.asarray(teams)
    if len(labels)==0:raise ValueError('No frame labels for tracklet attribution')
    if xy.ndim!=3 or xy.shape[2]!=2 or xy.shape[1]<1 or ball.shape!=(len(labels),2) or xy.shape[0]!=len(labels) or len(teams)!=xy.shape[1]:raise ValueError('Inconsistent tracklet arrays')
    if not np.isin(teams,[0,1]).all() or fps<=0 or radius<=0:raise ValueError('Invalid tracklet settings')
    distance=np.linalg.norm(xy-ball[:,None,:],axis=2);distance[~np.isfinite(distance)]=np.inf
    nearest=np.argmin(distance,axis=1);minimum=distance[np.arange(len(distance)),nearest]
    # A numerical equality check, not a fitted physical ambiguity margin.
    if distance.shape[1]>1:
        second=np.partition(distance,1,axis=1)[:,1]
        tied=np.isfinite(minimum)&np.isclose(minimum,second,rtol=1e-6,atol=1e-6)
    else:tied=np.zeros(len(labels),bool)
    ball_seen=np.isfinite(ball).all(axis=1);players_seen=np.isfinite(xy).all(axis=2).any(axis=1)
    rows=[];reasons={};ownership=np.full(len(labels),-1,dtype=int)
    for t,label in [(0,'H'),(1,'A')]:
        active=labels==label;denominator=int(active.sum())
        assigned=active&ball_seen&players_seen&(minimum<=radius)&(teams[nearest]==t)&~tied
        ownership[assigned]=nearest[assigned]
        reason=np.full(len(labels),'unassigned',dtype='<U32')
        reason[active&~ball_seen]='missing_ball'
        reason[active&ball_seen&~players_seen]='missing_players'
        reason[active&ball_seen&players_seen&(minimum>radius)]='no_track_within_radius'
        near=active&ball_seen&players_seen&(minimum<=radius)
        reason[near&(teams[nearest]!=t)]='nearest_track_other_team'
        reason[near&tied]='numerical_distance_tie'
        reason[assigned]='assigned'
        reasons[label]={k:int((active&(reason==k)).sum()) for k in ['assigned','missing_ball','missing_players','no_track_within_radius','nearest_track_other_team','numerical_distance_tie']}
        total=0
        for slot in np.flatnonzero(teams==t):
            count=int((assigned&(nearest==slot)).sum());total+=count
            rows.append({'team':label,'track_slot':int(slot)+1,'associated_control_seconds':count/fps,
                         'percent_of_team_possession':100*count/denominator if denominator else None,
                         'identity_status':'unverified tracklet; not a player or shirt number'})
        rows.append({'team':label,'track_slot':'UNASSIGNED','associated_control_seconds':(denominator-total)/fps,
                     'percent_of_team_possession':100*(denominator-total)/denominator if denominator else None,
                     'identity_status':'no unique nearby same-team association'})
    changes=np.r_[0,np.flatnonzero(ownership[1:]!=ownership[:-1])+1,len(labels)]
    bouts=[{'start_frame':int(a),'end_frame_exclusive':int(b),'track_slot':int(ownership[a])+1,
            'team':'H' if teams[ownership[a]]==0 else 'A','seconds':(b-a)/fps}
           for a,b in zip(changes[:-1],changes[1:]) if ownership[a]>=0]
    return {'schema_version':2,'method':'nearest-track proximity ledger, not confirmed player attribution',
            'radius_metres':radius,'warning':'Slots can fragment or change identity. Do not rename or aggregate them as individual players.',
            'ambiguity_note':'Numerical distance ties are unassigned. Proximity does not confirm contact; some flight frames may be associated.',
            'unassigned_reasons':reasons,'bouts':bouts,'rows':rows}
