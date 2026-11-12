"""Provider-track evidence and conservative same-person gap recovery.

No motion-based identity guesses; no identity is inferred from a tracking slot.
"""
from collections import defaultdict
import numpy as np

def recover_ids(frames,id_teams,max_gap_frames=10):
    """Fill an anonymous run only between matching named anchors on the SAME track_id.

    Runs are separated at period changes and missing track observations. Both
    anchors must be the same roster ID, and anonymous team tags must agree.
    The input is modified only at accepted anonymous objects; known IDs are untouched.
    """
    tracks=defaultdict(list);filled=0
    for frame in frames:
        for obj in frame.get('data',[]):
            if obj.get('track_id') is not None:tracks[(frame.get('period'),obj['track_id'])].append((frame['frame'],obj))
    for rows in tracks.values():
        start=0
        while start<len(rows):
            end=start+1
            while end<len(rows) and rows[end][0]==rows[end-1][0]+1:end+=1
            segment=rows[start:end];known=[i for i,(_,o) in enumerate(segment) if o.get('trackable_object') in id_teams]
            for a,b in zip(known[:-1],known[1:]):
                key=segment[a][1]['trackable_object']
                if b-a-1<1 or b-a-1>max_gap_frames or segment[b][1]['trackable_object']!=key:continue
                team=id_teams[key];accepted=True
                for _,obj in segment[a+1:b]:
                    group=obj.get('group_name','')
                    expected='home' if team==0 else 'away'
                    if obj.get('trackable_object') is not None or not group.startswith(expected):accepted=False;break
                if accepted:
                    for _,obj in segment[a+1:b]:obj['trackable_object']=key;filled+=1
            start=end
    return filled

def ownership(labels,xy,ball,teams,radius=2.):
    distance=np.linalg.norm(xy-ball[:,None,:],axis=2);distance[~np.isfinite(distance)]=np.inf
    nearest=np.argmin(distance,axis=1);minimum=distance[np.arange(len(ball)),nearest]
    second=np.partition(distance,1,axis=1)[:,1] if xy.shape[1]>1 else np.full(len(ball),np.inf)
    tied=np.isfinite(minimum)&np.isclose(minimum,second,rtol=1e-6,atol=1e-6)
    active=np.isin(labels,['H','A']);expected=np.where(np.asarray(labels)=='H',0,1)
    valid=active&(minimum<=radius)&(teams[nearest]==expected)&~tied
    return np.where(valid,nearest,-1).astype(np.int32)

def bridge_control(labels,xy,ball,teams,n_named,max_gap=15):
    """Recover only bracketed missing-ball gaps for the SAME stable named person.

    Player tracking must exist throughout, the team must be unchanged, and a
    straight ball corridor between observations must remain within 2m of that
    player with no competing player within 2m. Anonymous observations block it.
    This is inferred continuity, not newly observed ball contact.
    """
    baseline=ownership(labels,xy,ball,teams);result=baseline.copy();added=np.zeros(len(ball),bool)
    missing=~np.isfinite(ball).all(1);edges=np.r_[0,np.flatnonzero(missing[1:]!=missing[:-1])+1,len(ball)]
    for a,b in zip(edges[:-1],edges[1:]):
        if not missing[a] or a==0 or b==len(ball) or b-a>max_gap:continue
        owner=baseline[a-1]
        if owner<0 or owner>=n_named or baseline[b]!=owner:continue
        if not np.all(np.asarray(labels)[a-1:b+1]==labels[a-1]) or labels[a-1] not in ('H','A'):continue
        path=xy[a:b,owner]
        if not np.isfinite(path).all():continue
        corridor=np.linspace(ball[a-1],ball[b],b-a+2)[1:-1]
        if (np.linalg.norm(path-corridor,axis=1)>2).any():continue
        distance=np.linalg.norm(xy[a:b]-corridor[:,None,:],axis=2);distance[:,owner]=np.inf
        if np.any(np.isfinite(distance)&(distance<=2)):continue
        result[a:b]=owner;added[a:b]=True
    return result,added

def ledger_from_ownership(labels,owners,teams,fps=25):
    rows=[]
    for t,label in enumerate('HA'):
        active=np.asarray(labels)==label;n=int(active.sum());assigned=0
        for i in np.flatnonzero(teams==t):
            count=int(np.sum(active&(owners==i)));assigned+=count
            rows.append({'team':label,'track_slot':int(i)+1,'associated_control_seconds':count/fps,'percent_of_team_possession':100*count/n if n else None})
        rows.append({'team':label,'track_slot':'UNASSIGNED','associated_control_seconds':(n-assigned)/fps,'percent_of_team_possession':100*(n-assigned)/n if n else None})
    edges=np.r_[0,np.flatnonzero(owners[1:]!=owners[:-1])+1,len(owners)]
    bouts=[{'track_slot':int(owners[a])+1,'team':'H' if teams[owners[a]]==0 else 'A','start_frame':int(a),'end_frame_exclusive':int(b),'seconds':(b-a)/fps} for a,b in zip(edges[:-1],edges[1:]) if owners[a]>=0]
    return {'rows':rows,'bouts':bouts}
