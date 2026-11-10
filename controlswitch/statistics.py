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
    """Geometric proxy, not named-player contributions. Missing/in-flight time is unassigned."""
    labels=np.asarray(labels);d=np.linalg.norm(xy-ball[:,None,:],axis=2);d[~np.isfinite(d)]=np.inf
    nearest=np.argmin(d,axis=1);minimum=d[np.arange(len(d)),nearest];rows=[]
    for t,label in [(0,'H'),(1,'A')]:
        denominator=int((labels==label).sum());assigned=0
        for slot in np.flatnonzero(teams==t):
            count=int(((labels==label)&(nearest==slot)&(minimum<=radius)).sum());assigned+=count
            rows.append({'team':label,'track_slot':int(slot)+1,'associated_control_seconds':count/fps,
                         'percent_of_team_possession':100*count/denominator if denominator else None,
                         'identity_status':'unverified tracklet; not a player or shirt number'})
        rows.append({'team':label,'track_slot':'UNASSIGNED','associated_control_seconds':(denominator-assigned)/fps,
                     'percent_of_team_possession':100*(denominator-assigned)/denominator if denominator else None,
                     'identity_status':'missing ball, flight, ambiguous or no nearby same-team track'})
    return {'method':'nearest-track proximity proxy, not RNN player attribution','radius_metres':radius,
            'warning':'Fragmented slots may refer to the same person or change identity. Do not aggregate as named-player statistics.',
            'rows':rows}
