"""Event-derived external phase windows, NOT Metrica automatic tagging."""
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
from .external import ALL_MATCHES


def tracking_clocks(path,match):
    current=None;clocks={}
    for action,e in ET.iterparse(path,events=('start','end')):
        if action=='start' and e.tag=='FrameSet':
            if e.get('MatchId')!='DFL-MAT-'+match:raise ValueError('Tracking match mismatch')
            current=e.get('GameSection') if e.get('TeamId')=='BALL' else None
        if action=='end' and e.tag=='Frame':
            if current:
                n=int(e.get('N'));stamp=datetime.fromisoformat(e.get('T'));c=clocks.setdefault(current,{'first_frame':n,'first_time':stamp,'count':0});c['last_frame']=n;c['last_time']=stamp;c['count']+=1
            e.clear()
        if action=='end' and e.tag=='FrameSet':current=None;e.clear()
    if set(clocks)!={'firstHalf','secondHalf'}:raise ValueError('Missing half clock')
    for c in clocks.values():
        if c['last_frame']-c['first_frame']+1!=c['count'] or abs((c['last_time']-c['first_time']).total_seconds()*25-(c['count']-1))>.1:raise ValueError('Tracking clock is not consistent25Hz')
    return clocks


def event_phases(path,match,clocks):
    team_map={ALL_MATCHES[match][k]:i for i,k in enumerate(('home','away'))};events=list(ET.parse(path).getroot().iter('Event'));ids=[e.get('EventId') for e in events]
    if None in ids or len(ids)!=len(set(ids)):raise ValueError('Duplicate/missing provider event IDs')
    groups=defaultdict(list);counts=defaultdict(int)
    for e in events:
        if e.get('MatchId')!='DFL-MAT-'+match:raise ValueError('Event match mismatch')
        if e.find('Delete') is not None:continue
        stamp=datetime.fromisoformat(e.get('EventTime'));periods=[k for k,c in clocks.items() if c['first_time']<=stamp<=c['last_time']]
        if len(periods)!=1:continue
        period=periods[0];c=clocks[period];frame=int(round((stamp-c['first_time']).total_seconds()*25));owners=set();phases=set()
        for a in e.iter():
            if a.tag in ('Play','ShotAtGoal','BallClaiming') and a.get('Team') in team_map:
                owners.add(team_map[a.get('Team')]);phases.add(a.get('BallPossessionPhase') or 'event-'+e.get('EventId'))
        if len(owners)!=1 or len(phases)!=1:continue
        team=next(iter(owners));phase=next(iter(phases));groups[(period,phase,team)].append(frame);counts[period]+=1
    output={k:[] for k in clocks}
    for (period,phase,team),frames in groups.items():
        # Expand first/last provider action by1s, not to exact possession bounds.
        lo=max(0,min(frames)-25);hi=min(clocks[period]['count'],max(frames)+26)
        if lo<hi:output[period].append({'start':lo,'end':hi,'team':team,'code':'build up'})
    audit={'source_kind':'provider_event_phase_proxy','not_metrica_tags':True,'provider_actions':dict(counts),'phase_windows':{k:len(v) for k,v in output.items()},'reference_possession_labels_used':False,'clock_source':'TrackingN/T only; no possession/state fields read','interval_source':'First/last Play/ShotAtGoal/BallClaiming in provider phase plusfixed1s padding; not exact possession labels.'}
    return output,audit


def corrupt(intervals,n,seed):
    rng=np.random.default_rng(seed);out=[]
    for item in intervals:
        if rng.random()<.3:continue
        jitter=int(rng.integers(-50,51));a=max(0,item['start']+jitter);b=min(n,item['end']+jitter)
        if a>=b:continue
        team=1-item['team'] if rng.random()<.1 else item['team'];out.append({**item,'start':a,'end':b,'team':team})
    return out
