"""Automatic SportsCode intervals: parse locally; never treat them as possession truth."""
from pathlib import Path
import math,unicodedata,xml.etree.ElementTree as ET
from collections import Counter
import numpy as np


def normalise(value):
    text=unicodedata.normalize('NFKD',str(value or '')).encode('ascii','ignore').decode().casefold()
    return ' '.join(text.split())


def team_key(value):
    key=''.join(c for c in normalise(value) if c.isalnum())
    # Spelling documented by the organiser, not a manually inferred identity.
    return 'athleticclub' if key=='atleticclub' else key


def load(path,team_names,fps=25,n_frames=None):
    path=Path(path)
    if path.stat().st_size>20_000_000:raise ValueError('Oversized tagging XML')
    if len(team_names)!=2 or not math.isfinite(fps) or fps<=0:raise ValueError('Invalid tagging context')
    teams={team_key(name):i for i,name in enumerate(team_names)}
    if len(teams)!=2 or not all(teams):raise ValueError('Ambiguous team names')
    intervals=[];unknown=0;codes=Counter();invalid=0;clipped=0;conflicts=0
    for instance in ET.parse(path).iter('instance'):
        try:start=float(instance.findtext('start'));end=float(instance.findtext('end'))
        except (TypeError,ValueError):raise ValueError('Non-numeric tagging time') from None
        if not math.isfinite(start) or not math.isfinite(end) or end<=start:raise ValueError('Invalid tagging interval')
        lo=int(round(start*fps));hi=int(round(end*fps));tags={}
        for label in instance.findall('label'):
            group=normalise(label.findtext('group'));value=normalise(label.findtext('text'));tags.setdefault(group,set()).add(value)
        team_values=tags.get('team',set());mapped={teams.get(team_key(v)) for v in team_values}
        team=next(iter(mapped)) if len(mapped)==1 and None not in mapped else None
        if len(team_values)>1:conflicts+=1
        if team is None:unknown+=1
        code=normalise(instance.findtext('code'));codes[code]+=1
        if n_frames is not None:
            newlo=max(0,min(n_frames,lo));newhi=max(0,min(n_frames,hi));clipped+=int((lo,hi)!=(newlo,newhi));lo,hi=newlo,newhi
        if hi<=lo:invalid+=1;continue
        intervals.append({'start':lo,'end':hi,'code':code,'team':team,'type':tuple(sorted(tags.get('type',set()))),'half':tuple(sorted(tags.get('half',set())))})
    return intervals,{'instances_seen':sum(codes.values()),'usable_intervals':len(intervals),'unknown_team_instances':unknown,'conflicting_team_groups':conflicts,'clipped_intervals':clipped,'empty_after_clipping':invalid,'codes':dict(codes),'time_conversion':'round(seconds*fps); no fitted offset','possession_labels_used':False,'manual_identity_mapping_used':False}


ATTACKING={'build up','progression','attacking transition','creating chances','long ball'}
DEFENDING={'defending in defensive third','defending in middle third','defending in attacking third','defensive transition'}


def phase_evidence(intervals,n_frames):
    """Unweighted semantic votes only; conflicts abstain, restart windows not D."""
    delta=np.zeros((n_frames+1,2),np.int32);restart=np.zeros(n_frames+1,np.int32)
    for item in intervals:
        a,b=item['start'],item['end']
        if not 0<=a<b<=n_frames:raise ValueError('Out-of-range tagging interval')
        team=item['team'];code=item['code']
        if code=='set pieces':restart[a]+=1;restart[b]-=1
        if team not in (0,1):continue
        if code in ATTACKING:owner=team
        elif code in DEFENDING:owner=1-team
        else:continue
        delta[a,owner]+=1;delta[b,owner]-=1
    votes=delta[:-1].cumsum(0);active=votes>0;conflict=active.all(1);unopposed=active&~conflict[:,None]
    return {'team_votes':votes,'unopposed_team_evidence':unopposed,'conflict':conflict,'restart_window':restart[:-1].cumsum()>0,'meaning':'Automatic phase evidence only; not possession labels, exact restart times or verified events.'}
