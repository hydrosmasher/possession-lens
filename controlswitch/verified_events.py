"""Provider-coded Sportec events; never infers shots, passes or touches from tracking."""
from pathlib import Path
from datetime import datetime
import json,hashlib,xml.etree.ElementTree as ET
from .player_mapping import external_roster
from .external import ALL_MATCHES

def summarise(events,metadata,match='J03WN1'):
 roster,title=external_roster(metadata,match);spec=ALL_MATCHES[match];teams={spec['home']:'H',spec['away']:'A'};people={person:{**p,'team':'H' if t==0 else 'A'} for (t,person),p in roster.items()};events=list(ET.parse(events).getroot().iter('Event'))
 if any(e.get('MatchId')!='DFL-MAT-'+match for e in events):raise ValueError('Event match does not match roster')
 ids=[e.get('EventId') for e in events]
 if None in ids or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicated event IDs; resolve provider revisions first')
 active=[e for e in events if e.find('Delete') is None];starts={};ends=set()
 for e in active:
  kick=e.find('KickOff');end=e.find('FinalWhistle')
  if kick is not None:starts[kick.get('GameSection')]=min(starts.get(kick.get('GameSection'),datetime.max.replace(tzinfo=datetime.fromisoformat(e.get('EventTime')).tzinfo)),datetime.fromisoformat(e.get('EventTime')))
  if end is not None:ends.add(end.get('GameSection'))
 if not all(k in starts and k in ends for k in ('firstHalf','secondHalf')):raise ValueError('Cannot claim full-match event totals without both half boundaries')
 def empty():return {'touches':None,'completed_passes':0,'shots':0,'shots_on_target':0,'goals':0}
 totals={t:empty() for t in 'HA'};players={pid:{**p,**empty()} for pid,p in people.items()};timeline=[]
 def add(e,action,metric,on_target=False,goal=False,recipient=None):
  team=teams.get(action.get('Team'));pid=action.get('Player')
  if team is None or pid not in people or people[pid]['team']!=team:raise ValueError('Unmapped event actor; no name guessing')
  totals[team][metric]+=1;players[pid][metric]+=1
  if on_target:totals[team]['shots_on_target']+=1;players[pid]['shots_on_target']+=1
  if goal:totals[team]['goals']+=1;players[pid]['goals']+=1
  stamp=datetime.fromisoformat(e.get('EventTime'));period=2 if stamp>=starts['secondHalf'] else 1;seconds=(stamp-starts['secondHalf' if period==2 else 'firstHalf']).total_seconds();kind='goal' if goal else 'shot' if metric=='shots' else 'completed_pass'
  row={'event_id':e.get('EventId'),'period':period,'period_seconds':round(seconds,3),'team':team,'person_id':pid,'player_name':people[pid]['player_name'],'type':kind,'on_target':on_target if metric=='shots' else None}
  if recipient in people and people[recipient]['team']==team:row['recipient_name']=people[recipient]['player_name']
  timeline.append(row)
 for e in active:
  for play in e.iter('Play'):
   if (play.find('Pass') is not None or play.find('Cross') is not None) and play.get('Evaluation') in ('successfullyCompleted','successful'):add(e,play,'completed_passes',recipient=play.get('Recipient'))
  for shot in e.iter('ShotAtGoal'):
   goal=shot.find('SuccessfulShot') is not None;on_target=goal or shot.find('SavedShot') is not None;add(e,shot,'shots',on_target,goal)
  sub=e.find('Substitution')
  if sub is not None:
   team=teams.get(sub.get('Team'));a=people.get(sub.get('PlayerOut'));b=people.get(sub.get('PlayerIn'))
   if team not in ('H','A') or a is None or b is None or a['team']!=team or b['team']!=team:raise ValueError('Substitution roster join failed')
   stamp=datetime.fromisoformat(e.get('EventTime'));period=2 if stamp>=starts['secondHalf'] else 1;seconds=(stamp-starts['secondHalf' if period==2 else 'firstHalf']).total_seconds();timeline.append({'event_id':e.get('EventId'),'period':period,'period_seconds':round(seconds,3),'team':team,'person_id':sub.get('PlayerIn'),'player_name':b['player_name'],'player_out':a['player_name'],'type':'substitution'})
 for t in 'HA':
  for k in ['completed_passes','shots','shots_on_target','goals']:
   if sum(p[k] for p in players.values() if p['team']==t)!=totals[t][k]:raise ValueError('Player event totals do not reconcile to team totals')
 return {'source_kind':'licensed_provider_events','match_id':match,'title':title,'teams':totals,'players':players,'timeline':sorted(timeline,key=lambda r:(r['period'],r['period_seconds'],r['event_id'])),'touches_available':False,'definitions':{'completed_passes':'Provider Play with Pass or Cross, Evaluation successfullyCompleted/successful; includes crosses and set pieces.','shots':'Provider ShotAtGoal only.','shots_on_target':'Provider SuccessfulShot or SavedShot only; woodwork and blocked shots excluded.','touches':'Not available: event stream does not establish every physical touch.','timeline':'Provider event wall times expressed relative to each half kickoff; separate from model tracking-time timeline.'},'source':'Bassek et al. (2025), IDSSE/Sportec, DOI 10.1038/s41597-025-04505-y; CC BY 4.0','source_revision':'a715a38dfbaf5f58e431727c2b78d174101a703c','source_url':'https://huggingface.co/datasets/pysport/idsse-data/resolve/a715a38dfbaf5f58e431727c2b78d174101a703c/DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml','event_sha256':None}

def build(events,metadata,out):
 result=summarise(events,metadata);data=Path(events).read_bytes();result['event_sha256']=hashlib.sha256(data).hexdigest();result['git_blob']=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
 if result['git_blob']!='67b982007b8e3f5de83dcd104779acc92b76a813':raise ValueError('Pinned event source hash differs')
 Path(out).write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('events','metadata','out'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();build(a.events,a.metadata,a.out)
