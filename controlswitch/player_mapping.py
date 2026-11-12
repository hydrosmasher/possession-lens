"""Verified external roster joins. Never infers a person from a challenge slot."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
from .external import ALL_MATCHES,period_arrays
from .pipeline import digest,intervals
from .player_identity import external_identities
from .rnn_runtime import load_frozen,infer_frozen
from .statistics import tracklet_statistics
from .dashboard import export_html

def external_roster(metadata,match):
 if match not in ALL_MATCHES:raise ValueError('Unsupported external identity source')
 root=ET.parse(metadata).getroot();g=root.find('.//General');spec=ALL_MATCHES[match]
 if g is None or g.get('MatchId')!='DFL-MAT-'+match or g.get('HomeTeamId')!=spec['home'] or g.get('GuestTeamId')!=spec['away']:raise ValueError('Roster match/team identity mismatch')
 roster={}
 for team in root.iter('Team'):
  team_id=team.get('TeamId')
  if team_id not in (spec['home'],spec['away']):continue
  for p in team.iter('Player'):
   person=p.get('PersonId');first=(p.get('FirstName') or '').strip();last=(p.get('LastName') or '').strip();name=' '.join(v for v in (first,last) if v)
   if not person or not name:raise ValueError('External roster lacks stable identity or name')
   key=(0 if team_id==spec['home'] else 1,person)
   if key in roster:raise ValueError('Duplicate roster identity')
   roster[key]={'person_id':person,'player_name':name,'shirt_number':p.get('ShirtNumber') or None}
 if not roster:raise ValueError('No stable external roster mappings; generic challenge slot metadata is not a roster')
 return roster,g.get('MatchTitle') or match

def aggregate_named(parts,roster):
 """Parts are (ledger, team-possession seconds, stable IDs, slot teams)."""
 players={};totals={'H':0.,'A':0.};unassigned={'H':0.,'A':0.}
 for ledger,team_seconds,ids,teams in parts:
  for team in 'HA':totals[team]+=team_seconds[team]
  for row in ledger['rows']:
   team=row['team'];seconds=row['associated_control_seconds'];slot=row['track_slot']
   if slot=='UNASSIGNED':unassigned[team]+=seconds;continue
   t=0 if team=='H' else 1;i=slot-1
   if i<0 or i>=len(ids) or int(teams[i])!=t:raise ValueError('Track/identity alignment mismatch')
   key=(t,ids[i])
   if key not in roster:raise ValueError('Unmapped stable identity; refusing a guessed name')
   if key not in players:players[key]={**roster[key],'team':team,'associated_control_seconds':0.}
   players[key]['associated_control_seconds']+=seconds
 rows=list(players.values())
 for team in 'HA':
  assigned=sum(r['associated_control_seconds'] for r in rows if r['team']==team)
  if not np.isclose(assigned+unassigned[team],totals[team],rtol=0,atol=1e-5):raise ValueError('Named ledger does not reconcile')
  rows.append({'team':team,'person_id':'UNASSIGNED','player_name':'Unassigned control','shirt_number':None,'associated_control_seconds':unassigned[team]})
 for row in rows:row['percent_of_team_possession']=100*row['associated_control_seconds']/totals[row['team']] if totals[row['team']] else None
 return rows

def build_external(tracking,metadata,cache,model,match,out):
 out=Path(out)
 if out.exists():raise ValueError('Choose a new named external output folder')
 roster,title=external_roster(metadata,match);parts=external_identities(tracking,cache,match)
 with np.load(cache,allow_pickle=False) as d:data={k:d[k] for k in d.files}
 frozen=load_frozen(model);analyses=[];labels=[];blocks=[];offset=0
 for i,(xy,teams,ids) in enumerate(parts):
  s=period_arrays(data,i);codes,_=infer_frozen(frozen,s,data['pitch']);y=np.array(['H','A','D'])[codes]
  ledger=tracklet_statistics(y,xy,s['ball'],teams);analyses.append((ledger,{k:float((y==k).sum())/25 for k in 'HA'},ids,teams));labels.append(y);blocks.append([offset,offset+len(y)]);offset+=len(y)
 rows=aggregate_named(analyses,roster);spec=ALL_MATCHES[match]
 root=ET.parse(metadata).getroot();team_names={t.get('TeamId'):t.get('TeamName') or t.get('ShortName') or t.get('TeamId') for t in root.iter('Team')}
 payload={'schema_version':2,'mode':'external_named','title':title+' · external named-player example','model_name':'Frozen boundary RNN · external tracking; proximity estimates','n_frames':offset,'fps':25,'blocks':blocks,'teams':{'H':team_names.get(spec['home'],'Home team'),'A':team_names.get(spec['away'],'Away team')},'timeline':intervals(np.concatenate(labels)),'tracklets':None,
 'named_players':{'source_kind':'external_provider_stable_ids','match_id':match,'metadata_sha256':digest(metadata),'tracking_sha256':digest(tracking),'model_sha256':digest(model),'identity_join':'tracking PersonId + team, joined to external provider roster; never slot number','rows':rows,'warning':'Names are joined from external stable IDs. Control is a proximity proxy, not confirmed ball contact. This is not the challenge match. Source: Bassek et al. (2025), IDSSE/Sportec, doi:10.1038/s41597-025-04505-y, CC BY 4.0; transformed summaries.'}}
 out.mkdir(parents=True);(out/'match-analysis.json').write_text(json.dumps(payload,indent=2)+'\n');export_html(out/'dashboard.html',payload)
 import csv
 with (out/'named-player-contributions.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['team','person_id','player_name','shirt_number','associated_control_seconds','percent_of_team_possession']);w.writeheader();w.writerows(rows)
 return payload
