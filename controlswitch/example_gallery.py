"""Named external match library; no challenge data or guessed player identities."""
from pathlib import Path
import json,collections
import numpy as np
from .skillcorner_europe import load_match,digest
from .rnn_runtime import load_frozen,infer_frozen
from .statistics import tracklet_statistics
from .pipeline import intervals

MATCHES={'2068':('Serie A','2019-10-06'),'2269':('Ligue 1','2019-10-27'),'2417':('Bundesliga','2019-11-09'),'2440':('Premier League','2019-11-10'),'2841':('La Liga','2019-12-18'),'3442':('La Liga','2020-03-01'),'3518':('Serie A','2020-03-08'),'3749':('Bundesliga','2020-05-26'),'4039':('Premier League','2020-07-02')}

def roster(metadata):
    players=[p for p in metadata['players'] if p.get('trackable_object') is not None];seen=set();result=[]
    for p in players:
        key=p['trackable_object'];name=' '.join(str(p.get(k) or '').strip() for k in ('first_name','last_name')).strip()
        if key in seen or not name:raise ValueError('Ambiguous or missing provider roster identity; no guessed names permitted')
        seen.add(key);result.append({'person_id':str(key),'player_name':name,'shirt_number':p.get('number')})
    return result

def named_ledger(ledger,players,team_seconds):
    result={};unassigned={t:0. for t in 'HA'};bouts=collections.Counter()
    for bout in ledger['bouts']:
        i=bout['track_slot']-1
        if i<len(players):bouts[(bout['team'],i)]+=1
    for row in ledger['rows']:
        t=row['team'];slot=row['track_slot'];seconds=row['associated_control_seconds']
        if slot=='UNASSIGNED' or slot>len(players):unassigned[t]+=seconds;continue
        i=slot-1;p=players[i]
        if seconds>0:result[(t,p['person_id'])]={**p,'team':t,'associated_control_seconds':seconds,'associated_bouts':bouts[(t,i)]}
    for t in 'HA':
        total=sum(p['associated_control_seconds'] for p in result.values() if p['team']==t)
        if not np.isclose(total+unassigned[t],team_seconds[t],atol=1e-5):raise ValueError('Named + unresolved control does not reconcile')
    return result,unassigned

def build_match(raw,mid,model):
    path=Path(raw)/mid;m=json.loads((path/'match_data.json').read_text());players=roster(m);parts,_=load_match(raw,mid);state=load_frozen(model)
    totals={t:0. for t in 'HA'};unassigned={t:0. for t in 'HA'};merged={};blocks=[];labels=[];offset=0
    for s,pitch in parts:
        codes,_=infer_frozen(state,s,pitch);y=np.array(['H','A','D'])[codes]
        observable=np.isfinite(s['xy']).all(2).any(1)|np.isfinite(s['ball']).all(1)
        y[~observable]='X'  # Observation coverage, never provider possession labels.
        ledger=tracklet_statistics(y,s['xy'],s['ball'],s['teams']);seconds={t:float((y==t).sum()/25) for t in 'HA'}
        named,missing=named_ledger(ledger,players,seconds)
        for key,p in named.items():
            if key not in merged:merged[key]={**p,'associated_control_seconds':0.,'associated_bouts':0}
            for metric in ('associated_control_seconds','associated_bouts'):merged[key][metric]+=p[metric]
        for t in 'HA':totals[t]+=seconds[t];unassigned[t]+=missing[t]
        blocks.append([offset,offset+len(y)]);offset+=len(y);labels.append(y)
    rows=list(merged.values())
    for t in 'HA':
        rows.append({'person_id':'UNASSIGNED','player_name':'Unassigned control','shirt_number':None,'team':t,'associated_control_seconds':unassigned[t],'associated_bouts':None})
    for row in rows:row['percent_of_team_possession']=100*row['associated_control_seconds']/totals[row['team']] if totals[row['team']] else None
    league,date=MATCHES[mid]
    return {'schema_version':2,'mode':'external_named','example_id':'skillcorner-'+mid,'league':league,'date':date,'title':m['home_team']['name']+' vs '+m['away_team']['name'],'model_name':'Context RNN · estimated control; H/A only','dead_ball_supported':False,'time_basis':'Tracking time · halves concatenated','n_frames':offset,'fps':25,'blocks':blocks,'teams':{'H':m['home_team']['name'],'A':m['away_team']['name']},'timeline':intervals(np.concatenate(labels)),'tracklets':None,'named_players':{'source_kind':'external_provider_stable_ids','match_id':mid,'identity_join':'Provider trackable_object joined to match roster; anonymous tracks never renamed','metadata_sha256':digest(path/'match_data.json'),'tracking_sha256':digest(path/'structured_data.json'),'model_sha256':digest(model),'rows':rows,'warning':'Estimated proximity involvement, not confirmed touches, passes or causal impact. Only provider-mapped players are displayed; unresolved control remains in the denominator. H/A-only model does not classify stopped play. SkillCorner historical open data, MIT; transformed summaries.'}}

def build(raw,model,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);matches=[]
    for mid in MATCHES:
        payload=build_match(raw,mid,model);matches.append(payload)
        (out/(mid+'.json')).write_text(json.dumps(payload,allow_nan=False,separators=(',',':'))+'\n');print('Built named external example',mid,len(payload['named_players']['rows'])-2,'players',flush=True)
    legacy=Path(__file__).resolve().parents[1]/'examples/external-named-analysis.json'
    if legacy.exists():
        p=json.loads(legacy.read_text());p.update(example_id='sportec-J03WN1',league='Bundesliga',date='2023-05-27');matches.append(p)
    library={'schema_version':1,'matches':matches,'unavailable_examples':[{'match_id':'J03WMX','title':'Cologne–Bayern','reason':'Verified provider roster unavailable locally; no guessed names.'}],'note':'Licensed external summaries only. No challenge data. All displayed people have provider roster mappings; unassigned time is not a person.'}
    (out.parent/'match-library.json').write_text(json.dumps(library,allow_nan=False,separators=(',',':'))+'\n');return library

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('raw','model','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();build(a.raw,a.model,a.out)
