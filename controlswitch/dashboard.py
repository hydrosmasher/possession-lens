"""One offline dashboard for shareable demo data and private local match analyses."""
from pathlib import Path
import json
import numpy as np
from .statistics import team_statistics
from .pipeline import BLOCKS,FPS,pm,describe


def demo_payload():
    rng=np.random.default_rng(17);n=150000;blocks=[[0,67500],[82500,n]];segments=[]
    for half,(a,b) in enumerate(blocks):
        if segments and segments[-1][1]<a:segments.append([segments[-1][1],a,'X'])
        t=a;last=None
        while t<b:
            k=str(rng.choice(['H','A','D'],p=[.55 if half else .61,.35 if half else .29,.10]))
            stop=min(b,t+int(rng.integers(80,850)))
            if segments and segments[-1][2]==k and segments[-1][1]==t:segments[-1][1]=stop
            else:segments.append([t,stop,k])
            t=stop
    labels=np.empty(n,dtype='<U1')
    for a,b,k in segments:labels[a:b]=k
    stats=team_statistics(labels);rows=[]
    for team in 'HA':
        total=stats['counts'][team]/25
        for slot,share in enumerate([18,16,12,10,8],1):rows.append({'team':team,'track_slot':slot+(0 if team=='H' else 5),'associated_control_seconds':total*share/100,'percent_of_team_possession':share})
        rows.append({'team':team,'track_slot':'UNASSIGNED','associated_control_seconds':total*.36,'percent_of_team_possession':36})
    return {'schema_version':2,'mode':'synthetic_demo','title':'Blue XI vs Claret XI · illustrative match',
            'model_name':'Invented demo · no match accuracy claim','n_frames':n,'fps':25,'blocks':blocks,
            'teams':{'H':'Blue XI','A':'Claret XI'},'timeline':segments,
            'tracklets':{'rows':rows,'ambiguity_note':'These anonymous demo tracks are invented.'}}


def from_run(run_dir):
    run=Path(run_dir)
    if (run/'team-statistics.json').exists():
        stats=json.loads((run/'team-statistics.json').read_text());timeline=json.loads((run/'timeline.json').read_text())
        if not timeline:raise ValueError('Empty match timeline')
        teams=stats.get('teams',{'H':'Home team','A':'Away team'});n=int(timeline[-1][1])
        tracks=json.loads((run/'tracklet-proxies.json').read_text()) if (run/'tracklet-proxies.json').exists() else None
        return {'schema_version':2,'mode':'licensed_local','title':f"{teams['H']} vs {teams['A']}",
                'model_name':'Boundary-aware recurrent model · offline estimates','n_frames':n,'fps':25,
                'blocks':[list(b) for b in BLOCKS] if n==199319 else [[0,n]],'teams':teams,
                'timeline':timeline,'tracklets':tracks,'provenance':{'model_sha256':stats.get('model_sha256'),
                'competition_accuracy_measured':False,'individual_identities_verified':False}}
    summary=json.loads((run/'summary.json').read_text());timelines=json.loads((run/'timeline.json').read_text())
    variant=next((v for v in summary['variants'] if v['name']=='baseline_smooth50'),summary['variants'][0])
    return {'schema_version':2,'mode':'licensed_local','title':f"{summary['teams']['H']} vs {summary['teams']['A']}",
            'model_name':'Official reference baseline · not the trained RNN','n_frames':summary['n_frames'],
            'fps':summary['fps'],'blocks':summary['blocks'],'teams':summary['teams'],
            'timeline':timelines[variant['name']],'tracklets':None}


def export_html(out,payload=None):
    payload=demo_payload() if payload is None else payload
    web=Path(__file__).parent/'web';html=(web/'index.html').read_text()
    html=html.replace('<link rel="stylesheet" href="/style.css">','<style>'+(web/'style.css').read_text()+'</style>')
    html=html.replace('<script src="/app.js" defer></script>','')
    data=json.dumps(payload,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    html=html.replace('</body>','<script>window.__POSSESSION_LENS_DATA__='+data+';</script><script>'+(web/'app.js').read_text()+'</script></body>')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(html)
    return out


def export_local(run_dir,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);payload=from_run(run_dir)
    (out/'match-analysis.json').write_text(json.dumps(payload,allow_nan=False)+'\n')
    export_html(out/'dashboard.html',payload)
    return out/'dashboard.html'
