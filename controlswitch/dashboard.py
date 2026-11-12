"""One offline dashboard for shareable demo data and private local match analyses."""
from pathlib import Path
import json
from .pipeline import BLOCKS,FPS,pm,describe


def demo_payload():
    """The public demo always uses externally mapped player identities."""
    library=Path(__file__).resolve().parents[1]/'examples/match-library.json'
    if library.exists():
        matches=json.loads(library.read_text())['matches']
        return next(p for p in matches if p.get('example_id')=='skillcorner-2841')
    path=Path(__file__).resolve().parents[1]/'examples/external-named-analysis.json'
    if not path.exists():raise ValueError('The named external demo is missing; restore the examples folder.')
    payload=json.loads(path.read_text())
    if payload.get('mode')!='external_named' or payload.get('named_players',{}).get('source_kind')!='external_provider_stable_ids':raise ValueError('The demo requires external provider roster mappings')
    return payload


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
    named_path=web.parents[1]/'examples/external-named-analysis.json'
    named_script=''
    if named_path.exists():
        named=json.loads(named_path.read_text())
        if named.get('mode')!='external_named' or named.get('named_players',{}).get('source_kind')!='external_provider_stable_ids':raise ValueError('Bundled name example must use external provider identities')
        named_data=json.dumps(named,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
        named_script='window.__POSSESSION_LENS_NAMED_EXAMPLE__='+named_data+';'
    library_script=''
    library_path=web.parents[1]/'examples/match-library.json'
    if library_path.exists():
        library=json.loads(library_path.read_text())
        if any(p.get('mode')!='external_named' or p.get('named_players',{}).get('source_kind')!='external_provider_stable_ids' for p in library['matches']):raise ValueError('Example library requires provider-mapped external players')
        library_script='window.__POSSESSION_LENS_LIBRARY__='+json.dumps(library,allow_nan=False,separators=(',',':')).replace('<','\\u003c')+';'
    research_script=''
    research_path=web.parents[1]/'reports/external/rnn-interaction-context-001-report.json'
    if research_path.exists():
        r=json.loads(research_path.read_text());candidate=r.get('selected')
        if candidate:research_script='window.__POSSESSION_LENS_RESEARCH__='+json.dumps({'reference':r['reference_validation']['selection_spell_f1'],'candidate':candidate['selection_spell_f1']})+';'
    evidence_script=''
    evidence_path=web.parents[1]/'reports/external/rnn-europe-context-001-report.json'
    if evidence_path.exists():
        from .evidence import summary
        evidence_data=json.dumps(summary(json.loads(evidence_path.read_text())),allow_nan=False,separators=(',',':')).replace('<','\\u003c')
        evidence_script='window.__POSSESSION_LENS_EVIDENCE__='+evidence_data+';'
    html=html.replace('</body>','<script>window.__POSSESSION_LENS_DATA__='+data+';'+named_script+library_script+research_script+evidence_script+'</script><script>'+(web/'app.js').read_text()+'</script></body>')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(html)
    return out


def export_local(run_dir,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);payload=from_run(run_dir)
    (out/'match-analysis.json').write_text(json.dumps(payload,allow_nan=False)+'\n')
    export_html(out/'dashboard.html',payload)
    return out/'dashboard.html'
