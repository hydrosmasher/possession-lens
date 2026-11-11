"""Private, auditable track-proximity exports. No player identity inference."""
from pathlib import Path
import csv,json,math
from collections import Counter

def export_player_report(run_dir,out):
    run=Path(run_dir);out=Path(out)
    ledger=json.loads((run/'tracklet-proxies.json').read_text())
    stats=json.loads((run/'team-statistics.json').read_text())
    rows=ledger['rows'];bouts=ledger.get('bouts',[]);counts=Counter();seconds=Counter();last_end=-1
    keys=[(r['team'],r['track_slot']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate track entries')
    for b in bouts:
        if b['team'] not in 'HA' or b['end_frame_exclusive']<=b['start_frame'] or b['start_frame']<last_end:raise ValueError('Overlapping or invalid control intervals')
        if not math.isclose(b['seconds'],(b['end_frame_exclusive']-b['start_frame'])/25,abs_tol=1e-6):raise ValueError('Invalid interval duration')
        key=(b['team'],b['track_slot'])
        if key not in keys or b['track_slot']=='UNASSIGNED':raise ValueError('Interval has no track')
        counts[key]+=1;seconds[key]+=b['seconds'];last_end=b['end_frame_exclusive']
    exported=[]
    for team in 'HA':
        teamrows=[r for r in rows if r['team']==team];total=stats['counts'][team]/25
        if len([r for r in teamrows if r['track_slot']=='UNASSIGNED'])!=1:raise ValueError('Missing unassigned bucket')
        if not math.isclose(sum(r['associated_control_seconds'] for r in teamrows),total,abs_tol=1e-5):raise ValueError('Team ledger does not reconcile')
        for r in teamrows:
            value=r['associated_control_seconds'];key=(team,r['track_slot'])
            if not math.isfinite(value) or value<0:raise ValueError('Invalid associated time')
            if r['track_slot']!='UNASSIGNED' and not math.isclose(seconds[key],value,abs_tol=1e-5):raise ValueError('Intervals do not reconcile with track time')
            expected=100*value/total if total else None
            if (expected is None)!=(r['percent_of_team_possession'] is None) or expected is not None and not math.isclose(expected,r['percent_of_team_possession'],abs_tol=1e-5):raise ValueError('Incorrect share denominator')
            exported.append({**r,'control_intervals':counts[key] if r['track_slot']!='UNASSIGNED' else None,'identity_status':'anonymous unverified track; proximity is not confirmed contact'})
    out.mkdir(parents=True,exist_ok=True)
    fields=['team','track_slot','associated_control_seconds','percent_of_team_possession','control_intervals','identity_status']
    with (out/'anonymous-track-contributions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(exported)
    with (out/'anonymous-control-intervals.csv').open('w',newline='') as f:
        fields=['team','track_slot','start_frame','end_frame_exclusive','seconds'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(bouts)
    result={'status':'anonymous_track_proximity_only','identities_verified':False,'team_ledgers_reconciled':True,'rows':exported,'unassigned_reasons':ledger.get('unassigned_reasons',{}),'warning':ledger.get('warning'),'identity_linking':'Disabled: external validation safety gate not met. See reports/external/identity-link-002-report.json.'}
    (out/'player-control.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
