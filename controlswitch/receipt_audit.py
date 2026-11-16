"""Frozen064 component checks; no tuning or pipeline promotion."""
from pathlib import Path
import json
import numpy as np
from .receipt_experiment import component_metrics, samples, BRANCHES
from .receipt_evidence import receipt_features, FEATURE_NAMES
from .tracker_robustness import CONDITIONS, noisy_tracking
from .external import period_arrays
from .experiment import missing_mask
from .gap_possession import long_mask
from .skillcorner_europe import load_match
from .pipeline import digest
from . import tree_model


def run(root,cache,raw):
    root=Path(root).resolve();out=root/'local/receipt-audit-064';out.mkdir(exist_ok=False);study=root/'local/receipt-064';result=json.loads((study/'report.json').read_text());models=[];thresholds=[]
    for fi in (0,1):
        fd=study/f'fold-{fi}';c=json.loads((fd/'calibration.json').read_text());m={}
        for k in BRANCHES:
            if digest(fd/(k+'.npz')) != c[k]['model_sha256']:raise ValueError('Frozen model changed')
            m[k]=tree_model.load(fd/(k+'.npz'))
        models.append(m);thresholds.append({k:c[k]['threshold'] for k in BRANCHES})
    protocol={'purpose':'Frozen receipt-component robustness and provider transfer; no refitting or threshold changes; no deployment decision.',
              'seeds':{'masked':10516,'long_gap':10716,'tracker_noise':11120},
              'frozen_report_sha256':digest(study/'report.json'),
              'same_provider':'Opposite-game model per original fold; new error draws on same reused Bundesliga games, not fresh matches.',
              'cross_provider':'Existing historical SkillCorner2440; not used for receipt064 fitting/calibration, but previously used elsewhere in project. Average two frozen Bundesliga models, average their frozen thresholds. Exact visible provider owner ID matched against nearest same-team track slot; ignore other same-team candidate if not exact owner; opposite-team near candidates negative. This is provider ball-carrier association proxy, not verified receipts or full H/A/D.',
              'sampling':'Every5frames=5Hz at25Hz; valid trajectory window and candidate distance<=2.5m. No ID recovery or manual labels.',
              'criteria':'New-draw component gate: decoy acceptance <=.9distance-only and owning retention >=distance-only-.02 pooled and per fold. Report transfer descriptively; no threshold reselection from it.',
              'challenge_data_used':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');cols={'distance_only':[0],'motion_receipt':list(range(len(FEATURE_NAMES)))};folds=[];pooled={k:([],[]) for k in BRANCHES}
    for fi,mid in enumerate(('J03WN1','J03WMX')):
        with np.load(next(Path(cache).glob(mid+'-*.npz')),allow_pickle=False) as a:record={k:a[k] for k in a.files}
        points={k:([],[]) for k in BRANCHES}
        for half in (0,1):
            seq=period_arrays(record,half)
            for kind in CONDITIONS:
                s={k:v for k,v in seq.items() if k!='y'};mask=None
                if kind=='masked':mask=missing_mask(len(seq['y']),10516+half)
                elif kind=='long_gap':mask=long_mask(len(seq['y']),10716+half)
                elif kind=='tracker_noise':s=noisy_tracking(seq,11120+half)
                x,v=receipt_features(s,mask);xx,yy=samples(x,v,seq['y'],0,len(x))
                for k in BRANCHES:
                    p=tree_model.predict(models[fi][k],xx[:,cols[k]])[:,1];points[k][0].append(yy);points[k][1].append(p);pooled[k][0].append(yy);pooled[k][1].append((p>=thresholds[fi][k]).astype(float))
                print('Receipt frozen audit',fi,half,kind,len(yy),flush=True)
        folds.append({k:component_metrics(np.concatenate(points[k][0]),np.concatenate(points[k][1]),thresholds[fi][k]) for k in BRANCHES})
    summary={k:component_metrics(np.concatenate(v[0]),np.concatenate(v[1]),.5) for k,v in pooled.items()}
    for m in summary.values():
        for k in ('roc_auc','average_precision','brier'):m.pop(k)
    a=summary['distance_only'];b=summary['motion_receipt'];passed=b['opponent_decoy_acceptance']<=.9*a['opponent_decoy_acceptance'] and b['owning_candidate_retention']>=a['owning_candidate_retention']-.02 and all(f['motion_receipt']['owning_candidate_retention']>=f['distance_only']['owning_candidate_retention']-.02 for f in folds)
    print('Loading frozen cross-provider check2440',flush=True);parts,audit=load_match(raw,'2440',recover_provider_ids=False,include_owner=True);cross={k:([],[]) for k in BRANCHES};coverage=[]
    for half,(seq,pitch) in enumerate(parts):
        x,v=receipt_features({k:value for k,value in seq.items() if k not in ('y','owner')});ix=np.arange(0,len(x),5);owners=seq['owner'];owner_valid=owners>=0;label=np.zeros((len(ix),2),np.int8);eligible=np.zeros_like(label,bool)
        for team in (0,1):
            slots=np.flatnonzero(seq['teams']==team);d=np.linalg.norm(seq['xy'][ix][:,slots]-seq['ball'][ix,None,:],axis=2);nearest=slots[np.where(np.isfinite(d),d,np.inf).argmin(1)];exact=nearest==owners[ix];opposite=np.zeros(len(ix),bool);good=owner_valid[ix];opposite[good]=seq['teams'][owners[ix[good]]]!=team
            eligible[:,team]=good & v[ix,team] & (x[ix,team,0]<=2.5) & (exact|opposite);label[:,team]=exact.astype(np.int8)
        frame,team=np.nonzero(eligible);xx=x[ix[frame],team];yy=label[frame,team]
        for k in BRANCHES:
            p=np.mean([tree_model.predict(m[k],xx[:,cols[k]])[:,1] for m in models],axis=0);cross[k][0].append(yy);cross[k][1].append(p)
        coverage.append({'half':half+1,'sampled_frames':len(ix),'exact_visible_owner_frames':int(owner_valid[ix].sum()),'eligible_candidate_points':len(yy),'positive_owner_candidates':int(yy.sum()),'negative_opponent_candidates':int((yy==0).sum())});print('Transfer',coverage[-1],flush=True)
    transfer={k:component_metrics(np.concatenate(v[0]),np.concatenate(v[1]),float(np.mean([t[k] for t in thresholds]))) for k,v in cross.items()}
    report={'protocol':protocol,'new_draw_folds':folds,'new_draw_pooled':summary,'new_draw_component_gate_passed':bool(passed),'cross_provider':transfer,'cross_provider_coverage':coverage,'provider_loader_audit':audit,'independent_project_test':False,'verified_receipts':False,'promoted_to_default':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('AUDIT',summary,'gate',passed,flush=True);print('TRANSFER',transfer,flush=True);return report

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--raw-root', type=Path)
    args = parser.parse_args()
    run(args.root, args.cache_root or args.root/'data/external/cache',
        args.raw_root or args.root/'data/external/skillcorner-europe/raw')
