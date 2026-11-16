"""062: fixed ownership-ledger comparison on existing external observations."""
from pathlib import Path
import json
import numpy as np
import torch
from .ownership_ledger import association, decode_ledger, load_actor
from .player_set_state import player_nodes
from .external import period_arrays
from .tracker_robustness import CONDITIONS, condition
from .research_snapshot import cached_case
from .control_state import evaluate_accuracy
from .pipeline import digest, pm


def short_hits(y, prediction):
    runs = pm.valid_runs(y)
    gt = pm.possession_spells(y, runs)
    hits = pm.matched_gt_flags(pm.spell_overlaps(gt, pm.possession_spells(prediction, runs)), len(gt), .5)
    return sum(bool(h) for h, (a, b, _) in zip(hits, gt) if b-a < 25)


def run(root, cache, preserve_unsupported=False):
    root = Path(root).resolve(); out = root/('local/ownership-ledger-063' if preserve_unsupported else 'local/ownership-ledger-062')
    out.mkdir(exist_ok=False)
    actor_root = root/'local/actor-061'
    actor_report = json.loads((actor_root/'report.json').read_text())
    if not actor_report['component_gate_passed']:
        raise ValueError('Actor component gate failed')
    checkpoint = actor_root/'actor-encoder.npz'
    protocol = {
        'version': 'ownership-ledger-063' if preserve_unsupported else 'ownership-ledger-062',
        'preserve_unsupported': preserve_unsupported,
        'single_change_063': 'When both ownership supports are zero, preserve frozen H/A/D labels exactly; reset ledger at these intervals. All other parameters and acceptance checks unchanged.',
        'priority': 'Fix ownership and handovers before optimising aggregate score.',
        'reference': 'Frozen053; unchanged D mask and independent live-run resets.',
        'actor_sha256': digest(checkpoint),
        'actor_scope': 'Four existing SkillCorner games trained;2440 actor-only holdout, all previously used elsewhere. Weak provider ball-carrier labels; not verified touches.',
        'variants': ['geometry_ledger', 'actor_ledger'],
        'fixed': 'HC/Hflight/AC/Aflight; emissions qTeam*receipt or qTeam*(1-maxReceipt); same-team transition .25, unsupported opposite flight24, opposite receipt2+22*(1-r). No minimum duration. Receipt=sqrt(actorWeight*physicalSupport); missing ball forces0. See ownership_ledger.py. No parameters fitted on strong benchmark.',
        'acceptance': 'FP reduction>=10%,misses nonincrease,subsecond matched GT nondecrease,mean F1>=reference-.002,each noise fold nondecline,clean/gap per-fold loss<=.01,D mask exact. Highest eligible mean frozen before separate noise audit; no promotion from this first check.',
        'audit': 'Only frozen winner; new noise seeds10515/10715/11119. FP reduction>=5%,misses/short-matches nonworse,mean F1>=reference-.002. These are new observation draws on reused matches, not fresh games.',
        'challenge_data_used': False,
        'source_sha256': digest(root/'controlswitch/ownership_ledger.py'),
    }
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    torch.set_num_threads(1)
    actor = load_actor(checkpoint, protocol['actor_sha256'])
    descriptor = {'source':'short-spell-053','name':'short_focus'}
    rows=[]; base=[]; predictions={k:[] for k in protocol['variants']}; folds=[]
    for fi, mid in enumerate(('J03WN1','J03WMX')):
        paths=list(Path(cache).glob(mid+'-*.npz'))
        if len(paths)!=1: raise ValueError('Ambiguous strong cache')
        with np.load(paths[0],allow_pickle=False) as archive:
            record={k:archive[k] for k in archive.files}
        fold_rows=[]; fold_base=[]; fold_predictions={k:[] for k in predictions}
        for half in (0,1):
            seq=period_arrays(record,half)
            for ci, kind in enumerate(CONDITIONS):
                print('Ledger case',fi,half,kind,flush=True)
                s, mask=condition(seq,kind,half,False)
                y, current, q=cached_case(descriptor,fi,half*4+ci,root/'local')
                np.testing.assert_array_equal(y,seq['y'])
                nodes=player_nodes(s,record['pitch'],mask)
                local={}
                for name, scorer in [('geometry_ledger',None),('actor_ledger',actor)]:
                    _,support=association(nodes,scorer)
                    pred,latent=decode_ledger(q[:,:2],support,current,preserve_unsupported=preserve_unsupported)
                    np.testing.assert_array_equal(pred==2,current==2)
                    if preserve_unsupported:
                        absent=support.max(1)==0
                        np.testing.assert_array_equal(pred[absent],current[absent])
                    local[name]=pred; fold_predictions[name].append(pred)
                np.savez_compressed(out/f'case-{fi}-{half*4+ci}.npz',reference=y,current=current,**local)
                fold_rows.append({'y':y,'condition':kind});fold_base.append(current)
                del nodes
        f={'reference':evaluate_accuracy(fold_rows,fold_base),'variants':{k:evaluate_accuracy(fold_rows,p,fold_base) for k,p in fold_predictions.items()}}
        folds.append(f);rows+=fold_rows;base+=fold_base
        for k in predictions:predictions[k]+=fold_predictions[k]
    reference=evaluate_accuracy(rows,base);base_short=sum(short_hits(r['y'],p) for r,p in zip(rows,base));trials=[]
    for name,p in predictions.items():
        score=evaluate_accuracy(rows,p,base);short=sum(short_hits(r['y'],v) for r,v in zip(rows,p))
        guards={'false_switches':score['totals']['false_positive']<=.9*reference['totals']['false_positive'],'misses':score['totals']['missed']<=reference['totals']['missed'],'short_possessions':short>=base_short,'mean_noninferior':score['mean_spell_f1']>=reference['mean_spell_f1']-.002,'noise':all(f['variants'][name]['tracker_noise']['spell_f1_mean']>=f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),'clean_gap':all(f['variants'][name][c]['spell_f1_mean']>=f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3]),'dead_mask_exact':all(np.array_equal(a==2,b==2) for a,b in zip(p,base))}
        trials.append({'name':name,'scores':score,'short_matched':short,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None
    report={'protocol':protocol,'reference':reference,'reference_short_matched':base_short,'folds':folds,'trials':trials,'selected':selected,'promoted_to_default':False,'audit_required_before_promotion':bool(selected),'independent_test':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials:print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],t['guards'],flush=True)
    print('Selected',None if selected is None else selected['name'],flush=True)
    return report

if __name__=='__main__':
    run(Path.cwd(),Path('/Users/hydrosmasher/Desktop/Possession Lens/data/external/cache'))
