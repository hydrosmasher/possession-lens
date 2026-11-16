"""082: reattach visible recovered fragments using fixed model/path evidence."""
from pathlib import Path
import argparse
import json
import numpy as np
from .visible_bridge import join_visible_bridges
from .receipt_context_experiment import record_for, old_support, short_flags
from .research_snapshot import cached_case
from .external import period_arrays
from .tracker_robustness import CONDITIONS, condition
from .proposal_calibration import retention, flags, apply_scores
from .local_temporal_experiment import short_matches
from .control_state import evaluate_accuracy
from .pipeline import digest
from . import tree_model

BRANCHES = {'visible_bridge': 0}


def run(root, cache):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    out = root/'local/visible-bridge-082'; out.mkdir(exist_ok=False)
    previous = json.loads((root/'local/gap-fragment-081/report.json').read_text())
    protocol = {
        'version': 'visible-bridge-082',
        'motivation': '081gapguard removesonefalsefragmentbutremainingnoisebridgeisvisible. Testboundedjoiningofrecoveryfragmentsusingfixedmodel/pathobjectiveinsteadofrejectingtruecontrol.',
        'data': previous['protocol']['data'],
        'starting_candidate': '081gap_fragment. Survivingnovel077recoveriesonly; exactstartingreplayasserted. NoGT-derivedeventselection.',
        'observation_policy': 'Only1..75frameFULLYVISIBLEbridgesbetweenanexisting053same-teamspellandnovelrecovery;allbridgecurrentlabelsopposite. NoDcrossings,missingframeedits,ornewboundaries.',
        'rule': 'Equalblendconditional053andfrozen069localprobabilities. Joinobjective=sumbridge(logPcandidate-logPopponent)+removedswitchcostsatbridgeends;cost=12*clip(2-2*frozen031boundaryevent,.25,2),sameadaptiveformulaas069. ChoosehighestPOSITIVEsideobjective,atmostonejoinperproposal;nojoinifbothnegative. No fit,weightsweep,orGTinfeatures. Exactall070short/recoveryretentionmandatory;changesoutsideexplicitbridgesforbidden.',
        'branches': BRANCHES,
        'selection': previous['protocol']['selection'],
        'history': 'Adaptiveafter079/080/081andexternalerrorattributionontwoheavilyreusedBundesligagames/draws,notindependenttest. No fitting,thresholdsweep,challengeassets/video/labels/publicscore. Fixedbefore082scoring.',
        'source_sha256': {n: digest(root/'controlswitch'/n) for n in ('visible_bridge.py', 'visible_bridge_experiment.py')},
        'input_report_sha256': digest(root/'local/gap-fragment-081/report.json')}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows_all, base_all, proto_all, start_all = [], [], [], []
    preds_all = {k: [] for k in BRANCHES}; folds = []
    desc = {'source': 'short-spell-053', 'name': 'short_focus'}
    for fi, mid in enumerate(('J03WN1', 'J03WMX')):
        fd = out/f'fold-{fi}'; fd.mkdir(); record = record_for(cache, mid)
        rows, base, proto, start, reviews = [], [], [], [], []
        preds = {k: [] for k in BRANCHES}
        losses = {k: {'short_spells_lost_vs_070': 0, 'additional_053_spells_lost_vs_070': 0, 'prototype_recovered': 0, 'retained_recovered': 0, 'lost_recovered': 0} for k in BRANCHES}
        for half in (0, 1):
          seq = period_arrays(record, half)
          for ci, kind in enumerate(CONDITIONS):
            print('Visible bridge case', fi, half, kind, flush=True)
            s, mask = condition(seq, kind, half, False)
            y, frozen, q = cached_case(desc, fi, half*4+ci, root/'local')
            np.testing.assert_array_equal(y, seq['y'])
            with np.load(root/f'local/gap-fragment-081/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a:
                np.testing.assert_array_equal(frozen, a['current']); current = a['gap_fragment'].copy(); prototype = a['prototype'].copy()
            source = next(v for v in previous['folds'][fi]['reviews'] if v['half'] == half+1 and v['condition'] == kind and v['variant'] == 'gap_fragment')
            events = [{k: e[k] for k in ('start', 'end', 'team')} for e in source['events'] if not e['veto']]
            np.testing.assert_array_equal(apply_scores(frozen, events, np.ones(len(events)), .5), current)
            with np.load(root/f'local/local-temporal-069/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a: local = a['expert'].copy()
            with np.load(root/f'local/precision-055-cache/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a: boundary = a['event'].copy()
            seen = np.isfinite(s['ball']).all(1)
            if mask is not None: seen &= ~mask
            case = {}
            for name, tolerance in BRANCHES.items():
                p, review = join_visible_bridges(frozen, current, events, seen, q[:, :2], local, boundary)
                np.testing.assert_array_equal(p == 2, frozen == 2)
                np.testing.assert_array_equal(p[~seen], frozen[~seen])
                outside = np.ones(len(y), bool)
                for e in events: outside[e['start']:e['end']] = False
                for e in review:
                    if e['join']: outside[e['join']['start']:e['join']['end']] = False
                np.testing.assert_array_equal(p[outside], current[outside])
                loss = retention(y, frozen, prototype, p)
                loss['short_spells_lost_vs_070'] = int((short_flags(y, prototype) & ~short_flags(y, p)).sum())
                loss['additional_053_spells_lost_vs_070'] = int((flags(y, frozen) & flags(y, prototype) & ~flags(y, p)).sum())
                for k, v in loss.items(): losses[name][k] += v
                reviews.append({'half': half+1, 'condition': kind, 'variant': name, 'visible_bridge_guard': True, 'events': review,
                                'joins': sum(e['join'] is not None for e in review), 'retention': loss})
                preds[name].append(p); case[name] = p
            rows.append({'y': y, 'condition': kind}); base.append(frozen); proto.append(prototype); start.append(current)
            np.savez_compressed(fd/f'validation-{half*4+ci}.npz', reference=y, current=frozen, prototype=prototype, starting_candidate=current, **case)
        f = {'evaluate': mid, 'reference': evaluate_accuracy(rows, base), 'prototype': evaluate_accuracy(rows, proto, base),
             'starting_candidate': evaluate_accuracy(rows, start, base), 'variants': {k: evaluate_accuracy(rows, v, base) for k, v in preds.items()},
             'retention': losses, 'reviews': reviews}
        folds.append(f); (fd/'report.json').write_text(json.dumps(f, indent=2)+'\n')
        rows_all.extend(rows); base_all.extend(base); proto_all.extend(proto); start_all.extend(start)
        for k in BRANCHES: preds_all[k].extend(preds[k])
    reference = evaluate_accuracy(rows_all, base_all); starting = evaluate_accuracy(rows_all, start_all, base_all)
    trials = []; short0 = sum(short_matches(r['y'], p) for r, p in zip(rows_all, base_all))
    for name, p in preds_all.items():
        score = evaluate_accuracy(rows_all, p, base_all); sh = sum(short_matches(r['y'], v) for r, v in zip(rows_all, p))
        loss = {k: sum(f['retention'][name][k] for f in folds) for k in folds[0]['retention'][name]}
        guards = {'spell_gain': score['mean_spell_f1'] >= reference['mean_spell_f1']+.002,
                  'fp_reduction': score['totals']['false_positive'] < starting['totals']['false_positive'],
                  'fp_no_increase_vs_053': score['totals']['false_positive'] <= reference['totals']['false_positive'],
                  'misses': score['totals']['missed'] <= reference['totals']['missed'], 'short_gain': sh > short0,
                  'same_recoveries': loss['lost_recovered'] == 0, 'all_short_retained': loss['short_spells_lost_vs_070'] == 0,
                  'existing_retained': loss['additional_053_spells_lost_vs_070'] == 0,
                  'noise': all(f['variants'][name]['tracker_noise']['spell_f1_mean'] >= f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),
                  'clean_gap': all(f['variants'][name][c]['spell_f1_mean'] >= f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3])}
        trials.append({'name': name, 'scores': score, 'short_matched': sh, 'retention': loss, 'guards': guards, 'eligible': all(guards.values())})
    eligible = [t for t in trials if t['eligible']]
    report = {'protocol': protocol, 'reference': reference, 'starting_candidate': starting,
              'prototype': evaluate_accuracy(rows_all, proto_all, base_all), 'reference_short_matched': short0,
              'folds': folds, 'trials': trials,
              'selected': max(eligible, key=lambda t: t['scores']['mean_spell_f1']) if eligible else None,
              'independent_test': False, 'promoted_to_default': False, 'frozen_audit_required': bool(eligible)}
    (out/'freeze.json').write_text(json.dumps({'selected': report['selected'], 'before_new_draw_audit': True}, indent=2)+'\n')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for t in trials: print('RESULT', t['name'], t['scores']['mean_spell_f1'], t['scores']['totals'], 'short', t['short_matched'], t['retention'], t['guards'], flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--cache-root', type=Path, required=True)
    a = p.parse_args(); run(a.root, a.cache_root)
