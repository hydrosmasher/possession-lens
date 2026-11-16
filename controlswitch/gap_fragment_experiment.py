"""081: guard against new gap-separated fragments of existing control."""
from pathlib import Path
import argparse
import json
import numpy as np
from .gap_fragment import filter_gap_fragments
from .receipt_context_experiment import record_for, old_support, short_flags
from .research_snapshot import cached_case
from .external import period_arrays
from .tracker_robustness import CONDITIONS, condition
from .proposal_calibration import retention, flags, apply_scores
from .local_temporal_experiment import short_matches
from .control_state import evaluate_accuracy
from .pipeline import digest
from . import tree_model

BRANCHES = {'gap_fragment': 0}


def run(root, cache):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    out = root/'local/gap-fragment-081'; out.mkdir(exist_ok=False)
    previous = json.loads((root/'local/tracking-stability-079/report.json').read_text())
    protocol = {
        'version': 'gap-fragment-081',
        'motivation': 'External diagnosis shows remaining noisyWMXpulse19correct-ownerframesinside172frameGTspell,notafalse receipt. Test unsupported-gap fragment veto instead of requiringevenstrongerreceiptconfidence. Avoidaddingaseparatepulsewithoutchangingmissinglabels.',
        'data': previous['protocol']['source'],
        'starting_candidate': '077matched14, exactlyreplayedfrom079casecache. 079proposalstart/end/team only; noGT-derivedeventselection.',
        'observation_policy': 'Nearestsource053same-teamspellwithin75framebridge;bridgealloppositelabels,majoritymissingball,andzeroopposingold064receipt-supportframes. No D crossings, gapfill, or added switches.',
        'rule': 'Onlyexisting077novelrecoveries<=75frames. Vetoifoneadjacentsource053same-teamspellseparatedbya1..75framebridgewith>=.5missingfractionandNOopposingreceipt-supportframe. Otherwisekeep. Bound75reusesexisting3sproposalbound;majority=.5fixedbefore081scoring. Appliesallconditionsusingobservationsonly;no conditionflagsinfeatures.',
        'branches': BRANCHES,
        'selection': previous['protocol']['selection'],
        'history': 'Adaptivefollow-upafter079/080andexternalerrorattributionontwoheavilyreusedBundesligagames/draws,notindependenttest. No fitting,thresholdsweep,challengeassets/video/labels/publicscore. Thresholdsnotfittochallengeorselectedbysweep.',
        'source_sha256': {n: digest(root/'controlswitch'/n) for n in ('gap_fragment.py', 'gap_fragment_experiment.py')},
        'input_report_sha256': digest(root/'local/tracking-stability-079/report.json')}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows_all, base_all, proto_all, start_all = [], [], [], []
    preds_all = {k: [] for k in BRANCHES}; folds = []
    desc = {'source': 'short-spell-053', 'name': 'short_focus'}
    for fi, mid in enumerate(('J03WN1', 'J03WMX')):
        fd = out/f'fold-{fi}'; fd.mkdir(); record = record_for(cache, mid)
        oldroot = root/f'local/receipt-064/fold-{fi}'
        rc = json.loads((oldroot/'calibration.json').read_text())['motion_receipt']
        if digest(oldroot/'motion_receipt.npz') != rc['model_sha256']: raise ValueError('Receipt changed')
        old = tree_model.load(oldroot/'motion_receipt.npz')
        rows, base, proto, start, reviews = [], [], [], [], []
        preds = {k: [] for k in BRANCHES}
        losses = {k: {'short_spells_lost_vs_070': 0, 'additional_053_spells_lost_vs_070': 0, 'prototype_recovered': 0, 'retained_recovered': 0, 'lost_recovered': 0} for k in BRANCHES}
        for half in (0, 1):
          seq = period_arrays(record, half)
          for ci, kind in enumerate(CONDITIONS):
            print('Gap fragment case', fi, half, kind, flush=True)
            s, mask = condition(seq, kind, half, False)
            support, _, _, _ = old_support(s, mask, old, rc['threshold'])
            y, frozen, q = cached_case(desc, fi, half*4+ci, root/'local')
            np.testing.assert_array_equal(y, seq['y'])
            with np.load(root/f'local/tracking-stability-079/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a:
                np.testing.assert_array_equal(frozen, a['current']); current = a['starting_candidate'].copy(); prototype = a['prototype'].copy()
            source = next(v for v in previous['folds'][fi]['reviews'] if v['half'] == half+1 and v['condition'] == kind and v['variant'] == 'dedup_consensus')
            events = [{k: e[k] for k in ('start', 'end', 'team')} for e in source['events']]
            np.testing.assert_array_equal(apply_scores(frozen, events, np.ones(len(events)), .5), current)
            seen = np.isfinite(s['ball']).all(1)
            if mask is not None: seen &= ~mask
            case = {}
            for name, tolerance in BRANCHES.items():
                p, review = filter_gap_fragments(frozen, current, events, seen, support)
                np.testing.assert_array_equal(p == 2, frozen == 2)
                np.testing.assert_array_equal(p[~seen], frozen[~seen])
                outside = np.ones(len(y), bool)
                for e in events: outside[e['start']:e['end']] = False
                np.testing.assert_array_equal(p[outside], current[outside])
                loss = retention(y, frozen, prototype, p)
                loss['short_spells_lost_vs_070'] = int((short_flags(y, prototype) & ~short_flags(y, p)).sum())
                loss['additional_053_spells_lost_vs_070'] = int((flags(y, frozen) & flags(y, prototype) & ~flags(y, p)).sum())
                for k, v in loss.items(): losses[name][k] += v
                reviews.append({'half': half+1, 'condition': kind, 'variant': name, 'gap_guard': True, 'events': review,
                                'vetoes': sum(e['veto'] for e in review), 'retention': loss})
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
