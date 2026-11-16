"""079: test recovery invariance to persistent duplicate tracks and jitter."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from .tracking_stability import deduplicate, smooth_visible, filter_unstable
from .local_temporal import load, symmetric
from .receipt_context_experiment import record_for, old_support, short_flags
from .external import period_arrays
from .tracker_robustness import CONDITIONS, condition
from .short_proposals import recover_short
from .proposal_calibration import retention, flags
from .local_temporal_experiment import short_matches
from .control_state import evaluate_accuracy
from .pipeline import digest
from . import tree_model

BRANCHES = ('dedup_consensus', 'dedup_median_consensus')


def run(root, cache):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    out = root/'local/tracking-stability-079'; out.mkdir(exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    protocol = {
        'version': 'tracking-stability-079',
        'hypothesis': 'Tracker duplicate slots and jitter can manufacture short local-evidence spikes. Test whether a novel recovery has any receipt-supported local-majority evidence after observation cleanup.',
        'source': 'Existing IDSSE/Sportec first-division Bundesliga WMX/WN1, CC BY4.0. Opposite-match frozen069teacher; no new fitting, downloads, challenge inputs/video/labels/publicscore.',
        'starting_candidate': '077matched14, score.632060/FP1135; source events are its surviving070novel recovery proposals. Compare053and070too.',
        'duplicate_rule': 'Same-team persistent trajectories: >=250co-visibleframes,co-visible/unionvisibility>=.95,separation95thpercentile<=.5m. Prefermostvisiblethensmoothesttrack. No jersey/name mapping or reference labels. Whole-half offline rule, not a synthetic-condition flag.',
        'variants': list(BRANCHES),
        'jitter_rule': 'Three-frame median of ball/player coordinates and stored speeds onlywhereallthreevaluesfinite; neverinterpolate/fillgaps or use maskedballpoints. Apply onlywhen persistent duplicatesfound;otherwiseexact077candidate.',
        'gate': 'Keepnewcandidateunless raw069conditionalcandidatepeak>.5onsomeold064receipt-supportedframeANDcleaned069candidateprobability<.5onALLsameold-supportframes. No duplicates/no support/ties=>keep. D/missingball/outsideproposalframesexact. No minimumduration,new switch,thresholdsweepor evaluationlabelinfeatures.',
        'selection': 'Highestmean amongpredeclaredvariantswithmean>=053+.002,FP<077and<=053,misses<=053,shortmatches>053;zero lost070recoveredindividualspells,zero070matchedshortspellloss,noadditional053lossvs070. Eachnoisefoldnondeclinevs053;clean/masked/longgapfoldloss<=.01. Passingwinnerrequiresfrozendifferent-error-drawauditbeforepromotion.',
        'history': 'Adaptiveafter077/078;twoheavilyreusedexternalgames/errors,notfreshindependenttest. Targetedfailureknownfromaggregate noisyWMXscore; no rule thresholds selected on evaluation labels.',
        'source_sha256': {n: digest(root/'controlswitch'/n) for n in ('tracking_stability.py', 'tracking_stability_experiment.py')},
        'teacher_sha256': [digest(root/f'local/local-temporal-069/fold-{fi}/local-expert.npz') for fi in (0, 1)],
        'starting_report_sha256': digest(root/'local/receipt-context-077/report.json')}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows_all, base_all, proto_all, start_all = [], [], [], []
    preds_all = {k: [] for k in BRANCHES}; folds = []
    for fi, mid in enumerate(('J03WN1', 'J03WMX')):
        fd = out/f'fold-{fi}'; fd.mkdir()
        record = record_for(cache, mid)
        teacher = load(root/f'local/local-temporal-069/fold-{fi}/local-expert.npz')
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
            print('Stability case', fi, half, kind, flush=True)
            s, mask = condition(seq, kind, half, False)
            support, motion, rv, rp = old_support(s, mask, old, rc['threshold'])
            seen = np.isfinite(s['ball']).all(1)
            if mask is not None: seen &= ~mask
            with np.load(root/f'local/local-temporal-069/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a:
                y, frozen, proposal, raw = (a[k].copy() for k in ('reference', 'current', 'local_expert', 'expert'))
            np.testing.assert_array_equal(y, seq['y'])
            prototype, events = recover_short(frozen, proposal, seen, rp, rv, motion[:, :, 0] <= 2.5, rc['threshold'])
            with np.load(root/f'local/receipt-context-077/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a:
                np.testing.assert_array_equal(prototype, a['prototype']); current = a['matched14'].copy()
            events = [e for e in events if e['accepted'] and (current[e['start']:e['end']] == e['team']).all()]
            cleaned, duplicates = deduplicate(s)
            case = {}
            for name in BRANCHES:
                if duplicates and events:
                    observations = cleaned if name == 'dedup_consensus' else smooth_visible(cleaned, mask)
                    print('Rechecking local evidence', fi, half, kind, name, 'duplicates', len(duplicates), 'proposals', len(events), flush=True)
                    g = symmetric(teacher, observations, record['pitch'], mask)
                else: g = raw
                p, review = filter_unstable(frozen, current, events, support, raw, g, bool(duplicates))
                np.testing.assert_array_equal(p == 2, frozen == 2)
                np.testing.assert_array_equal(p[~seen], frozen[~seen])
                outside = np.ones(len(y), bool)
                for e in events: outside[e['start']:e['end']] = False
                np.testing.assert_array_equal(p[outside], current[outside])
                loss = retention(y, frozen, prototype, p)
                loss['short_spells_lost_vs_070'] = int((short_flags(y, prototype) & ~short_flags(y, p)).sum())
                loss['additional_053_spells_lost_vs_070'] = int((flags(y, frozen) & flags(y, prototype) & ~flags(y, p)).sum())
                for k, v in loss.items(): losses[name][k] += v
                reviews.append({'half': half+1, 'condition': kind, 'variant': name, 'duplicates': duplicates,
                                'events': review, 'vetoes': sum(e['veto'] for e in review), 'retention': loss})
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
