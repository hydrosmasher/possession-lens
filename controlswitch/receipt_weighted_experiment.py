"""078: weight rare short ownership and close-opponent decoys in receipt fit."""
from pathlib import Path
import argparse
import json
import numpy as np
from .receipt_context import context_features, gate_recoveries, short_retention_threshold, FEATURE_NAMES
from .receipt_context_experiment import record_for, old_support, short_flags
from .receipt_experiment import component_metrics
from .external import period_arrays
from .tracker_robustness import CONDITIONS, condition
from .short_proposals import recover_short
from .proposal_calibration import retention, flags
from .local_temporal_experiment import short_matches
from .control_state import evaluate_accuracy
from .pipeline import pm, digest
from . import tree_model


def fitting_weights(y, frames, teams, x):
    """Targets affect fitting weights only, never the observation features."""
    short = np.zeros(len(y), bool)
    for a, b, _ in pm.possession_spells(y, pm.valid_runs(y)):
        if b-a <= 75: short[a:b] = True
    positive = np.asarray(y)[frames] == teams
    weights = np.ones(len(frames), float)
    weights[positive & short[frames]] = 4.
    weights[~positive & (x[:, FEATURE_NAMES.index('opponent_distance')] <= 2.5)] = 2.
    return weights


def run(root, cache):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    root, cache = Path(root).resolve(), Path(cache).resolve()
    out = root/'local/receipt-weighted-078'; out.mkdir(exist_ok=False)
    prior = json.loads((root/'local/receipt-context-077/report.json').read_text())
    protocol = {
        'version': 'receipt-weighted-078',
        'motivation': '077context improves per-frame ranking but minimum genuine-short-spell calibration cutoff rejects no recovery proposals. Reweight training towards rare short truecontrol and hard close-opponent negatives; do not raise cutoff using evaluation labels.',
        'inherits': prior['protocol'],
        'only_fit_change': 'Same34features,rows,70/30split,50frameembargo,HGB120/15leaves/minleaf100/lr.05/L2=20/seed177. Sampleweight4for own-team examplesinsideGTspells<=75frames;weight2for opponentdecoy exampleswhere BOTHteamsnearest<=2.5m;otherwise1. No sweep/earlystop/evaluation-basedcutoff.',
        'controls': 'Frozen053, receipt070 and cached unweighted077context34 under identical cases.',
        'selection': prior['protocol']['acceptance'].replace('context34only', 'weighted34only').replace('>=matched14', '>=unweighted077context34'),
        'history': 'Adaptivefollow-upafter077ontwoheavilyreusedexternalBundesligadevelopmentmatches;notfreshindependenttest. Nochallengeinputs/video/labels/publicscoresornewdownloads.',
        'source_sha256': {n: digest(root/'controlswitch'/n) for n in ('receipt_context.py', 'receipt_context_experiment.py', 'receipt_weighted_experiment.py')},
        'control_report_sha256': digest(root/'local/receipt-context-077/report.json')}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    allrows, allbase, allproto, allcontrol, allpred = [], [], [], [], []
    folds = []; total_loss = {'short_spells_lost_vs_070': 0, 'additional_053_spells_lost_vs_070': 0, 'prototype_recovered': 0, 'retained_recovered': 0, 'lost_recovered': 0}
    with threadpool_limits(limits=1):
      for fi, (train, val) in enumerate((('J03WMX', 'J03WN1'), ('J03WN1', 'J03WMX'))):
        fd = out/f'fold-{fi}'; fd.mkdir()
        oldroot = root/f'local/receipt-064/fold-{fi}'
        rc = json.loads((oldroot/'calibration.json').read_text())['motion_receipt']
        if digest(oldroot/'motion_receipt.npz') != rc['model_sha256']: raise ValueError('Old receipt changed')
        old = tree_model.load(oldroot/'motion_receipt.npz'); record = record_for(cache, train)
        fitx, fity, fitw, calx, caly, groups = [], [], [], [], [], []
        for half in (0, 1):
          seq = period_arrays(record, half); cut = int(.7*len(seq['y']))
          for kind in CONDITIONS:
            print('Weighted training features', fi, half, kind, flush=True)
            s, mask = condition(seq, kind, half, True)
            x, valid = context_features(s, mask); support, _, _, _ = old_support(s, mask, old, rc['threshold'])
            for start, end, xx, yy, fitting in ((0, cut-50, fitx, fity, True), (cut+50, len(x), calx, caly, False)):
                ix = np.arange(start, end, 5)
                frame, team = np.nonzero(valid[ix] & (x[ix, :, 0] <= 2.5) & (seq['y'][ix, None] < 2))
                values = x[ix[frame], team]
                xx.append(values); yy.append((seq['y'][ix[frame]] == team).astype(np.int8))
                if fitting: fitw.append(fitting_weights(seq['y'], ix[frame], team, values))
            for a, b, k in pm.possession_spells(seq['y'], pm.valid_runs(seq['y'])):
                if a < cut+50 or b-a > 75: continue
                use = support[a:b, k]
                if use.any() and valid[a:b, k][use].all(): groups.append(x[a:b, k][use].copy())
        fx, fy, fw = np.concatenate(fitx), np.concatenate(fity), np.concatenate(fitw)
        cx, cy = np.concatenate(calx), np.concatenate(caly)
        del fitx, fity, fitw, calx, caly, record, x, valid, support
        model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, min_samples_leaf=100, learning_rate=.05, l2_regularization=20, random_state=177, early_stopping=False)
        model.fit(fx, fy, sample_weight=fw)
        path = fd/'weighted34.npz'; tree_model.save(path, model, 'hist_boost'); exported = tree_model.load(path)
        p = model.predict_proba(cx)[:, 1]
        np.testing.assert_allclose(tree_model.predict(exported, cx)[:, 1], p, rtol=1e-12, atol=1e-12)
        peaks = [float(tree_model.predict(exported, g)[:, 1].max()) for g in groups]
        threshold = short_retention_threshold(peaks)
        freeze = {'train_match': train, 'evaluate_match': val, 'fit_points': len(fy), 'fit_positive': int(fy.sum()),
                  'weight_counts': {str(w): int((fw == w).sum()) for w in (1, 2, 4)},
                  'calibration_points': len(cy), 'calibration_short_groups': len(groups), 'calibration_short_peaks': peaks,
                  'threshold': threshold, 'model_sha256': digest(path), 'component': component_metrics(cy, p, threshold),
                  'before_opposite_match_evaluation': True}
        (fd/'freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
        print('FROZEN weighted34', fi, 'fit', len(fy), 'groups', len(groups), 'threshold', threshold, flush=True)
        del fx, fy, fw, cx, cy, groups, model
        record = record_for(cache, val)
        rows, base, proto, control, pred, reviews, compy, compp = [], [], [], [], [], [], [], []
        losses = {k: 0 for k in total_loss}
        for half in (0, 1):
          seq = period_arrays(record, half)
          for ci, kind in enumerate(CONDITIONS):
            print('Weighted frozen evaluation', fi, half, kind, flush=True)
            s, mask = condition(seq, kind, half, False)
            x, valid = context_features(s, mask); support, motion, rv, rp = old_support(s, mask, old, rc['threshold'])
            seen = np.isfinite(s['ball']).all(1)
            if mask is not None: seen &= ~mask
            with np.load(root/f'local/local-temporal-069/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a:
                y, current, proposal = (a[k].copy() for k in ('reference', 'current', 'local_expert'))
            np.testing.assert_array_equal(y, seq['y'])
            prototype, events = recover_short(current, proposal, seen, rp, rv, motion[:, :, 0] <= 2.5, rc['threshold'])
            events = [e for e in events if e['accepted']]
            with np.load(root/f'local/receipt-context-077/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a:
                np.testing.assert_array_equal(prototype, a['prototype']); unweighted = a['context34'].copy()
            scores = tree_model.predict(exported, x.reshape(-1, len(FEATURE_NAMES)))[:, 1].reshape(-1, 2)
            p, review = gate_recoveries(current, events, support, valid, scores, threshold)
            np.testing.assert_array_equal(p == 2, current == 2)
            np.testing.assert_array_equal(p[~seen], current[~seen])
            outside = np.ones(len(y), bool)
            for e in events: outside[e['start']:e['end']] = False
            np.testing.assert_array_equal(p[outside], current[outside])
            loss = retention(y, current, prototype, p)
            loss['short_spells_lost_vs_070'] = int((short_flags(y, prototype) & ~short_flags(y, p)).sum())
            loss['additional_053_spells_lost_vs_070'] = int((flags(y, current) & flags(y, prototype) & ~flags(y, p)).sum())
            for k, v in loss.items(): losses[k] += v
            ix = np.arange(0, len(y), 5)
            frame, team = np.nonzero(valid[ix] & (x[ix, :, 0] <= 2.5) & (y[ix, None] < 2))
            compy.append((y[ix[frame]] == team).astype(np.int8)); compp.append(scores[ix[frame], team])
            rows.append({'y': y, 'condition': kind}); base.append(current); proto.append(prototype); control.append(unweighted); pred.append(p)
            reviews.append({'half': half+1, 'condition': kind, 'events': review, 'proposals': len(events),
                            'rejected': sum(not e['kept'] for e in review), 'abstained': sum(not e['context_complete'] for e in review), 'losses': loss})
            np.savez_compressed(fd/f'validation-{half*4+ci}.npz', reference=y, current=current, prototype=prototype, unweighted=unweighted, weighted34=p)
        fold = {'train': train, 'evaluate': val, 'freeze': freeze, 'reference': evaluate_accuracy(rows, base),
                'prototype': evaluate_accuracy(rows, proto, base), 'control': evaluate_accuracy(rows, control, base),
                'candidate': evaluate_accuracy(rows, pred, base), 'component': component_metrics(np.concatenate(compy), np.concatenate(compp), threshold),
                'retention': losses, 'reviews': reviews}
        folds.append(fold); (fd/'report.json').write_text(json.dumps(fold, indent=2)+'\n')
        allrows.extend(rows); allbase.extend(base); allproto.extend(proto); allcontrol.extend(control); allpred.extend(pred)
        for k, v in losses.items(): total_loss[k] += v
    reference, prototype = evaluate_accuracy(allrows, allbase), evaluate_accuracy(allrows, allproto, allbase)
    control, candidate = evaluate_accuracy(allrows, allcontrol, allbase), evaluate_accuracy(allrows, allpred, allbase)
    short0 = sum(short_matches(r['y'], p) for r, p in zip(allrows, allbase))
    sh = sum(short_matches(r['y'], p) for r, p in zip(allrows, allpred))
    guards = {'spell_gain': candidate['mean_spell_f1'] >= reference['mean_spell_f1']+.002,
              'matched_control': candidate['mean_spell_f1'] >= control['mean_spell_f1'],
              'fp_reduction': candidate['totals']['false_positive'] < prototype['totals']['false_positive'],
              'fp_no_increase_vs_053': candidate['totals']['false_positive'] <= reference['totals']['false_positive'],
              'misses': candidate['totals']['missed'] <= reference['totals']['missed'],
              'short_gain': sh > short0, 'same_recoveries': total_loss['lost_recovered'] == 0,
              'all_short_retained': total_loss['short_spells_lost_vs_070'] == 0,
              'existing_retained': total_loss['additional_053_spells_lost_vs_070'] == 0,
              'noise': all(f['candidate']['tracker_noise']['spell_f1_mean'] >= f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),
              'clean_gap': all(f['candidate'][c]['spell_f1_mean'] >= f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3])}
    report = {'protocol': protocol, 'folds': folds, 'reference': reference, 'prototype': prototype, 'control': control,
              'candidate': candidate, 'reference_short_matched': short0, 'short_matched': sh, 'retention': total_loss,
              'guards': guards, 'eligible': all(guards.values()), 'promoted_to_default': False,
              'independent_test': False, 'verified_receipts': False, 'frozen_audit_required': all(guards.values())}
    (out/'freeze.json').write_text(json.dumps({'eligible': report['eligible'], 'before_new_draw_audit': True}, indent=2)+'\n')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('RESULT weighted34', candidate['mean_spell_f1'], candidate['totals'], 'short', sh, total_loss, guards, flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--cache-root', type=Path, required=True)
    a = p.parse_args(); run(a.root, a.cache_root)
