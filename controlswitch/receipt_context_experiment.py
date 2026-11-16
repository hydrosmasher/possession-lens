"""077: external-only same-player receipt context, with short-spell calibration."""
from pathlib import Path
import argparse
import json
import numpy as np
from .receipt_context import context_features, gate_recoveries, short_retention_threshold, FEATURE_NAMES
from .receipt_evidence import receipt_features
from .receipt_experiment import component_metrics
from .external import period_arrays
from .tracker_robustness import CONDITIONS, condition
from .short_proposals import recover_short
from .proposal_calibration import retention, flags
from .local_temporal_experiment import short_matches
from .control_state import evaluate_accuracy
from .pipeline import pm, digest
from . import tree_model

BRANCHES = ('matched14', 'context34')


def record_for(cache, mid):
    files = list(Path(cache).glob(mid+'-*.npz'))
    if len(files) != 1: raise ValueError('Ambiguous external cache')
    with np.load(files[0], allow_pickle=False) as a:
        record = {k: a[k] for k in a.files}
    if str(record['match_id']) != mid: raise ValueError('Wrong external match')
    return record


def old_support(sequence, mask, old_model, threshold):
    x, v = receipt_features(sequence, mask)
    p = tree_model.predict(old_model, x.reshape(-1, 14))[:, 1].reshape(-1, 2)
    return v & (x[:, :, 0] <= 2.5) & (p >= threshold), x, v, p


def short_flags(y, p):
    spells = pm.possession_spells(y, pm.valid_runs(y))
    return flags(y, p) & np.array([b-a <= 75 for a, b, _ in spells], bool)


def run(root, cache):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    root, cache = Path(root).resolve(), Path(cache).resolve()
    out = root/'local/receipt-context-077'; out.mkdir(exist_ok=False)
    protocol = {
        'version': 'receipt-context-077',
        'motivation': '076geometry veto lost already detected genuine spells. Test richer same-slot receipt compatibility only on novel070recovery proposals; baseline053labels outside proposals immutable.',
        'source': 'Existing IDSSE/Sportec first-division Bundesliga WMX/WN1, CC BY4.0. No challenge assets/video/labels/publicscore, new downloads, or Barca text labels.',
        'label_limit': 'Near-player team equals external live ownership, a proxy for receipt compatibility, not verified touches. Genuine short reference possession spells constrain calibration.',
        'features': list(FEATURE_NAMES),
        'context': 'Offline +/-24frames(.96s at25Hz), same nearest current slot followed through full window; complete ball and candidate track required. Ball/player co-motion, competition distances and approach/departure. Coordinates/teams only; no identities/clock/labels.',
        'split': 'Opposite-match folds. First70percent of EACH fitting half, ending50frames beforecut, fits; last30percent starting50frames aftercut calibrates. Allfour frozen training-corruption conditions; every5th eligible known-live frame withcandidate<=2.5m. Both branches use exactly same rows and validity.',
        'branches': {'matched14': 'Original14columns on contextual-valid rows, matched training control', 'context34': 'All34columns, only promotable branch'},
        'training': 'HGB120iterations,max15leaves,minleaf100,learningrate.05,L2=20,seed177,noearlystop; fixed before evaluation. Fit/export native parity asserted.',
        'calibration': 'For each complete known-live reference shortspell<=75frames fullyinsidecalibrationregion, collect old064-supported frames. Ifall suchsupportframes context-valid, peaknewscore must survive. Cutoff=min ALL positive shortspellpeaks over allfittinghalf/conditioncalibrationgroups; no groups=>cutoff0/no filtering. Thresholds/model hashes saved BEFORE opposite-match evaluation. No evaluation score threshold sweep.',
        'gate': 'Only070acceptednovel H/A proposals1..75frames withinoneopposite053spell; require old064support. Keep ifanyold-supportedframe newscore>=cutoff. Ifanyoldsupportframe unavailablelongcontext, abstain/keep. Never filterexisting053pulse. No newcandidate, durationfloor, smoothing, D/missingchanges.',
        'acceptance': 'context34only; mean spellF1>=053+.002 and>=matched14; FP<070and<=053; misses<=053; matchedshort>053; zero loss ofsame070recoveredindividualspells AND zero loss ofANYshortspellmatchedby070; no additional053matchedspelllosscomparedto070. Eachtracker_noisefoldnondeclinevs053; eachclean/masked/longgapfolddecline<=.01; D/missingframesexact. Winnerrequiresfrozendifferent-error-drawauditbeforepromotion.',
        'limitations': 'Twoheavilyreusedexternaldevelopmentgames,notfreshindependenttesting. CalibrationgroupsareGT-derivedproxygroups,notout-of-samplemodelproposals. AllninehistoricalSkillCornermatchespreviouslyseenandpartiallabels; notfreshfullHADtest.',
        'source_sha256': {name: digest(root/'controlswitch'/name) for name in ('receipt_context.py', 'receipt_context_experiment.py')},
        'input_report_sha256': digest(root/'local/short-proposals-070/report.json')}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    allrows, allbase, allproto = [], [], []
    allpred = {k: [] for k in BRANCHES}
    components = {k: {'y': [], 'p': []} for k in BRANCHES}
    folds = []
    with threadpool_limits(limits=1):
      for fi, (train, val) in enumerate((('J03WMX', 'J03WN1'), ('J03WN1', 'J03WMX'))):
        print('Receipt context fold', fi, train, '->', val, flush=True)
        fd = out/f'fold-{fi}'; fd.mkdir()
        oldroot = root/f'local/receipt-064/fold-{fi}'
        rc = json.loads((oldroot/'calibration.json').read_text())['motion_receipt']
        if digest(oldroot/'motion_receipt.npz') != rc['model_sha256']: raise ValueError('Old receipt changed')
        old = tree_model.load(oldroot/'motion_receipt.npz')
        record = record_for(cache, train)
        fitx, fity, calx, caly, groups = [], [], [], [], []
        for half in (0, 1):
          seq = period_arrays(record, half); cut = int(.7*len(seq['y']))
          for kind in CONDITIONS:
            print('Training features', fi, half, kind, flush=True)
            s, mask = condition(seq, kind, half, True)
            x, valid = context_features(s, mask)
            support, _, _, _ = old_support(s, mask, old, rc['threshold'])
            for start, end, xx, yy in ((0, cut-50, fitx, fity), (cut+50, len(x), calx, caly)):
                ix = np.arange(start, end, 5)
                frame, team = np.nonzero(valid[ix] & (x[ix, :, 0] <= 2.5) & (seq['y'][ix, None] < 2))
                xx.append(x[ix[frame], team]); yy.append((seq['y'][ix[frame]] == team).astype(np.int8))
            for a, b, k in pm.possession_spells(seq['y'], pm.valid_runs(seq['y'])):
                if a < cut+50 or b-a > 75: continue
                use = support[a:b, k]
                if use.any() and valid[a:b, k][use].all(): groups.append(x[a:b, k][use].copy())
        fx, fy = np.concatenate(fitx), np.concatenate(fity)
        cx, cy = np.concatenate(calx), np.concatenate(caly)
        del fitx, fity, calx, caly, record, x, valid, support
        models, freezes = {}, {}
        for name, width in (('matched14', 14), ('context34', len(FEATURE_NAMES))):
            model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, min_samples_leaf=100, learning_rate=.05, l2_regularization=20, random_state=177, early_stopping=False)
            model.fit(fx[:, :width], fy)
            path = fd/(name+'.npz'); tree_model.save(path, model, 'hist_boost'); exported = tree_model.load(path)
            p = model.predict_proba(cx[:, :width])[:, 1]
            np.testing.assert_allclose(tree_model.predict(exported, cx[:, :width])[:, 1], p, rtol=1e-12, atol=1e-12)
            peaks = [float(tree_model.predict(exported, g[:, :width])[:, 1].max()) for g in groups]
            threshold = short_retention_threshold(peaks)
            freeze = {'train_match': train, 'evaluate_match': val, 'fit_points': len(fy), 'fit_positive': int(fy.sum()),
                      'calibration_points': len(cy), 'calibration_short_groups': len(groups), 'calibration_short_peaks': peaks,
                      'threshold': threshold, 'model_sha256': digest(path), 'component': component_metrics(cy, p, threshold),
                      'before_opposite_match_evaluation': True}
            models[name] = exported; freezes[name] = freeze
            (fd/(name+'-freeze.json')).write_text(json.dumps(freeze, indent=2)+'\n')
            print('FROZEN', name, 'fit', len(fy), 'groups', len(groups), 'threshold', threshold, flush=True)
        del fx, fy, cx, cy, groups, model
        record = record_for(cache, val)
        rows, base, proto = [], [], []
        pred = {k: [] for k in BRANCHES}; reviews = []; comps = {k: {'y': [], 'p': []} for k in BRANCHES}
        losses = {k: {'short_spells_lost_vs_070': 0, 'additional_053_spells_lost_vs_070': 0, 'prototype_recovered': 0, 'retained_recovered': 0, 'lost_recovered': 0} for k in BRANCHES}
        for half in (0, 1):
          seq = period_arrays(record, half)
          for ci, kind in enumerate(CONDITIONS):
            print('Frozen receipt evaluation', fi, half, kind, flush=True)
            s, mask = condition(seq, kind, half, False)
            x, valid = context_features(s, mask)
            support, motion, rv, rp = old_support(s, mask, old, rc['threshold'])
            seen = np.isfinite(s['ball']).all(1)
            if mask is not None: seen &= ~mask
            with np.load(root/f'local/local-temporal-069/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a:
                y, current, proposal = (a[k].copy() for k in ('reference', 'current', 'local_expert'))
            np.testing.assert_array_equal(y, seq['y'])
            prototype, events = recover_short(current, proposal, seen, rp, rv, motion[:, :, 0] <= 2.5, rc['threshold'])
            events = [e for e in events if e['accepted']]
            with np.load(root/f'local/short-proposals-070/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a:
                np.testing.assert_array_equal(prototype, a['receipt_supported'])
            original_flags, proto_flags = flags(y, current), flags(y, prototype)
            proto_short = short_flags(y, prototype)
            ix = np.arange(0, len(y), 5)
            frame, team = np.nonzero(valid[ix] & (x[ix, :, 0] <= 2.5) & (y[ix, None] < 2))
            component_y = (y[ix[frame]] == team).astype(np.int8)
            case = {}
            for name, width in (('matched14', 14), ('context34', len(FEATURE_NAMES))):
                scores = tree_model.predict(models[name], x.reshape(-1, len(FEATURE_NAMES))[:, :width])[:, 1].reshape(-1, 2)
                p, review = gate_recoveries(current, events, support, valid, scores, freezes[name]['threshold'])
                np.testing.assert_array_equal(p == 2, current == 2)
                np.testing.assert_array_equal(p[~seen], current[~seen])
                outside = np.ones(len(y), bool)
                for e in events: outside[e['start']:e['end']] = False
                np.testing.assert_array_equal(p[outside], current[outside])
                candidate_flags = flags(y, p)
                loss = retention(y, current, prototype, p)
                loss['short_spells_lost_vs_070'] = int((proto_short & ~short_flags(y, p)).sum())
                loss['additional_053_spells_lost_vs_070'] = int((original_flags & proto_flags & ~candidate_flags).sum())
                for k, v in loss.items(): losses[name][k] += v
                cp = scores[ix[frame], team]
                comps[name]['y'].append(component_y); comps[name]['p'].append(cp)
                pred[name].append(p); case[name] = p
                reviews.append({'half': half+1, 'condition': kind, 'branch': name, 'events': review,
                                'proposals': len(events), 'rejected': sum(not e['kept'] for e in review),
                                'abstained': sum(not e['context_complete'] for e in review), 'losses': loss})
            rows.append({'y': y, 'condition': kind}); base.append(current); proto.append(prototype)
            np.savez_compressed(fd/f'validation-{half*4+ci}.npz', reference=y, current=current, prototype=prototype, **case)
        fold = {'train': train, 'evaluate': val, 'freeze': freezes, 'reference': evaluate_accuracy(rows, base),
                'prototype': evaluate_accuracy(rows, proto, base), 'variants': {k: evaluate_accuracy(rows, v, base) for k, v in pred.items()},
                'component': {k: component_metrics(np.concatenate(comps[k]['y']), np.concatenate(comps[k]['p']), freezes[k]['threshold']) for k in BRANCHES},
                'retention': losses, 'reviews': reviews}
        folds.append(fold); (fd/'report.json').write_text(json.dumps(fold, indent=2)+'\n')
        allrows.extend(rows); allbase.extend(base); allproto.extend(proto)
        for k in BRANCHES:
            allpred[k].extend(pred[k]); components[k]['y'].extend(comps[k]['y'])
            components[k]['p'].extend([(p >= freezes[k]['threshold']).astype(float) for p in comps[k]['p']])
    reference, prototype = evaluate_accuracy(allrows, allbase), evaluate_accuracy(allrows, allproto, allbase)
    short0 = sum(short_matches(r['y'], p) for r, p in zip(allrows, allbase))
    scores = {k: evaluate_accuracy(allrows, p, allbase) for k, p in allpred.items()}
    trials = []
    for name in BRANCHES:
        score = scores[name]; sh = sum(short_matches(r['y'], p) for r, p in zip(allrows, allpred[name]))
        lost = {k: sum(f['retention'][name][k] for f in folds) for k in folds[0]['retention'][name]}
        guards = {'spell_gain': score['mean_spell_f1'] >= reference['mean_spell_f1']+.002,
                  'matched_control': score['mean_spell_f1'] >= scores['matched14']['mean_spell_f1'],
                  'fp_reduction': score['totals']['false_positive'] < prototype['totals']['false_positive'],
                  'fp_no_increase_vs_053': score['totals']['false_positive'] <= reference['totals']['false_positive'],
                  'misses': score['totals']['missed'] <= reference['totals']['missed'],
                  'short_gain': sh > short0, 'same_recoveries': lost['lost_recovered'] == 0,
                  'all_short_retained': lost['short_spells_lost_vs_070'] == 0,
                  'existing_retained': lost['additional_053_spells_lost_vs_070'] == 0,
                  'noise': all(f['variants'][name]['tracker_noise']['spell_f1_mean'] >= f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),
                  'clean_gap': all(f['variants'][name][c]['spell_f1_mean'] >= f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3])}
        trials.append({'name': name, 'scores': score, 'short_matched': sh, 'retention': lost, 'guards': guards, 'eligible': name == 'context34' and all(guards.values())})
    eligible = [t for t in trials if t['eligible']]
    pooled = {k: component_metrics(np.concatenate(v['y']), np.concatenate(v['p']), .5) for k, v in components.items()}
    for v in pooled.values():
        for k in ('roc_auc', 'average_precision', 'brier'): v.pop(k)
    report = {'protocol': protocol, 'folds': folds, 'reference': reference, 'prototype': prototype,
              'reference_short_matched': short0, 'trials': trials, 'pooled_component_decisions': pooled,
              'selected': eligible[0] if eligible else None, 'promoted_to_default': False,
              'independent_test': False, 'verified_receipts': False, 'frozen_audit_required': bool(eligible)}
    (out/'freeze.json').write_text(json.dumps({'selected': report['selected'], 'before_new_draw_audit': True}, indent=2)+'\n')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for t in trials: print('RESULT', t['name'], t['scores']['mean_spell_f1'], t['scores']['totals'], 'short', t['short_matched'], t['retention'], t['guards'], flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--cache-root', type=Path, required=True)
    a = p.parse_args(); run(a.root, a.cache_root)
