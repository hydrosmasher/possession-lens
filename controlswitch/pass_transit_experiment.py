"""076: fixed pass-flight veto on licensed external tracking only."""
from pathlib import Path
import argparse
import json
import numpy as np
from .pass_transit import veto_pass_transits
from .external import period_arrays
from .tracker_robustness import CONDITIONS, condition
from .control_state import evaluate_accuracy
from .local_temporal_experiment import short_matches
from .proposal_calibration import retention
from .pipeline import digest


def run(root, cache):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    out = root/'local/pass-transit-076'
    out.mkdir(exist_ok=False)
    protocol = {
        'version': 'pass-transit-076',
        'hypothesis': 'Rayo combinations and Valladolid through-ball descriptions motivate protecting pass continuity against brief opponent proximity; reports do not establish measured tracking patterns.',
        'context_sources': [
            'https://www.fcbarcelona.com/en/news/4086449/valencia-1-2-barca-flick-starts-with-win-as-lewy-scores-a-brace',
            'https://www.fcbarcelona.com/en/football/first-team/news/4098302',
            'https://www.fcbarcelona.com/en/football/first-team/news/4102104/fc-barcelona-7-0-real-valladolid-supersonic/featured'],
        'data': 'Existing IDSSE/Sportec Bundesliga WMX/WN1, CC BY4.0; no new match downloads. No challenge inputs, video, labels, or challenge scores accessed.',
        'limits': 'Two repeatedly used external development matches, not fresh testing. Official Barca reports are qualitative context only, not training labels. No match videos watched or downloaded.',
        'variants': ['current_transit_veto', 'recovery_transit_veto'],
        'rule': 'Fixed before scoring: isolated H/A pulse<=75frames, opposite flanks, different surrounding-team slots within2.5m before/after in25frame windows; all ball frames visible; no D crossing; flight>=12frames; speed>=4m/s; max linear residual<=.75m; max6frame velocity residual<=max(2m/s,.25*speed); closest opponent same-slot relative velocity>=4m/s. Abstain otherwise. No learned coefficients/threshold sweep/team identity prior.',
        'controls': 'Frozen053; receipt-supported070. Existing fitting-match-only weights reused, no new fitting. Only veto existing pulses, never create switches.',
        'selection': 'Mean spellF1 gain>=.002 vs053; FP strictly reduce vs respective input; misses<=053; short matches>=053; no lost070recovered spells for recovery variant; each tracker-noise fold nondecline; clean/masked/long-gap per-fold decline<=.01. D and missing-ball frames exact. Winner requires frozen different-error-draw audit before promotion.',
        'source_sha256': {name: digest(root/'controlswitch'/name) for name in ('pass_transit.py', 'pass_transit_experiment.py')},
        'input_report_sha256': digest(root/'local/short-proposals-070/report.json')}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    names = protocol['variants']
    rows, bases, protos = [], [], []
    preds = {name: [] for name in names}
    folds = []
    ret_total = {'prototype_recovered': 0, 'retained_recovered': 0, 'lost_recovered': 0}
    for fi, mid in enumerate(('J03WN1', 'J03WMX')):
        fd = out/f'fold-{fi}'; fd.mkdir()
        files = list(cache.glob(mid+'-*.npz'))
        if len(files) != 1: raise ValueError('Ambiguous external cache')
        with np.load(files[0], allow_pickle=False) as a: record = {k: a[k] for k in a.files}
        foldrows, foldbase, foldproto = [], [], []
        foldpred = {name: [] for name in names}
        reviews = []
        for half in (0, 1):
            seq = period_arrays(record, half)
            for ci, kind in enumerate(CONDITIONS):
                print('Pass-transit check', fi, half, kind, flush=True)
                s, mask = condition(seq, kind, half, False)
                with np.load(root/f'local/short-proposals-070/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a:
                    y, current, proto = (a[k].copy() for k in ('reference', 'current', 'receipt_supported'))
                np.testing.assert_array_equal(y, seq['y'])
                seen = np.isfinite(s['ball']).all(1)
                if mask is not None: seen &= ~mask
                case = {}
                for name, initial in zip(names, (current, proto)):
                    p, events = veto_pass_transits(initial, s, mask)
                    np.testing.assert_array_equal(p == 2, current == 2)
                    np.testing.assert_array_equal(p[~seen], current[~seen])
                    foldpred[name].append(p); case[name] = p
                    reviews.append({'half': half+1, 'condition': kind, 'variant': name, 'vetoes': sum(e['veto'] for e in events), 'events': events})
                ret = retention(y, current, proto, case['recovery_transit_veto'])
                for k, v in ret.items(): ret_total[k] += v
                foldrows.append({'y': y, 'condition': kind}); foldbase.append(current); foldproto.append(proto)
                np.savez_compressed(fd/f'validation-{half*4+ci}.npz', reference=y, current=current, prototype=proto, **case)
        fold = {'evaluate': mid, 'reference': evaluate_accuracy(foldrows, foldbase),
                'prototype': evaluate_accuracy(foldrows, foldproto, foldbase),
                'variants': {k: evaluate_accuracy(foldrows, v, foldbase) for k, v in foldpred.items()}, 'reviews': reviews}
        folds.append(fold); (fd/'report.json').write_text(json.dumps(fold, indent=2)+'\n')
        rows.extend(foldrows); bases.extend(foldbase); protos.extend(foldproto)
        for name in names: preds[name].extend(foldpred[name])
    reference = evaluate_accuracy(rows, bases)
    prototype = evaluate_accuracy(rows, protos, bases)
    short0 = sum(short_matches(r['y'], p) for r, p in zip(rows, bases))
    trials = []
    for name, p in preds.items():
        score = evaluate_accuracy(rows, p, bases)
        short = sum(short_matches(r['y'], v) for r, v in zip(rows, p))
        initial = reference if name == names[0] else prototype
        guards = {
            'spell_gain': score['mean_spell_f1'] >= reference['mean_spell_f1']+.002,
            'fp_reduction': score['totals']['false_positive'] < initial['totals']['false_positive'],
            'misses': score['totals']['missed'] <= reference['totals']['missed'],
            'short_retention': short >= short0,
            'same_recoveries': name == names[0] or ret_total['lost_recovered'] == 0,
            'noise': all(f['variants'][name]['tracker_noise']['spell_f1_mean'] >= f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),
            'clean_gap': all(f['variants'][name][c]['spell_f1_mean'] >= f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3])}
        trials.append({'name': name, 'scores': score, 'short_matched': short, 'guards': guards, 'eligible': all(guards.values())})
    eligible = [t for t in trials if t['eligible']]
    report = {'protocol': protocol, 'reference': reference, 'prototype': prototype, 'reference_short_matched': short0,
              'folds': folds, 'trials': trials, 'retention': ret_total,
              'selected': max(eligible, key=lambda t: t['scores']['mean_spell_f1']) if eligible else None,
              'promoted_to_default': False, 'independent_test': False, 'frozen_audit_required': bool(eligible)}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for t in trials: print('RESULT', t['name'], t['scores']['mean_spell_f1'], t['scores']['totals'], t['guards'], flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--cache-root', type=Path, required=True)
    a = p.parse_args(); run(a.root, a.cache_root)
