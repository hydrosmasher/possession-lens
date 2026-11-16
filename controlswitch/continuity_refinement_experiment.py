"""083: GRU continuation with historical replay and new external noise draws."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from .continuity_refinement import continuity_loss, residual_live, anchored_decode
from .short_spell_training import open_entry, training_anchor
from .spatial_symmetry import augmented_window
from .gated_rnn import load_checkpoint, save_checkpoint, predict, CORE
from .motion_gru import loss_value
from .joint_state import joint_features, swap_joint
from .receipt_context_experiment import record_for, short_flags
from .research_snapshot import cached_case
from .external import period_arrays
from .tracker_robustness import CONDITIONS, condition, noisy_tracking
from .transfer_corpus import WEAK_IDS
from .proposal_calibration import retention, flags
from .local_temporal_experiment import short_matches
from .control_state import evaluate_accuracy
from .pipeline import digest

BRANCHES = ('replay_ce', 'replay_continuity')


def conditional(bundle, x, sx):
    a, _ = predict(bundle['model'], x, bundle['mean'], bundle['scale'])
    b, _ = predict(bundle['model'], sx, bundle['mean'], bundle['scale'])
    g = ((a+b[:, [1, 0, 2]])*.5)[:, :2]
    return g/np.maximum(g.sum(1, keepdims=True), 1e-12)


def run(root, cache, corpus):
    root, cache, corpus = Path(root).resolve(), Path(cache).resolve(), Path(corpus).resolve()
    out = root/'local/continuity-refinement-083'; out.mkdir(exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    manifest = json.loads((corpus/'manifest.json').read_text())
    if not manifest['complete'] or set(manifest['protocol']['weak_matches']) != set(WEAK_IDS): raise ValueError('Unexpected licensed corpus')
    protocol = {
        'version': 'continuity-refinement-083',
        'aim': 'Fine-tune053GRUtoavoidwhole-possessionfragmentation,notonlyrejectreceipts. ComparematchedordinaryCEwithsame-GT-spelladjacentKL.',
        'data': 'Existing CC BY4.0 IDSSE/Sportec Bundesliga WMX/WN1 plus five historical MIT SkillCorner first-division training matches4039/3749/3518/2068/2440. No newly acquired match or challenge inputs/video/labels/publicscore.',
        'folds': [{'fit_strong': 'J03WMX', 'evaluate': 'J03WN1'}, {'fit_strong': 'J03WN1', 'evaluate': 'J03WMX'}],
        'data_expansion': 'Original4trackingconditionsplus2newindependentnoiseviews perstronghalf,seeds18618+halfand28618+half. Originalclean/short/long/noisecorpusreused. Newnoiseviewsareaugmentationofexistingmatches,NOTfreshmatches. Replayfivepreviouslylicensedtrainingmatchesonly;nochangeinweaklabels.',
        'sampling': '80percentstrong/20percenthistoricalweakwindows;uniformentrieswithinsourcepool;halfanchorsuniformshort<=75frameknownspells,halfknown-live;shortCEweightsdoubledas053;weakunknownexcludedandweaklabelsremainpartial. Spatialreflections/teamswapsas053.',
        'training': 'Bothbranchesstartsamecorresponding053weights/scaler;2layerbidirectionalGRU64/projection64/dropout.25/508features.12fixediterationsx512windows,batch32,AdamWlr.0001/weightdecay.001,clip1,seed183+fold. ExistingCE+auxboundaryBCE.1inboth. No scalerfit,architecturesearch,earlystop,checkpointselectionor evaluationtuning.',
        'loss': 'Onlyreplay_continuityadds.1weightedmeanadjacent symmetricKL withinSAMEknownliveGTlabel;labelchanges/unknown/D/paddingexcluded. No durationfloor, no inferenceGTinput.',
        'inference': 'Newconditional=.5control040+.5newGRU viaexactresidualh053+.5*(newGRU-oldGRU),normalised/clippedonlyfornumericalstability. Freeze053Dmaskandcached031boundarycosts. ConstrainmissingH/Aframes to053labelsand081survivingnovelreceiptspans to their predictedteam;flanksunconstrainedsofragmentscanreattach. Predictionsonlyanchors,neverGT. Sameanchor-no-trainingcontrol reported separately.',
        'selection': 'Highestpassingtrainedbranchonly:mean>=053+.002and>=sameanchorsno-trainingcontrol;FP<081and<=053;misses<=053;shortmatches>053;zero070recoveredindividualspellloss,zeroANY070shortmatchloss,noadditional053matchedlossvs070. Eachnoisefoldnondeclinevs053;clean/masked/longgapfoldloss<=.01. D/missinglabelsandreceiptanchors exact. Passingwinnerfrozenbeforeaudit.',
        'audit': 'Winneronly,sameexternalmatchesnewerrordraws18518/18718/19122;recompute053/081/083fromobservations,mean/FP/retention/noiseguardspassbeforepromotion. These wouldnotbenewmatches.',
        'limits': 'Twoheavilyreusedfull-HADdevelopmentmatches;fivehistoricalpartial-HAgamesnotindependentfullHADtest. Sevenexistingmatchesintotal,butonlyoneBundesligastrongmatchperfoldplusfiveweakgamesforfit. No new-matchgeneralisationclaim.',
        'source_sha256': {n: digest(root/'controlswitch'/n) for n in ('continuity_refinement.py', 'continuity_refinement_experiment.py')},
        'corpus_sha256': digest(corpus/'manifest.json'),
        'initial_checkpoints': [digest(root/f'local/short-spell-053/fold-{fi}/short_focus.npz') for fi in (0, 1)]}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    extras = []; extra_dir = out/'extra-corpus'; extra_dir.mkdir()
    for mid in ('J03WMX', 'J03WN1'):
        record = record_for(cache, mid)
        for half in (0, 1):
            seq = period_arrays(record, half)
            original = next(e for e in manifest['entries'] if e['source'] == 'strong' and e['match_id'] == mid and e['half'] == half+1 and e['condition'] == 'clean')
            for seed in (18618+half, 28618+half):
                print('Extra training view', mid, half, seed, flush=True)
                x = joint_features(noisy_tracking(seq, seed), record['pitch'])
                path = extra_dir/f'{mid}-h{half}-noise{seed}.npy'; np.save(path, x, allow_pickle=False)
                extras.append({'original': original, 'path': path.name, 'seed': seed, 'rows': len(x), 'sha256': digest(path)})
        del record
    (extra_dir/'manifest.json').write_text(json.dumps({'extra_views': extras, 'new_matches': 0}, indent=2)+'\n')
    allrows, allbase, allproto, allstart = [], [], [], []
    allpred = {k: [] for k in ('anchors_control', *BRANCHES)}; folds = []
    for fi, (train, val) in enumerate((('J03WMX', 'J03WN1'), ('J03WN1', 'J03WMX'))):
        fd = out/f'fold-{fi}'; fd.mkdir()
        checkpoint = root/f'local/short-spell-053/fold-{fi}/short_focus.npz'
        initial = load_checkpoint(checkpoint); mean, scale = initial['mean'], initial['scale']
        strong = [open_entry(corpus, e) for e in manifest['entries'] if e['source'] == 'strong' and e['match_id'] == train]
        weak = [open_entry(corpus, e) for e in manifest['entries'] if e['source'] == 'weak']
        if len(strong) != 8 or set(e['match_id'] for e in weak) != set(WEAK_IDS): raise ValueError('Wrong fitting pool')
        for e in extras:
            if e['original']['match_id'] != train: continue
            entry = open_entry(corpus, e['original']); entry['x'] = np.load(extra_dir/e['path'], mmap_mode='r', allow_pickle=False)
            entry['condition'] = 'extra_noise'; strong.append(entry)
        if len(strong) != 12: raise ValueError('Missing added noise views')
        pos = sum(float((e['t']*e['v']).sum()) for e in strong)
        neg = sum(float(((1-e['t'])*e['v']).sum()) for e in strong); pw = min(10., neg/max(pos, 1))
        models = {}; traces = {}
        for name in BRANCHES:
            model = load_checkpoint(checkpoint)['model']; torch.manual_seed(183+fi); rng = np.random.default_rng(183+fi)
            opt = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.001)
            trace = []; sampled = {}
            for iteration in range(1, 13):
                model.train(); losses, base_losses, consistency_losses, norms = [], [], [], []; started = time.monotonic()
                for step in range(16):
                    windows = []
                    for _ in range(32):
                        pool = weak if rng.random() < .2 else strong
                        e = pool[int(rng.integers(len(pool)))]; anchor = training_anchor(e, rng, True)
                        start = max(0, anchor-int(rng.integers(CORE))); code = int(rng.integers(8))
                        windows.append(augmented_window(e['x'], e['y'], e['focus_w'], e['t'], e['v'], start, mean, scale, code=code, spatial=True))
                        key = e['source']+':'+e['match_id']+':'+e['condition']; sampled[key] = sampled.get(key, 0)+1
                    xx, yy, ww, tt, vv = [torch.from_numpy(np.stack([z[k] for z in windows])) for k in range(5)]
                    opt.zero_grad(set_to_none=True); logits, head = model(xx)
                    ordinary = loss_value(logits, yy, ww, head, tt, vv, pw)
                    consistency = continuity_loss(logits, yy, ww)
                    loss = ordinary+(.1*consistency if name == 'replay_continuity' else 0)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite fine-tuning loss')
                    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
                    losses.append(float(loss.detach())); base_losses.append(float(ordinary.detach())); consistency_losses.append(float(consistency.detach())); norms.append(float(norm))
                row = {'iteration': iteration, 'training_loss': float(np.mean(losses)), 'base_loss': float(np.mean(base_losses)), 'continuity_loss': float(np.mean(consistency_losses)), 'preclip_gradient_norm': float(np.mean(norms)), 'seconds': time.monotonic()-started}
                trace.append(row); print('Fine tune', fi, name, row, flush=True)
                (fd/(name+'-learning.json')).write_text(json.dumps({'trace': trace, 'sampled_windows': sampled}, indent=2)+'\n')
            save_checkpoint(fd/(name+'.npz'), model, mean, scale, 12)
            models[name] = load_checkpoint(fd/(name+'.npz')); traces[name] = trace
            (fd/(name+'-freeze.json')).write_text(json.dumps({'iterations':12,'sha256':digest(fd/(name+'.npz')),'before_opposite_match_evaluation':True},indent=2)+'\n')
            del model, opt
        del strong, weak
        record = record_for(cache, val); rows, base, proto, starting = [], [], [], []
        pred = {k: [] for k in allpred}; reviews = []
        losses = {k: {'short_spells_lost_vs_070':0,'additional_053_spells_lost_vs_070':0,'prototype_recovered':0,'retained_recovered':0,'lost_recovered':0} for k in pred}
        for half in (0, 1):
          seq = period_arrays(record, half)
          for ci, kind in enumerate(CONDITIONS):
            print('Fine tune evaluation', fi, half, kind, flush=True)
            s, mask = condition(seq, kind, half, False)
            y, current, q = cached_case({'source':'short-spell-053','name':'short_focus'}, fi, half*4+ci, root/'local')
            np.testing.assert_array_equal(y, seq['y'])
            with np.load(root/f'local/precision-055-cache/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a: event = a['event'].copy()
            with np.load(root/f'local/gap-fragment-081/fold-{fi}/validation-{half*4+ci}.npz', allow_pickle=False) as a:
                prototype, candidate = a['prototype'].copy(), a['gap_fragment'].copy()
            seen = np.isfinite(s['ball']).all(1)
            if mask is not None: seen &= ~mask
            anchors = np.full(len(y), -1, int); missing = ~seen & (current < 2); anchors[missing] = current[missing]
            changed = candidate != current
            anchors[changed] = candidate[changed]
            h = q[:, :2]/q[:, :2].sum(1, keepdims=True)
            np.testing.assert_array_equal(anchored_decode(h, event, current, np.full(len(y), -1, int)), current)
            x = joint_features(s, record['pitch'], mask); sx = swap_joint(x)
            old = conditional(initial, x, sx)
            values = {'anchors_control': h}
            for name in BRANCHES: values[name] = residual_live(h, old, conditional(models[name], x, sx))
            case = {}
            for name, live in values.items():
                p = anchored_decode(live, event, current, anchors)
                np.testing.assert_array_equal(p == 2, current == 2)
                np.testing.assert_array_equal(p[~seen], current[~seen])
                np.testing.assert_array_equal(p[changed], candidate[changed])
                loss = retention(y, current, prototype, p)
                loss['short_spells_lost_vs_070'] = int((short_flags(y, prototype) & ~short_flags(y, p)).sum())
                loss['additional_053_spells_lost_vs_070'] = int((flags(y, current) & flags(y, prototype) & ~flags(y, p)).sum())
                for k, v in loss.items(): losses[name][k] += v
                pred[name].append(p); case[name] = p
                reviews.append({'half':half+1,'condition':kind,'branch':name,'retention':loss})
            rows.append({'y':y,'condition':kind}); base.append(current); proto.append(prototype); starting.append(candidate)
            np.savez_compressed(fd/f'validation-{half*4+ci}.npz', reference=y, current=current, prototype=prototype, starting_candidate=candidate, **case)
        fold = {'fit':train,'evaluate':val,'training':traces,'reference':evaluate_accuracy(rows,base),'starting_candidate':evaluate_accuracy(rows,starting,base),
                'variants':{k:evaluate_accuracy(rows,p,base) for k,p in pred.items()},'retention':losses,'reviews':reviews}
        folds.append(fold); (fd/'report.json').write_text(json.dumps(fold,indent=2)+'\n')
        allrows.extend(rows);allbase.extend(base);allproto.extend(proto);allstart.extend(starting)
        for k in pred: allpred[k].extend(pred[k])
    reference = evaluate_accuracy(allrows,allbase); starting = evaluate_accuracy(allrows,allstart,allbase)
    scores = {k:evaluate_accuracy(allrows,p,allbase) for k,p in allpred.items()}; trials=[]
    for name,p in allpred.items():
        score=scores[name]; short=sum(short_matches(r['y'],v) for r,v in zip(allrows,p))
        lost={k:sum(f['retention'][name][k] for f in folds) for k in folds[0]['retention'][name]}
        guards={'spell_gain':score['mean_spell_f1']>=reference['mean_spell_f1']+.002,'neutral_control':score['mean_spell_f1']>=scores['anchors_control']['mean_spell_f1'],
                'fp_reduction':score['totals']['false_positive']<starting['totals']['false_positive'],'fp_vs_053':score['totals']['false_positive']<=reference['totals']['false_positive'],
                'misses':score['totals']['missed']<=reference['totals']['missed'],'short_gain':short>sum(short_matches(r['y'],v) for r,v in zip(allrows,allbase)),
                'same_recoveries':lost['lost_recovered']==0,'all_short_retained':lost['short_spells_lost_vs_070']==0,'existing_retained':lost['additional_053_spells_lost_vs_070']==0,
                'noise':all(f['variants'][name]['tracker_noise']['spell_f1_mean']>=f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),
                'clean_gap':all(f['variants'][name][c]['spell_f1_mean']>=f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3])}
        trials.append({'name':name,'scores':score,'short_matched':short,'retention':lost,'guards':guards,'eligible':name in BRANCHES and all(guards.values())})
    eligible=[t for t in trials if t['eligible']]
    report={'protocol':protocol,'folds':folds,'reference':reference,'starting_candidate':starting,'trials':trials,
            'selected':max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None,'promoted_to_default':False,'independent_test':False,'frozen_audit_required':bool(eligible)}
    (out/'freeze.json').write_text(json.dumps({'selected':report['selected'],'before_new_draw_audit':True},indent=2)+'\n')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials: print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],'short',t['short_matched'],t['retention'],t['guards'],flush=True)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--corpus-root',type=Path,required=True)
    a=p.parse_args();run(a.root,a.cache_root,a.corpus_root)
