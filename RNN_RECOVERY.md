# Possession Lens — recovering possession spells

## Fresh external result: Fortuna Düsseldorf–FC St. Pauli (J03WQQ)

**Recommended checkpoint: `models/rnn-boundary-001.npz`.** It modestly reduces missed possessions and false positives on a fresh match. Clean merging remains unresolved. The previous symmetric model and original reports are preserved.

Training and boundary fitting use J03WOH only. All settings use J03WPY validation only. J03WMX and J03WN1 were not loaded for development. The spell-balanced and boundary-aware candidates, previous model, baseline window (50), model hashes and dropout seed (300) were declared together before any J03WQQ scoring. Both candidates were tested in that one fixed comparison; no test-time parameter sweep occurred.

| Model | Clean spell F1 | Ball-dropout spell F1 | Clean predicted spells |
|---|---:|---:|---:|
| Official smooth50 baseline | 0.3608 | 0.2254 | 406 |
| Previous symmetric RNN | 0.4860 | 0.4232 | 236 |
| Recommended boundary-aware RNN | 0.5025 | 0.4234 | 239 |
| Experimental spell-balanced RNN | 0.5055 | 0.4122 | 293 |

Reference: **401 possession spells**, 142,345 frames. This match shares its home club with training: fresh-match evidence is not proof of unseen-team generalisation. The boundary model's clean frame macro F1 is 0.9032 versus 0.9013 for the previous model. Under synthetic ball dropout it is 0.8466 versus 0.8419. Spell-score gains are modest; no statistical-significance claim is made.

## Misses, false positives and merging

Misses and false positives below use one-to-one matching at possession IoU 0.5; the headline spell F1 averages several IoU thresholds and therefore is not computed from this one table alone.

| Model | Condition | Missed reference spells | False-positive spells | Merge-overlap proxy | Predicted spells under 0.4 s |
|---|---|---:|---:|---:|---:|
| Previous symmetric RNN | clean | 246 | 81 | 55 | 0 |
| Previous symmetric RNN | masked | 265 | 118 | 49 | 0 |
| Boundary-aware RNN | clean | 237 | 75 | 55 | 0 |
| Boundary-aware RNN | masked | 264 | 112 | 47 | 0 |
| Spell-balanced RNN | clean | 227 | 119 | 52 | 0 |
| Spell-balanced RNN | masked | 256 | 177 | 40 | 0 |

The merge-overlap proxy counts predictions overlapping multiple same-team reference spells. It is descriptive geometry, not a separately validated event metric. Brief-prediction counts are a flicker proxy; a short possession is not automatically an error.

The recommended boundary-aware model reduces clean misses from **246 to 237** and false positives from **81 to 75**, with no predictions shorter than 0.4 seconds. Under dropout, misses fall from **265 to 264**, false positives from **118 to 112**, and merge-overlap count from **49 to 47**. Clean merge-overlap count stays at **55**. This addresses part of the requested problem without claiming that merging is solved.

The more aggressive spell-balanced model recovers 19 extra clean reference spells and reduces the merge proxy, but raises clean false positives from 81 to 119 and lowers dropout spell F1 from 0.4232 to 0.4122. It is retained as **experimental**, rather than made the default.

## Method

The recommended model retains the frozen team-symmetric RNN weights. A binary logistic verifier, trained on external training labels only, estimates proximity to a possession transition from team-invariant geometry, interaction features and recurrent probability changes. Its probability adjusts the Viterbi transition cost: strong boundary evidence lowers the cost, weak evidence raises it. It does not force transitions or use labels at inference.

The classifier uses centred context and decoding uses the whole sequence, so this is **offline** analysis. Targets mark a +/-12-frame neighbourhood of valid external-training transitions. Training includes fixed synthetic ball gaps, sampled at 5 Hz; inference/scoring remain 25 Hz.

Validation selected base transition penalty **24**, boundary strength **0.5**, and minimum dwell **1 frame**. Floors of 5 and 10 frames were tested but did not win; **the recommended model has no added hard duration floor**. It relies on transition evidence and sequence decoding to avoid flicker.

A second development track reweighted recurrent training by inverse square root of training possession length, clipped to [0.25, 4], with dead-frame weight 0.5 and one global training-mean normalisation. Validation selected penalty 12 and a 1-frame floor. It improves recall/merging but introduces the precision/ball-gap tradeoff measured above. No test labels were used for these weights or settings.

## Reproduce locally

The notebook defaults to the boundary-aware checkpoint. Downloads, training, test scoring and competition inference remain disabled by default. Use Python 3.12 with the pinned dependencies.

```sh
python -m possession_lens rnn-predict \
  --data-root data/raw/provided \
  --model models/rnn-boundary-001.npz \
  --out work/rnn/predictions-boundary-001
```

A frozen rerun of the now-viewed external comparison:

```sh
python -m controlswitch.rnn_fresh_test \
  --match J03WQQ \
  --tracking "$HOME/Downloads/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml" \
  --metadata "$HOME/Downloads/DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml" \
  --checkpoint models/rnn-spell-balanced-experimental-001.npz \
  --secondary-checkpoint models/rnn-boundary-001.npz \
  --reference-checkpoint models/rnn-symmetric-001.npz \
  --out work/recovery-reproduction
```

To retrain the verifier, use `python -m controlswitch.rnn_boundaries --help`; to reproduce the duration-weighted experiment, use `python -m controlswitch.rnn_spell_balance --help`. Both commands accept only the recorded J03WOH/J03WPY training and validation caches and cannot take a test-match cache as those roles. Full protocols, hashes, all validation trials, diagnostics and test metrics are in `models/rnn-boundary-001-report.json`.

J03WQQ is now viewed. Further changes need another independent match; do not use any of the three viewed tests for tuning. Competition data stays local, with no training/calibration or output/video checks. Individual outputs remain unverified tracklet proxies.

Data attribution: Bassek et al. (2025), IDSSE/Sportec, doi:10.1038/s41597-025-04505-y, CC BY 4.0; https://figshare.com/articles/dataset/28196177.
