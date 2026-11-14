# Multi-source control024, exact inference025 and abstention026

Candidate019 remains the recommended submission model. These studies provide deeper negative-result evidence and an exact runtime improvement; they do not support a new accuracy claim or replacement merely because a trial is newer.

## Expanded control training024

The017 correction uses one Sportec training match per opposite-match fold, while the frozen RNN already uses five SkillCorner training games.024 adds those same five permitted first-division games to the control classifier:4039,3749,3518,2068,2440. It never opens2269 validation or2841/2417/3442 viewed tests. No new download or fresh-match claim. Dataset licence/split details remain in submission-training-lineage.json.

Up to1500 known25Hz H/A frames per half are sampled uniformly, with clean/short/long-gap observations and both team orders. Unknown labels are excluded and never become D. This adds90,000 augmented examples. Forward training has192,444 examples and reverse172,056. The estimator keeps100 iterations,15leaves,lr.08,minleaf40,L2=10,no early stopping. Seed124 differs from017 seed117; provider semantics, sample allocation, fixed capacity and seed mean this is a tested configuration, not a causal proof that adding data is harmful.

Two fixed variants are tested: full expanded classifier, and50/50 old/new log-odds control blend. RNN,019 dead model and final decoder stay fixed.

| Configuration | Mean four-condition spell F1 | Misses | False positives | Geometric merges |
| --- | ---: | ---: | ---: | ---: |
| 019 reference | 0.462596 | 2458 | 1100 | 544 |
| expanded | 0.454515 | 2511 | 1032 | 562 |
| half_blend | 0.459021 | 2490 | 1057 | 558 |

Neither variant passes the guards. The models and all external intermediates remain in private ignored research folders; source, training audit and aggregate results are shared. This finding supports retaining019, not a general claim against multi-provider data or larger training pools.

## Exact inference optimisation025

Two equivalent computations are removed from the deployed019 path. The swapped244-column control features are obtained by an exact column permutation, rather than repeating all geometry/context calculations. Context-RNN emissions are obtained directly, avoiding an unused intermediate decoder before the actual019 decoder. Other recurrent model families retain their existing fallback path. Tests verify exact swapped features, double-swap identity and unchanged decoding.

| Local external probe | Old median seconds | New median seconds | Runtime reduction |
| --- | ---: | ---: | ---: |
| clean,5000frames | 0.2831 | 0.2471 | 12.7% |
| long_gap,5000frames | 0.2984 | 0.2628 | 11.9% |
| tracker_noise,5000frames | 0.3042 | 0.2663 | 12.4% |

Three paired repeats per condition, with warmed calls and alternating order. Probabilities and decoded labels are bitwise identical on all probes. The complete199,319-frame challenge file also has the same SHA-256 as frozen019; only automatic format/checksum checks are performed, never video/ground-truth evaluation. This is a measured local speed improvement, not a claim for all machines or loads. Reproduction benchmark has no model fitting or network access.

Full CSV SHA-256 remains `019e78085b8807b43b6afcc172599509a89fc27eb2ccbd9c5144c9cb1014d961`.

## Missing-ball abstention026

This fixed follow-up evaluates the020 balanced_half boundary decoder with honest X labels where ball observations are missing, the prediction is H/A and the conditional H/A margin is below0.1 or0.2. It never abstains on observed-ball frames or replaces D with X. Thresholds and limits are fixed before scoring.

The official scorer fills X for temporal metrics but charges raw X as a frame miss.026 uses exactly that policy: raw arrays for macroF1, official filled arrays for temporal diagnostics. It also requires every fold/condition to abstain on at most5% of frames and lose at most0.02 macroF1, in addition to existing spell/miss/merge/false-positive and condition guards. No manually selected frames or challenge-specific rules.

| Configuration | Mean spell F1 | Misses | False positives | Geometric merges |
| --- | ---: | ---: | ---: | ---: |
| 019 reference | 0.462596 | 2458 | 1100 | 544 |
| margin 0.1 | 0.476112 | 2392 | 1146 | 535 |
| margin 0.2 | 0.473195 | 2402 | 1147 | 533 |

Neither abstention variant passes all guards: both add false-positive spell instances, and0.2 also exceeds the abstention cap in at least one fold/condition. X is not a free accuracy improvement; it can delay a real turnover by carrying the old team forward. No new submission CSV or checkpoint is promoted from026.

## Reproduction

Existing training inputs only; optional sklearn requirements are needed for024.025 and026 use base NumPy/pandas.

```sh
python -m controlswitch.multisource_control --skill-root data/external/skillcorner-europe/raw --cache-root data/external/cache --reference-root local/control-017-new --dead-root local/dead-blend-019-new --team-model models/rnn-europe-context-001-experimental.npz --out local/multisource-024-new
python -m controlswitch.inference_benchmark --cache data/external/cache/J03WN1-56ef948961a1.npz --out local/inference-025-new.json
python -m controlswitch.uncertainty_had --previous local/boundary-020-new --out local/uncertainty-026-new
```

All output locations must be new. Runtime comparisons are available without historical Git files because the benchmark retains the unoptimised reference computation. All comparative labels are from extensively reused external matches; none is a fresh independent test. Earlier candidate weights, team-named CSV, technical document and upload bundle stay preserved.
