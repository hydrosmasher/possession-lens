# Complete H/A/D boundary studies020–023

019 is the reference throughout. These studies fit direct H/A turnover evidence using external training only, while keeping019 emissions fixed. They reveal a reproducible recall/false-switch trade-off rather than a substantial safe improvement. No challenge video, tracking, labels, public-window score or manually annotated challenge information is accessed.

## Existing stress draws

| Method | Four-condition spell F1 | Misses | False positives | Geometric merges | Guard outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen019 | 0.462596 | 2458 | 1100 | 544 | Reference |
| 020 balanced_half | 0.476920 | 2387 | 1141 | 536 | Reject |
| 020 balanced_full | 0.485806 | 2292 | 1321 | 503 | Reject |
| 020 recovery_half | 0.476561 | 2381 | 1162 | 533 | Reject |
| 021 supported_boundaries | 0.463150 | 2457 | 1099 | 544 | Pass |
| 021 hierarchical | 0.450388 | 2543 | 1108 | 563 | Reject |
| 021 hierarchical_supported | 0.454192 | 2521 | 1083 | 561 | Reject |
| 022 minimum | 0.463456 | 2458 | 1100 | 544 | Reject |
| 022 geometric | 0.466987 | 2437 | 1107 | 545 | Reject |

020 balanced_full achieves the highest headline F1 here, but adds221 false-positive spells versus019. It is a higher-recall research alternative, not an improvement on every objective. The existing policy against increasing false switches rejects it for automatic replacement. The conservative021 supported-boundary method passes all guards, but its gain is only0.0005535, with2 newly matched and1 lost reference-spell instance. It does not justify extra submission complexity or a claim of significant improvement.022 is an adaptive follow-up to021; both fixed combinations fail at least one guard.

## Changed random observation draws

023 freezes019,021 supported boundaries and020 balanced_full before evaluating a different set of simulated gaps/noise. Short-gap seed1214+i, long-gap1414+i, tracker-noise1818+i; training remains unchanged and clean input is the same. These are the same previously viewed matches, not fresh games or an unbiased holdout. No fit, decoder change or re-selection occurs in023.

| Frozen method | New-draw mean spell F1 | Misses | False positives | Geometric merges |
| --- | ---: | ---: | ---: | ---: |
| 019 | 0.464115 | 2466 | 1124 | 547 |
| 021 supported | 0.464368 | 2465 | 1123 | 547 |
| 020 high recall | 0.483476 | 2317 | 1334 | 507 |

The conservative change again has a tiny gain,0.0002525. The aggressive variant again gains more F1 but adds210 false-positive spell instances. Thus the trade-off persists beyond one synthetic mask draw, without establishing real-match generalisation or significance. Diagnostic totals repeat physical possessions across stress conditions; merges mean geometric overlaps, not independently verified events.

## Mechanism and supervision

020 fits a2-class L2 logistic event head on each training Bundesliga match, both halves and clean/short/long/tracker-noise conditions. Target: a direct known H/A change within+-12frames, with the entire local neighbourhood known live. D transitions are excluded. Every fifth valid frame is eligible; negatives are randomly downsampled to positive count per half/condition. Its balanced outputs are event evidence, not naturally calibrated probabilities. Training uses25epochs, seed120, L2=.001.

Features use the existing team-invariant geometry, interaction and boundary representation plus pooled absolute H/A probability changes at+-12/25/50frames. No labels or timestamps enter features. Only H/A switching cost changes; stop/restart cost3 stays fixed, and there is no minimum-duration floor.020 compares balanced strengths.5/1 and recovery-only strength.5, fixed before fitting.

021 multiplies event evidence by soft opposite-team support on both sides of the boundary, or tests a binary live/dead decoder followed by H/A decoding within live runs. This hierarchy removes ownership uncertainty from play-state likelihood, but degrades the measured external result and is rejected.022 compares minimum and geometric-mean event/support combinations; no new fitting. All follow-up order and guards are recorded in protocols.

Both J03WMX and J03WN1 have been used extensively. Opposite-match folds keep train/validation roles separate within a run, but do not restore pristine test status. Head fitting uses emissions from component models fitted on that training fold, which can create optimistic boundary evidence. No0.8 spellF1 or independent generalisation claim is supported.

## Reproduce

Use the existing first-division Bundesliga caches and frozen017 control/019 dead models from the documented research runs. No optional ML framework is required for020–023; they use base NumPy/pandas.

```sh
python -m controlswitch.boundary_had --cache-root data/external/cache --control-root local/control-017-new --dead-root local/dead-blend-019-new --team-model models/rnn-europe-context-001-experimental.npz --out local/boundary-020-new
python -m controlswitch.structured_had --previous local/boundary-020-new --out local/structured-021-new
python -m controlswitch.boundary_evidence --previous local/boundary-020-new --out local/evidence-022-new
python -m controlswitch.boundary_stability --cache-root data/external/cache --control-root local/control-017-new --dead-root local/dead-blend-019-new --event-root local/boundary-020-new --team-model models/rnn-europe-context-001-experimental.npz --out local/stability-023-new
```

All run directories must be new. Private external emissions, event heads and predictions remain in ignored local research folders; public reports contain aggregate evidence only. Candidate019 weights, CSV and prepared upload bundle remain unchanged by these studies.
