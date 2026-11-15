# Focused model studies —15 November 2026

Start: frozen 052, overall external development spellF 1 0.615370. The two existing Bundesliga matches are heavily reused; no fresh-test or challenge result is claimed. Target 0.85–0.90 remains aspirational.

053 trains two matched 12 iteration GRU continuations from 052. The focused branch samples half its training anchors uniformly from known-live spells <=3 s and doubles their existing CE weights. Ordinary continuation is the control. No label is an input.

054 independently keeps 052 recurrent weights fixed and refits boundary evidence on 052 emissions. Aligned 116 uses the same existing boundary features; event 120 adds four invariant frozenGRU boundary summaries. This tests emission alignment and the previously unused neural boundary representation.

Both keep halfGRU/half 040 control, existing dead/quality experts and default decoder costs. Each predeclares mean gain>=.003 and per-fold regression guards, freezes one winner before new-observation-draw audit, and keeps 052 if that audit fails. Final release chooses the highest original-condition mean among eligible audited alternatives; different audit scores are not compared as a new selection metric. These checks reuse the same matches, not independent validation. Previous work and user evaluation CSV stay preserved.

Duration diagnostics count one-to-one official matches at IoU 0.5; these recall fractions are not headline meanF 1. Miss-cause categories are heuristic descriptions. No additional match was downloaded.

## Completed session

Starting benchmark: candidate 052 **0.615370**. Final checkpoint: **053, overall external development possession-spell F 1 0.628641**. The target is 0.85–0.90; it remains unmet, and competition rank is unknown.

The selected GRU continues 052 for 12 fixed iterations × 512 windows (6,144 additional sampled windows; 52 iterations and 26,624 sampled windows including its 043/052 history). Half of its training anchors are drawn uniformly from known-live spells of at most 3 seconds, then uniformly within that spell. Those frames receive twice their previous spell-weighted CE weight. AdamW learning rate 0.0001, weight decay 0.001, gradient clipping 1, seed 153, dropout 0.25, spatial reflections and team swaps remain fixed. The unchanged two-layer bidirectional 64-unit GRU still uses 508 features, projection 64, LayerNorm and the narrow auxiliary boundary objective. Forward weights fit WMX only; reverse development fits WN 1. Labels affect training sampling/loss only and never enter inference.

The two Bundesliga matches are heavily reused external development data. Four conditions pool opposite-match predictions: clean 0.727686, short ball gaps 0.636442, long gaps 0.610947, tracking noise 0.539490. These use the organisers' official spell metric, not the challenge ground truth. No challenge video, labels or public score is used for fitting, selection or comparison.

After the fixed winner was selected, different simulated-error draws scored 0.606253 → 0.622752. These are the same matches, not an independent-match test.

| Reused condition |052 spell F 1 |Final spell F 1 |
|---|---:|---:|
| clean | 0.7129 | 0.7277 |
| masked | 0.6219 | 0.6364 |
| long_gap | 0.5927 | 0.6109 |
| tracker_noise | 0.5339 | 0.5395 |

Across repeated conditions, misses 1746 → 1603, false positives 915 → 1101, and geometric merges 422 → 369. These counts repeat physical possessions across conditions. False detections and missed possessions are both reported; a higher F 1 does not imply every error category improved.

053's ordinary-continuation control scored 0.624470; short-focused training scored 0.628641 and passed its frozen different-draw check 0.606253→0.622752. At IoU 0.5 on clean external observations, matched sub-second spells increased 40→57 out of 162, and 1–3 second spells 188→212 out of 330. These recall counts are not the headline mean F 1. Its original false-positive count rose 915→1101, so the precision trade-off remains visible.054 separately tests alignment of the turnover head to 052 emissions, with and without the already-trained GRU boundary signal. Full outcomes and fixed protocols are retained in the numbered reports.

## Separate provider check: a regression to investigate

On the previously reused PSG–Marseille Ligue 1 example, mean partial-label spell F 1 declines **0.300976 → 0.278404** for 052→053. This is a different-provider diagnostic, not a fresh test. Its censored team-possession groups have no verified live/dead labels, so these values cannot be equated to the complete H/A/D benchmark. It was frozen after selection and was not used to refit, tune or reselect the model. The regression exposes transfer risk:053 is a stronger candidate on the current Bundesliga benchmark, not a demonstrated universal improvement.052 remains preserved as a fallback. See reports/external/provider-056-report.json.

The stronger switch-cost checks 055 scored 0.629150(cost 15) and 0.628001(cost 18); neither passed the predeclared improvement requirements. No winner audit was run. The released candidate 053 keeps switch-cost base 12.
