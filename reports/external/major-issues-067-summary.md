# Major-issue audit and missing-ball test

## What the external audit found

There are1,603 missed spell instances across the two repeatedly used Bundesliga games under four observation conditions.1,201 (74.9%) last at most three seconds. These counts repeat real spells under different corruptions; they are not independent matches.

The heuristic cause audit identifies581 instances with an overlong same-team predicted spell,745 with little correct-team coverage,200 with boundary/partial-coverage errors, and77 with fragmentation. These are descriptions, not proof of causation. Under tracker noise,246 of489 missed instances have the ball unseen for most of the spell.

## Missing-ball specialist 066: rejected

The specialist uses team shape/movement and observed anchors before/after a bracketed ball gap. It does not generate a ball path. It revises only predicted-live frames within eligible gaps up to12seconds; observed frames, existing dead-ball labels, open-ended gaps and longer gaps stay unchanged. Training uses only the fitting external match, with team swaps, fixed training corruption draws and spell-balanced weights.

| Comparison | Mean spell F1 | Missed spells | False-positive spells | Accuracy on eligible reference-live gap frames |
|---|---:|---:|---:|---:|
| Existing053 |0.628641|1603|1101|81.34%|
| Matched prior-only gap decoder |0.628457|1612|1069|81.20%|
| Learned gap specialist |0.624543|1638|1075|80.97%|

Neither variant passes acceptance. No new-draw candidate audit is warranted and neither model is promoted. Existing053 and fallback052 remain preserved. The specialist is archived for research.

## Concrete reliability change 067

New recurrent-pipeline exports additionally write `observation-quality.json`: frame intervals and counts for missing ball, ball gaps of at least two seconds, fewer than three finite tracked players on either team, and position steps implying ball speed above60m/s. Gaps and velocity checks reset within each half; outside-half frames are excluded. These fixed observation-availability rules do not use possession labels, training or video. They are not calibrated confidence or verified match events and do not change possession predictions.

The notebook shows the root-cause audit, rejected comparison and observation-availability table. The new report is attached to future exports; previously generated submission/evaluation files are preserved. A synthetic export integration test verifies identical CSV labels with the added report.

## Next priority

Recover short1–3second possessions that are swallowed by an overlong same-team spell. Distinguish insufficient model evidence from excess smoothing before changing the decoder. Preserve supported short interceptions and the current missing-ball fallback. Do not relax false-switch safeguards simply to improve an aggregate score.
