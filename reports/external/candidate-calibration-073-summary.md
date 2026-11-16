# External proposal calibration — end-of-day update

The new filter was fitted and calibrated using external fitting-match data only. For each opposite-match fold, the first fitting half trains a regularised logistic scorer and the second chooses its cutoff. The opposite evaluation match supplies no fitting statistics, weights or thresholds. The broader training pool includes genuine and false short candidates, rather than only missed spells: the first attempt had eight fitting proposals and no positive examples, so it was stopped and preserved as a pilot.

The scorer uses eleven local-model, receipt, motion and duration features. Baseline probabilities and whether the baseline already recognises the candidate are excluded to avoid a shortcut between the broader training pool and actual repairs. The cutoff must retain every genuine calibration candidate and every individual spell recovered by the prototype on calibration data. Parameters and cutoff are frozen before evaluation; no threshold is revised after viewing opposite-match results.

| Evaluation | Mean spell F1 | Matched <=3s spells | Missed spells | False-positive spells |
|---|---:|---:|---:|---:|
| Current submission053 |0.628641|787|1603|1101|
| Recovery prototype070 |0.631917|824|1561|1137|
| Calibrated filter073 |0.631617|818|1569|1131|

The filter removes six false-positive spell instances relative to the prototype, but retains only38 of its46 individual recovered spell instances. Eight are lost. It also remains above the submission model's false-positive count and fails a tracking-noise guard. It is rejected; the submission model and CSV remain unchanged. The same-spell audit prevents a superficially similar total score from hiding lost recoveries.

## Calibration sizes and cutoffs

| Fitting match | Scorer candidates | Genuine candidates | Calibration candidates | Genuine calibration candidates | Frozen cutoff |
|---|---:|---:|---:|---:|---:|
| J03WMX |192|111|132|83|0.2675620914|
| J03WN1 |176|132|142|99|0.1450804323|

Candidate counts include correlated repetitions of the same match under different observation corruptions. These are match-proxy classification scores, not calibrated confidence probabilities or verified touches. The two Bundesliga matches have been extensively reused. Generator models had already seen both halves of their fitting match: the calibration half is separate from scorer fitting only, not from generator fitting. These constraints make optimistic calibration possible. Opposite-match results are development comparisons, not independent competition-performance estimates.

## What is saved and verified

A portable pickle-free filter artifact stores feature ordering, training-only normalisation, weights, cutoff and provenance. Research artifacts are blocked by the loader unless explicitly opened for research, preventing accidental replacement of the approved model. Missing/dead frames are protected by the proposal rules; the portable application also rejects proposals that cross a dead interval or lie outside a single opposite-team spell. Exported scorer probabilities agree exactly with the fitting artifacts, and all16 frozen-score evaluation replays match the saved candidate outputs.

254 automated tests pass, including calibration retention, feature symmetry/shortcut exclusion, artifact loading and dead-ball preservation. The notebook contains the full comparison and retention audit. Prior work and evaluation/submission CSVs are preserved.

## Tomorrow's finalisation priorities

Keep053 as the submission default unless a new candidate passes all release checks. Freeze the code, verify local notebook/inference reproduction and the official CSV format, align the short technical document with the actually deployed method, and review the GitHub/submission bundle. Additional model research must not compromise that finalisation. No score or top-five-percent ranking is guaranteed.
