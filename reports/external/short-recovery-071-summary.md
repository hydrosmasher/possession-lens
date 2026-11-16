# Short possessions: evidence versus smoothing

## Diagnosis 068

Among 1,201 missed reference possession instances lasting at most three seconds:

| Diagnostic category | Instances |
|---|---:|
| Weak team evidence |549|
| Fragmented or misaligned evidence |387|
| Recoverable by removing H/A smoothing |142|
| Predicted dead mask blocks most of the spell |123|

These are heuristic, external-reference diagnostics.142 (11.8%) become matched at official IoU 0.5 when H/A smoothing is removed while the deployed D mask stays fixed. Removing smoothing everywhere is harmful: full benchmark spell F 1 falls from 0.628641 to 0.487254 and false-positive spells rise from 1,101 to 3,927. It is not a deployable solution. Weak and fragmented/misaligned evidence together account for 78% of these missed instances.

## Targeted experiments

A small local temporal convolutional expert learns a 42-frame radius (1.68 seconds in each direction) on top of existing 508 observation features. It complements the RNN by recognising local changes without recurrent memory. Each fold uses only its fitting game's windows and scaler. Twenty fixed training iterations, explicit short-spell sampling/weights, team/pitch reflections, dropout and gradient clipping are frozen before evaluation. The baseline boundary scores, costs and D mask stay fixed; all 16 baseline cases replay exactly under the locked-D decoder.

| Variant | Mean spell F 1 | Matched <=3 s spells | Misses | False positives | Promoted |
|---|---:|---:|---:|---:|---|
| Existing 053 |0.628641|787|1603|1101|Current default|
| Local temporal blend 069 |0.629000|862|1518|1230|No|
| Bounded receipt-supported proposals 070 |0.631917|824|1561|1137|No|
| Stricter model-agreement proposals 071 |0.630321|814|1578|1128|No|

The strongest tested proposal variant recovers 37 additional short-spell matches, with 42 fewer total misses and 22 fewer geometric merges. It also adds 36 false-positive spells and fails one per-fold tracking-noise guard, so it is rejected. The unfiltered local expert recovers 75 additional short matches but creates substantially more false positives. Stricter agreement loses useful recoveries without eliminating the false-positive increase.

070 only inserts a 1–75-frame proposed possession wholly inside a longer opposite-team prediction, with visible ball throughout the candidate and its flanks and at least one frozen 064 receipt-compatible close-player observation. Existing missing-ball and dead frames remain unchanged.071 additionally requires average baseline H/A evidence to agree and a receipt-supported frame without competing close-opponent support. Model agreement is not a verified receipt/touch.

## Verification and limits

All experiments use the same two heavily reused external Bundesliga games under four tracking conditions. Counts repeat the same underlying spells under different observation draws.069 is evaluated in both opposite-match directions.070–071 are adaptive follow-ups after observing 069, not independent tests. No candidate passes the predeclared checks, so no frozen new-draw promotion audit is run. No challenge video, labels, scores or inputs were used and no new submission CSV was generated.

The current model053, fallback052 and previous submissions remain preserved. Research models are archived separately in `models/archive/local069-manifest.json`; they are not loaded by submission inference. The notebook shows the diagnosis and all rejected comparisons.245 automated tests cover the new model's finite receptive field, windowed inference, serialization and gradients, exact decoder replay, diagnostic categories, and preservation of missing/dead frames in bounded proposals.

The next needed improvement is proposal selection calibrated on external fitting data, with explicit false-handover examples. Lowering smoothing or changing thresholds on the challenge match is not justified by these results.
