# Compact paired-switch verifier011

Study010 had170 input dimensions for170 examples (only28 unsupported spells) and removed genuine possessions. This follow-up reduces the same training examples to20 dimensions and increases L2 regularisation from0.0005 to0.05. The feature/loss choices were fixed after viewing010 and before evaluating011. No new GRU training, match or labels are introduced.

The20 values are10 GRU confidence/duration summaries,3 ball visibility/speed/movement summaries,6 middle-minus-outer proximity/interaction contrasts and1 proximity change relative to the flanks. Absolute pitch position, team centroids/spread and redundant pooled geometry are removed. Role ordering preserves team-swap invariance. Compacting and stronger regularisation are tested together; their individual effects cannot be separated by this study.

Training still uses only match-excluded proposals on4039,3749,3518,2068,2440. Classifier training is45 epochs, seed110. The same four settings are retained: rejection thresholds0.1/0.25 and maximum middle lengths5/10 seconds. Unknown neighbourhoods and ambiguous purity targets remain excluded. Only the middle spell of an unsupported A→B→A excursion may change; there is no cascading recomputation or blanket smoothing.

The same F1, misses, false-positive, merge, short-run and same-spell-retention guards apply. Validation2269 is repeatedly reused; results are exploratory development evidence, not independent test performance. Production/gallery models and the saved eligible candidate remain frozen unless explicitly documented otherwise.

## Reproduction

Use the010 command in `EVENT_PRECISION_010.md`, add `--compact`, and choose a new output directory. The011 notebook cell is disabled by default. The verifier checkpoint and any eligible bundle record the compact feature policy; a170-feature verifier and20-feature verifier cannot be interchanged.

`fit_classifier` now accepts an optional non-negative `l2` parameter. Its default remains0.0005, preserving previous experiment settings. The intercept remains unpenalised. Tests check invalid regularisation rejection, coefficient shrinkage, compact dimension, label-gap exclusion, team-swap invariance and conservative edits.

## Result

Best011 trial (threshold0.1,5-second cap) edited3 spells and scored **0.583167**. It retained all27 recovered reference spells, kept merge overlaps66, reduced misses292→291 and false positives122→115. It failed only the stricter112 false-positive guard.

At the10-second cap, false positives fell to110 but merge overlaps rose to68; F1 was0.582606 with292 misses and all27 recoveries retained. No setting passed every guard. The compact model demonstrates a more conservative tradeoff than010, but remains validation-only.
