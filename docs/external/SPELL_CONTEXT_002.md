# Short-spell weighted context RNN (validation only)

This experiment targets the interaction RNN's low recall of short possessions. It compares from-scratch one-layer (24 units) and two-layer (24 + 24 units) networks, with 130 geometry/interaction features. Training uses external matches 4039, 3749, 3518, 2068 and 2440. Only 2269 is used for validation. No previously viewed test or competition match is loaded.

The existing audited short-spell loss weighting boosts uncensored short labelled runs; runs truncated by unknown labels or period boundaries receive the censored-run treatment. Unknown targets receive zero weight. Team swaps, deterministic missing-ball augmentation, dropout 0.15 and L2 0.0001 are unchanged. Each variant runs at most 12 sampled iterations of 1,024 windows. Four decoder penalties (24, 36, 48, 60) are evaluated per iteration. This is a fixed 96-trial validation search, not 96 independent tests.

The comparison reference is `models/archive/rnn-interaction-context-validation-only.npz`, not the older context RNN. A candidate must have no more pooled missed reference spells or geometric merge overlaps than this reference, and satisfy the historical European RNN's false-positive/short-pulse guard. Among eligible candidates, selection maximises mean clean/masked spell F1. A selected checkpoint is archived even if its score does not beat the reference; the report explicitly records whether it improved. No checkpoint is automatically promoted.

Run from the repository root:

```sh
python -m controlswitch.spell_context \
  --raw data/external/skillcorner-europe/raw \
  --reference models/archive/rnn-interaction-context-validation-only.npz \
  --anchor models/rnn-europe-001-experimental.npz \
  --out local/spell-context-002 --iterations 12
```

Use a new output folder for each run. Epoch curves report unweighted train/validation probe cross-entropy separately from the weighted training objective; their difference helps diagnose fitting, but does not establish generalisation. The validation match has been reused in earlier experiments. Reported scores are conditional on partial provider possession labels, not full-match independent ground truth. Unknown intervals are not scored as dead ball. A fresh licensed external test is still needed before claiming generalisation or replacing the gallery model.

## Completed result

All 24 sampled iterations and 96 validation trials completed. No candidate passed all predeclared guards, so no new checkpoint is promoted or archived as eligible. See the report’s `best_unrestricted_by_architecture` for rejected peak scores and diagnostics, and the learning log for fitting curves. The previous interaction RNN remains the research reference, and production/gallery weights are unchanged.
