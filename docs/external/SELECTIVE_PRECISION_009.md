# Emission-protected verification009

After viewing008, this exploratory follow-up fixed four settings before scoring: verifier mixes0.25/0.5, protection thresholds0.25/0.5 and transition penalty28. It adds no training or parameter fitting and introduces no new match. It uses the frozen high-recall GRU and the008 match-excluded verifier.

At each frame, compare each team's possession probability one second before and after it. If the larger absolute change meets the threshold, keep the original GRU boundary head; otherwise blend the verifier. The maximum across both teams keeps the rule invariant under a home/away swap even if dead-state probability changes. This is bounded offline context, not a live forecast. Boundary/gate shape and team-swap behavior are tested.

The same guards apply: spell F1 above high-recall reference, misses<=292, false positives<=112, geometric merges<=66, actual sub0.4-second H/A runs0 and no loss of the same recovered spells. Validation2269 is repeatedly reused; this follow-up is explicitly post008 exploration, not independent evaluation.

## Result

Best unrestricted mix0.25/threshold0.5 reached **0.583734** mean clean/dropout validation spell F1. Pooled misses decreased292→289 and false positives122→108 relative to the high-recall GRU, but merge overlaps increased66→68 and2 of27 recovered reference spells were lost. It fails the retention and merge guards.

Mix0.25/threshold0.25 retained all27 recovered spells and kept merges66, but had293 misses,120 false positives and0.575295 spell F1. Aggregate miss counts and same-spell retention are deliberately reported separately. No setting passed every guard; no model is promoted. Current saved validation candidate remains decoder004 (0.555356 on this same validation); production and gallery checkpoints are also unchanged.

## Reproduce

After producing the high-recall backbone and008 verifier as described in `CROSSFIT_PRECISION_008.md`, run with Python3.11/3.12 and `requirements-gated.txt`:

```sh
python -m controlswitch.selective_precision \
  --raw data/external/skillcorner-europe/raw \
  --reference models/archive/rnn-gated-decoder-004-validation-only.npz \
  --backbone local/recovery-backbone/final-trained.npz \
  --verifier local/crossfit-precision/verifier.npz \
  --out local/selective-precision
```

Use a new output folder. All four trials and guards are saved. No provided-game inputs, video, manual labels or viewed test matches are read. The notebook reproduction is disabled by default. This bounded study does not justify extending the same validation search indefinitely; eventual promotion needs independent licensed external evaluation.
