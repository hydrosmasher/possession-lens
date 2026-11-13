# Gated decoder refinement and short-spell audit

This saves/reproduces the already-viewed November12 in-memory validation study. It is not new preregistered or independent test evidence. Only external validation match2269 is loaded; there is no training, provided-match data use or fresh-test evaluation.

Frozen gated-003 weights are unchanged. The saved decoder-004 checkpoint changes decoder penalty24 to20 and keeps boundary strength1. Mean clean/simulated-ball-dropout validation spell F1 increases **0.552732→0.555356**. Pooled misses decrease **318→316**, false-positive spells **115→112**, and geometric merge overlaps stay **66**. Production/gallery models remain unchanged.

## Correct interpretation of short predictions

The legacy `predicted_spells_under_0_4_seconds` diagnostic measures predictions clipped to known reference-label runs. A long prediction crossing unknown-label gaps can therefore produce a very short scored fragment without any rapid predicted switch. All eight legacy short fragments for gated-003, and all seven for decoder-004, are of this type. Both models have **zero full-sequence sub-0.4-second H/A predictions** on these clean/dropout validation sequences.

The original JSON reports are preserved. New diagnostics add `full_prediction_spells_under_0_4_seconds` and `scored_short_fragments_from_longer_predictions`. Full-sequence counts are descriptive, not a claim about which events are physically correct. Dwell floors1,5 and10 frames did not change spell F1, misses, false positives or merge counts. Blanket smoothing is not justified by those censored fragments.

## Reproduction

With Python3.11/3.12 and optional `requirements-gated.txt` installed, run from the repository root:

```sh
python -m controlswitch.gated_decoder \
  --raw data/external/skillcorner-europe/raw \
  --checkpoint models/archive/rnn-gated-003-validation-only.npz \
  --out local/gated-decoder-reproduction
```

Use a new output folder. The fixed decoder study tests penalties20,24,28 and strengths0.75,1.0. Eligibility requires misses<=318, false positives<=115, geometric merge overlaps<=66 and zero full-sequence short predictions. This threshold/grid had already been examined in memory. The saved reference and selected scores must be reproduced, not described as a new held-out success.

The script verifies every original tensor/normalisation array is bit-identical in the new checkpoint and checks reloaded decoder settings reproduce the selected results. Historical gated-003 reports/checkpoint remain untouched. Validation is partial automatic H/A supervision; the result does not establish challenge-match accuracy or generalisation to fresh matches.
