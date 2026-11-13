# Paired-switch / short-spell verification010

Earlier verifiers changed framewise transition costs and sometimes merged genuine possessions. This study judges a short A→B→A excursion as a pair of switches. It can replace only the middle B spell with A, leaving every other frame unchanged. It does not create new candidate boundaries, shift remaining boundaries, or rescore the whole sequence after an edit.

Training proposals come from the five008 match-excluded GRUs. Each fold checkpoint excluded the proposal match from weights and normalisation; protocol and fold-integrity checks are required. The final high-recall GRU remains frozen. No additional recurrent training or validation-label fitting occurs.

Candidate middle spells are at most10 seconds, with at least0.4 seconds of observed predicted control on both flanks. The training label uses only entirely known external-label neighbourhoods extending one second before/after the middle spell: middle-team purity>=0.8 is positive, <=0.2 is negative; ambiguous or unknown neighbourhoods are excluded. This is a provider-label proxy, not human-verified physical control or a guaranteed IoU match.

Features pool role-ordered ball/player proximity, trajectory and movement over the middle spell and both flanks, plus model probability/head confidence and duration. Roles refer to middle/outer team, keeping features invariant under team swaps. No labels enter inference. A regularised logistic classifier trains45 epochs, seed110, on170 examples (142 positive/28 negative). This is a small, imbalanced set relative to the170 feature dimensions; training fit alone cannot justify acceptance.

Before evaluation, the protocol fixes rejection thresholds0.1/0.25 and maximum middle lengths5/10 seconds (four trials). Only low predicted validity permits an edit. Candidates are processed by confidence, with checks preventing a neighbouring edit from cascading into new proposals. These are inferred possession corrections, not verified touches, passes, shots or other event-feed statistics.

Eligibility uses the same safety checks: better mean clean/dropout spell F1 than high-recall reference, misses<=292, false positives<=112, geometric merges<=66, actual sub0.4-second predictions0 and no loss of the same27 recovered reference spells. Existing validation2269 is reused; no new held-out test, provided match or video is loaded. Production and gallery models remain unchanged.

## Reproduce

Use Python3.11/3.12 with `requirements-gated.txt`, licensed external inputs, the high-recall backbone and the five008 fold checkpoints (see `CROSSFIT_PRECISION_008.md`). Then:

```sh
python -m controlswitch.event_precision \
  --raw data/external/skillcorner-europe/raw \
  --reference models/archive/rnn-gated-decoder-004-validation-only.npz \
  --backbone local/recovery-backbone/final-trained.npz \
  --fold-dir local/crossfit-precision \
  --out local/event-precision
```

Use a new output directory. Protocol, source checkpoint hashes, aggregate training counts, all trials and retention checks are saved. An eligible three-file candidate bundle is reloaded and checked against selected metrics. Default notebook execution reads reports only; reproduction requires an explicit flag. Temporary inputs and intermediate predictions remain local.

## Result

No setting passed the guards. Best010 trial (threshold0.1,5-second cap) edited14 spells and scored0.557654, versus0.579144 high-recall reference. Misses increased292→310, false positives decreased122→112 and merges increased66→74. One of27 recovered reference spells was lost. Longer/more permissive edits did worse. The high-dimensional verifier was rejected.
