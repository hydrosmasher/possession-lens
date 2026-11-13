# Match-excluded boundary verifier008

Study007 trained its verifier using predictions from a GRU that had already fitted those matches. This study replaces those predictions with five match-excluded folds. Each temporary GRU starts from scratch, and its normalisation uses only the other four matches. No pretrained checkpoint is used to initialise any fold. Reading the excluded match to prepare a private feature cache does not make it a training input; fold fitting explicitly iterates only over the other four IDs.

The five external development matches are4039,3749,3518,2068,2440. Each fold uses a two-layer bidirectional GRU with a boundary head,12 fixed epochs,16 batches of32 windows per epoch, AdamW learning rate0.001, weight decay0.0001, gradient clipping1, and seed108+fold index. Team swapping, simulated ball dropout and masked unknown/context labels follow the existing gated architecture. No validation checkpoint selection is performed within folds.

The verifier fits the same invariant geometry/trajectory/probability features and the same natural/high-head sample scheme as007, using only excluded-match predictions. Its regularised logistic fitting is20 epochs, seed91. The frozen high-recall GRU is then used for validation inference, with the fitted verifier altering only transition costs. Its distribution can differ from the scratch-fold models; match exclusion alone does not eliminate that risk.

Before running, the study fixes mixes0.25/0.5 and penalties20/28/40 (six trials). Existing validation2269 is loaded only after fold training and verifier fitting. Eligibility requires spell F1 above the high-recall GRU, misses<=292, false positives<=112, merge overlaps<=66, no actual sub0.4-second H/A predictions, and zero loss of the same recovered reference spells using official IoU0.5 matching. Clean/dropout conditions remain separate in each report. No provided-game assets or previously viewed test matches are loaded.

These are cross-fitted **training predictions**, not new held-out final evaluation. Validation2269 has been reused repeatedly. Provider labels are partial automatic supervision. Neither production nor gallery models are automatically promoted.

## Reproduction

Use Python3.11/3.12 and `requirements-gated.txt`. Obtain the licensed external inputs separately. From the repository root, first reproduce the frozen high-recall backbone if it is not already available:

```sh
python -m controlswitch.gated_recovery \
  --raw data/external/skillcorner-europe/raw \
  --checkpoint models/archive/rnn-gated-decoder-004-validation-only.npz \
  --out local/recovery-backbone --decoder-followup

python -m controlswitch.crossfit_precision \
  --raw data/external/skillcorner-europe/raw \
  --reference models/archive/rnn-gated-decoder-004-validation-only.npz \
  --backbone local/recovery-backbone/final-trained.npz \
  --out local/crossfit-precision
```

Both output folders must be new. The crossfit run stores a private feature cache (roughly1.5GB), fold checkpoints, training histories, sample audit, verifier and all six trials. Keep these intermediate assets local. The shareable reports contain aggregate external metrics, protocol and hashes. If a candidate passes every guard, its three-file inference bundle is reloaded and checked against selected metrics.

The notebook includes a disabled-by-default reproduction cell and the full trial/retention table. Default notebook execution never launches training.

## Results

All five folds completed. A separate moment calculation from cached training-only features reproduced each fold checkpoint's normalisation; see the aggregate fold-integrity audit. Temporary fold weights are retained locally, not presented as final test models.

The best008 trial (mix0.25, penalty28) reached **0.580626** mean clean/dropout validation spell F1 versus0.579144 for the high-recall GRU. False positives decreased122→106; misses stayed292. Merge overlaps increased66→69, and2 of the27 individually recovered reference spells were lost. No trial passed every guard.

The verifier therefore improved the measured precision/recall tradeoff but did not satisfy preserving the same recovered possessions. The existing saved model remains frozen. This is reused-validation development evidence, not new generalisation accuracy.
