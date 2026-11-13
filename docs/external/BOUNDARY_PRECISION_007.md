# Boundary precision experiment007

This experiment freezes the high-recall GRU and trains a separate switch verifier on the five existing licensed external training matches. It uses team-invariant geometry, player/ball interactions, probability contrasts and trajectory changes. Labels never enter inference. The verifier changes transition costs, not the GRU's possession emissions.

The backbone is a deterministic replay of006's four focused iterations, seed73. The saved replay must reproduce0.5791444378581307 mean clean/dropout validation spell F1 at penalty28. Replay weights are retained as a research artifact, not an accepted model. No new architecture or backbone tuning is introduced.

Verifier fitting uses only4039,3749,3518,2068,2440: every fifth valid frame plus additional high-boundary-head frames. Unknown-label neighbourhoods are excluded. Suspected transitions are selected by model output, not validation error labels. A regularised logistic classifier trains for20 epochs with seed91. It is fit on in-sample backbone outputs, so verifier training fit alone cannot establish generalisation.

Before running, the study fixes verifier mixes0.25,0.5,0.75,1 and transition penalties12,20,28,40. Only existing validation2269 selects settings. Eligibility requires mean clean/dropout spell F1 above0.5791444378581307, misses<=292 (retaining all24 previously recovered spells in aggregate), false positives<=112, geometric merge overlaps<=66 and zero full-sequence sub0.4-second H/A predictions. An additional audit requires zero loss of the same reference spells matched by the recovery model but missed by the saved model (official IoU0.5 matching within known-label runs). This audit was specified before viewing007 results, in a supplemental protocol. Matching is conditional on provider labels and is not a verified-event judgement.

This is exploratory selection on repeatedly reused partial automatic labels. No fresh test, provided match, video or challenge ground truth is loaded. Production and gallery checkpoints remain unchanged.

## Reproduction

With Python3.11/3.12 and `requirements-gated.txt`, run from the repository root:

```sh
python -m controlswitch.boundary_precision \
  --raw data/external/skillcorner-europe/raw \
  --reference models/archive/rnn-gated-decoder-004-validation-only.npz \
  --out local/boundary-precision-reproduction
```

Use a new output folder. The notebook includes a disabled reproduction cell, aggregate scores and every guard. Protocol, training sample audit, full trial results and checkpoint hashes are saved. An eligible candidate is a bundle of backbone, verifier and decoder settings, reloaded to verify identical results. A bare backbone alone cannot reproduce the verifier.

## Result

None of the16 fixed settings passed the initial aggregate guards. The best unrestricted verifier setting (mix0.25, penalty20) reached **0.571451** mean clean/dropout spell F1, compared with **0.579144** for the frozen high-recall GRU and **0.555356** for the saved model. Its pooled misses were298, false positives124 and merge overlaps67, versus292/122/66 for the high-recall GRU and316/112/66 for the saved model.

A stronger verifier (mix0.5, penalty40) reduced false positives to103 but increased misses to314 and merge overlaps to77; spell F1 was0.5578. This does not satisfy preserving recovered possessions. Zero full-sequence short predictions is insufficient evidence of good segmentation.

No candidate is promoted. This training scheme does not demonstrate improved boundary precision at preserved recall. Additional smoothing or selecting on classification loss would conceal the tradeoff. Future verifier work should assess match-wise out-of-fold training predictions and boundary-event reliability, rather than keep expanding this validation grid. Independent evaluation is still needed.

The same-spell audit found 27 recovered reference spells across clean/dropout conditions (before netting losses elsewhere). The best verifier trial lost 5 of them. All settings failed to retain every recovered spell. The cohort counts describe conditional automatic labels, not verified physical events.
