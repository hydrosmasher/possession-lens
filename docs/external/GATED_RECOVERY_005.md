# Spell-focused GRU fine-tuning

This bounded experiment compares two fine-tuning strategies from the frozen decoder-004 checkpoint: ordinary uniform sampling and a spell/boundary-focused variant. The two-layer bidirectional GRU architecture, training-only normalisation and inference policy remain fixed.

The focused variant gives short, fully observed possessions larger loss weights and centres half its eligible sampling attempts around direct, known H/A changes. Censored spell durations are not trusted. Unknown-label gaps never become transition targets. It changes both weighting and sampling together, so their separate contributions cannot be inferred from this comparison.

Both variants use seed73, four iterations of16 batches of16 windows, AdamW learning rate0.0002, weight decay0.0001, gradient clipping1, and the existing masked boundary-head objective. Training has bounded windows with a256-frame core and64 frames of context on each side; context/padding/unknown labels do not receive class loss. Team swapping and simulated ball dropout remain training augmentations. Five external training matches are4039,3749,3518,2068,2440. Only existing validation2269 is scored; no provided-match inputs or previously viewed test matches are loaded.

Iteration2 and4 are evaluated at decoder penalties16,20,24 and boundary strength1. Selection requires better mean clean/dropout possession-spell F1 than decoder004 and no increase in pooled misses, false positives, geometric merge overlaps or full-sequence sub0.4-second predictions. These guards are descriptive and cannot establish physical correctness of every possession.

Fixed clean probes report training and validation cross-entropy separately from the weighted training objective. Lower objective alone does not justify promotion. One repeatedly used, partially labelled validation match cannot establish generalisation; results are development evidence only. Production and gallery checkpoints remain frozen.

## Reproduce

Install `requirements-gated.txt` in Python3.11/3.12, then from the repository root:

```sh
python -m controlswitch.gated_recovery \
  --raw data/external/skillcorner-europe/raw \
  --checkpoint models/archive/rnn-gated-decoder-004-validation-only.npz \
  --out local/gated-recovery-reproduction
```

Use a new output directory. The notebook includes a disabled-by-default reproduction cell. The script saves the protocol before reading data, all learning curves and validation trials, the selected candidate if eligible, and a reload verification. Raw external data must be obtained under its original licence; it is not bundled here.

## Initial result (005)

No candidate passed every guard. Best unrestricted focused trial: iteration4, penalty24, mean spell F1 **0.578926** versus **0.555356** reference. Pooled misses decreased316→293, false positives increased112→122, geometric merges remained66 and full-sequence short predictions remained0. The saved current model was retained. Ordinary fine-tuning peaked at0.5544 and did not improve the reference.

The focused clean-probe training CE rose0.2464→0.2741 from iteration1 to4 while validation CE fell0.6694→0.4712. This suggests better validation classification fit in this bounded run, but does not establish generalisation or eliminate overfitting on a repeatedly reused match.

## Exploratory decoder follow-up (006)

After viewing005, a second protocol repeats only the focused fine-tuning deterministically and evaluates iteration4 at penalties28,32,40,48. This is explicitly post005 exploratory validation selection, not an independent test. Run the same command with `--decoder-followup` and a new output directory. All prior guards remain. No extra iterations, matches or provided-game inputs are introduced.

## Follow-up result (006)

No candidate passed every guard. Penalty28 reached **0.579144** mean validation spell F1, with misses316→292, false positives112→122, geometric merges66→66 and full-sequence short predictions0→0. Penalty40 reached0.576078 and restored false positives to112, but increased merges66→69 (misses296). No checkpoint was promoted or saved as an eligible candidate. All16 scored trials across005/006 are preserved, including rejected results.

This exposes a recall/segmentation tradeoff rather than demonstrating a deployment-ready improvement. More decoder searching on2269 would increase validation-selection bias; the next substantive experiment should improve boundary precision on training data, and eventual promotion needs independent licensed external evaluation. No new anonymous top-five-league test dataset has been established in this update.
