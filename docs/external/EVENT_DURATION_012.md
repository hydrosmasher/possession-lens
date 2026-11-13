# Compact verifier duration refinement012

After viewing011, this bounded follow-up fixes intermediate middle-spell caps6,7,8 seconds (150,175,200 frames). Rejection threshold stays0.1. The20-feature verifier and high-recall GRU remain frozen; there is no training, refitting or feature change. The three caps were specified before012 evaluation.

The rationale is the011 tradeoff: a5-second cap retained all27 recovered spells and kept merges66 but had115 false positives; a10-second cap had110 false positives but68 merge overlaps. This is explicit post011 exploratory selection on repeatedly reused validation2269, not an independent test. No provided match, video or viewed test is read.

The original guard thresholds remain: mean clean/dropout spell F1 above0.579144, misses<=292, false positives<=112, geometric merges<=66, actual short H/A predictions0 and zero loss of the same27 recovered reference spells. Any accepted research bundle remains validation-only; production/gallery models do not change automatically.

## Reproduce

After producing the high-recall backbone and011 compact verifier (see `COMPACT_EVENT_011.md`):

```sh
python -m controlswitch.event_precision \
  --raw data/external/skillcorner-europe/raw \
  --reference models/archive/rnn-gated-decoder-004-validation-only.npz \
  --backbone local/recovery-backbone/final-trained.npz \
  --verifier local/event-precision-compact/verifier.npz \
  --out local/event-duration-refinement --duration-refinement
```

Use a new output folder. This mode requires the20-feature compact verifier and validates its dimensions and finite arrays. All three trials and retention checks are saved. An eligible bundle is reloaded, including its decoder settings, to verify identical metrics. The notebook reproduction flag is disabled by default.

## Result

No setting passed every guard. The6-second cap scored0.582781 with291 misses,113 false positives,67 merge overlaps and all27 recoveries retained. Seven/eight seconds both scored0.582606 with292 misses,110 false positives,68 merge overlaps and all27 recoveries retained. All actual short-run counts remained0.

The5-second011 setting remains the most conservative improvement over the raw high-recall GRU:0.583167 mean spell F1,291 misses,115 false positives,66 merges and zero lost recoveries. Clean spell F1 improved0.604762→0.608000; simulated-ball-dropout F1 improved0.553527→0.558333. It still fails the strict112 false-positive guard, so no promotion is made. The broader009 peak0.583734 remains a rejected trial because it lost2 recovered spells and added2 merges.

Further tuning of these duration thresholds on the same match would increase validation-selection bias. These studies identify a useful conservative direction but do not establish fresh-match or challenge accuracy.
