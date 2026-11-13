# Gated recurrent architecture experiment (validation only)

This is a bounded comparison of four architectures, not a claim to have found the globally best RNN. It uses the same five external SkillCorner training matches (4039, 3749, 3518, 2068, 2440) and the repeatedly used 2269 validation match. No previously viewed test or competition data is loaded. Partial provider H/A possession labels are weak supervision, not independent full-match ground truth.

## Implemented methods

- One-layer GRU; two-layer bidirectional GRU; two-layer bidirectional LSTM; two-layer bidirectional GRU with an auxiliary boundary head.
- A learned continuous 130-to-48 feature projection, layer normalisation at the projection and recurrent outputs, and GELU. This is an embedding of numeric tracking measurements, not a vocabulary or player-identity embedding.
- Locked input/output dropout (0.15), standard between-layer dropout in stacked recurrent modules, and AdamW weight decay (0.0001). There is no dropout applied directly to recurrent hidden-to-hidden connections.
- Xavier input/output weights, gate-wise orthogonal recurrent weights, and an input forget-gate bias of 1 for LSTMs. Other biases start at zero.
- Global gradient-norm clipping at 1, with nonfinite-gradient errors and pre-clipping gradient norms logged each iteration.
- Bounded-window backpropagation: 256 supervised core frames plus 64 context frames on each side at 25 Hz. Hidden states reset for each window. Padding, context-only labels and unknown labels are excluded from classification loss. This is window-truncated BPTT, not stateful streaming across an entire half.
- Matching overlap-save inference uses the same window policy, with a symmetric home/away-swapped average. Bidirectional variants are offline. Recurrent context is bounded by the actual input window, rather than an unrestricted whole-match backward pass. The precomputed offline geometry features also contain centred rolling context of up to 50 frames (2 seconds); even the unidirectional GRU is not a causal live model.
- Train-only input standardisation; deterministic team swaps and simulated missing-ball augmentation. Probe cross-entropy and the train/validation gap are logged separately from the training objective.
- The auxiliary head learns transitions only where the entire supervision neighbourhood is known. It changes decoder transition costs; it cannot insert a possession without emission evidence.

Each architecture trains for 12 sampled iterations of 512 windows, batch size 32, AdamW learning rate 0.001. Iterations 4, 8 and 12 are scored with penalties 24, 36, 48 and 60. The boundary architecture also tests strengths 0, 0.5 and 1. This is at most 72 validation trials, not independent test evidence. No search settings are relaxed after observing scores.

Selection requires mean clean/masked spell F1 above the previous interaction RNN, with pooled misses, false-positive spells and geometric merge overlaps no worse than that reference, plus the historical short-pulse guard. The highest eligible score is archived; nothing is automatically promoted to production or the gallery. A fresh licensed test is still required for a defensible generalisation claim. Reusing this validation match can overfit it even when training is regularised.

## Methods assessed but not applied

Scheduled sampling addresses a model that consumes earlier target tokens. This classifier never consumes previous possession labels, during training or inference, so introducing target feedback would create an unnecessary new dependency and exposure-bias problem. There is no teacher-forcing schedule here.

Word2vec/GloVe represent text tokens. Tracking inputs are continuous geometric observations; the continuous feature projection is the appropriate trainable representation here. Player IDs are deliberately excluded to discourage memorising named rosters.

Language-oriented RL fine-tuning and linguistic-diversity rewards do not correspond to this frame-labelling task. This experiment uses supervised frame/boundary losses and validates the decoded possession spells directly. No RL improvement is claimed.

## Reproduction

Use Python 3.11 or 3.12 and install the optional requirements in an isolated environment:

```sh
python -m pip install -r requirements-gated.txt
python -m controlswitch.gated_experiment \
  --raw data/external/skillcorner-europe/raw \
  --reference models/archive/rnn-interaction-context-validation-only.npz \
  --anchor models/rnn-europe-001-experimental.npz \
  --out local/gated-reproduction
```

Use a new output directory. Training uses CPU and one thread for repeatability; different library/hardware versions may change floating-point results. Portable checkpoints are compressed NumPy arrays with JSON metadata, loaded with `allow_pickle=False`. Existing NumPy models and the offline HTML dashboard do not require PyTorch. Optional tests skip cleanly when PyTorch is not installed.

Official implementation references: [GRU](https://docs.pytorch.org/docs/2.9/generated/torch.nn.GRU.html), [LSTM](https://docs.pytorch.org/docs/2.9/generated/torch.nn.LSTM.html), [gradient clipping](https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm_.html), [initialisation](https://docs.pytorch.org/docs/2.9/nn.init.html).

## Completed result

All 48 sampled iterations and 72 predeclared validation trials completed. The selected model is the two-layer bidirectional GRU with the auxiliary boundary head, iteration 12, decoder penalty 24, boundary strength 1. Mean clean/masked validation spell F1 is **0.552732** versus **0.512649** for the interaction RNN and **0.278277** for the 50-frame nearest-player reference baseline. Clean and masked candidate scores are 0.572131 and 0.533333 respectively. These are repeatedly reused validation results, not fresh test scores.

Across clean/masked validation, missed reference spells decrease 330 to 318, false-positive spells 135 to 115, and geometric merge overlaps 68 to 66. The original diagnostic counted eight reference-clipped short scored intervals, not eight full-sequence rapid switches. The decoder-004 audit found all eight originate from longer predictions cut by unknown reference-label gaps; full-sequence sub-0.4-second prediction count is zero. Historical JSON metrics are preserved for reproducibility. The eligible LSTM alternative reached 0.548170; its original short count also used clipped scoring intervals. No model is promoted to default/gallery.

Weights: `models/archive/rnn-gated-003-validation-only.npz`. The freeze record identifies the exact checkpoint and implementation hashes. All four architecture results and decoder trials are retained, including rejected configurations.
