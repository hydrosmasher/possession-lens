# RNN depth and iteration experiment

## Protocol and result
J03WOH training, J03WPY validation; all challenge inputs excluded. Initial experiment 001 allowed 16 sampled iterations; follow-up 002 allowed 32 after the first curves were still improving. Each iteration samples up to 1,024 shuffled windows of 128 frames, minibatches of 16, not a full epoch. All architectures receive the same window budget and seed. Standardisation fits training only. Team swap and ball dropout augmentation use training only. The model receives observations, never labels as recurrent inputs.

One-, two- and three-layer tanh RNNs (24 units per layer) use NumPy BPTT, locked 15% hidden dropout, L2 0.0001, Adam and norm-5 gradient clipping. State resets at training-window boundaries. Full-match inference carries state across chunks; full validation spell metrics separately check this difference. Best weights minimise fixed validation-window cross entropy with patience 3 and minimum improvement 0.0001. Architecture choice uses full clean/dropout validation spell F1 at a fixed decoder penalty 24, with team-symmetric probability averaging.

Experiment 002 clean validation / dropout spell F1: 1 layer 0.5054 / 0.4329 (selected iteration 25, stopped 28); 2 layers 0.4879 / 0.4346 (selected iteration 28, stopped 31); 3 layers 0.5037 / 0.4174 (selected iteration 15, stopped 18). Refer to JSON for exact values. Lower validation cross entropy did not guarantee better possession segmentation. Two layers had the lowest probe CE, but one layer won the declared segmentation objective.

Both train and validation loss fall initially. Validation later fluctuates; early stopping restores an earlier checkpoint. Remaining positive gaps and repeated reuse of one validation match mean overfitting is still possible. This is not proof of a perfectly fitted model. Training may still benefit from more independent matches; layers alone are not a reliable remedy for underfitting.

Initial experiment 001's loss probe compared augmented training with clean validation; do not interpret that gap as equivalent-condition generalisation. Experiment 002 uses clean tracking for both. Preserve both reports rather than hiding the earlier run. Experiment 001's best full segmentation score was slightly higher than experiment 002; longer training did not uniformly improve the target metric.

## Retrospective audit, not fresh evidence
After freezing experiment 002's selected checkpoint, it was compared with the current default and smooth50 baseline on J03WMX, J03WN1 and J03WQQ. These matches were previously viewed. Candidate mean clean/dropout spell F1: 0.3896, 0.3869, 0.5026; default: 0.3906, 0.3278, 0.4629; baseline: 0.2612, 0.2460, 0.2931. No tuning followed this audit. The candidate improves over default on two of three and beats baseline on all three, but fresh-match evidence is still needed. Default rnn-boundary-001.npz stays unchanged. No challenge outputs are rescored against truth or video.

## Reproduce
Install requirements.txt, import licensed local XMLs, and retain the documented external caches. These commands refuse existing output folders.

```sh
python -m controlswitch.rnn_capacity --training-cache data/external/cache/J03WOH-ec4dcf30200a.npz --validation-cache data/external/cache/J03WPY-845202c29ad9.npz --reference models/rnn-boundary-001.npz --out local/capacity-reproduction --max-iterations 32 --probe-mode clean
python -m controlswitch.rnn_capacity_audit --cache-root data/external/cache --candidate local/capacity-reproduction/candidate-model.npz --reference models/rnn-boundary-001.npz --out local/capacity-retrospective-audit
```

To reproduce experiment 001 use --max-iterations 16 --probe-mode augmented and a new output folder. Floating-point/library differences may affect training. Checkpoints and reports record hashes. Models rnn-capacity-002-{1,2,3}layer-experimental.npz correspond to each architecture's retained weights. None replaces the default. Source attribution: Bassek et al. (2025), IDSSE/Sportec, doi:10.1038/s41597-025-04505-y, CC BY 4.0.
