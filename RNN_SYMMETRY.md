# Possession Lens — team-symmetric RNN

## New independent result: J03WN1

On the freshly downloaded VfL Bochum–Bayer Leverkusen match, **team-symmetric inference beats both the baseline and the previous decoded RNN**. The model, decoder and comparison were frozen before evaluation. No competition match data was used for training, tuning or scoring.

| Model | Clean spell F1 | Ball-dropout spell F1 | Clean predicted spells |
|---|---:|---:|---:|
| Official smooth50 baseline | 0.2838 | 0.2083 | 436 |
| Previous decoded RNN | 0.2884 | 0.2426 | 371 |
| Team-symmetric decoded RNN | 0.3707 | 0.2883 | 265 |

Reference: **500 possession spells**, across 141,561 frames. The new model's clean frame macro F1 is 0.8297 versus 0.7601 for the previous RNN and 0.3488 for the baseline. Under dropout, macro F1 is 0.7542, 0.7023 and 0.3547 respectively. Reduced flicker still comes with missed/merged possessions: 265 predicted spells is substantially below the reference.

## What changed

The recurrent weights, architecture, normalisation statistics and transition penalty (24) are unchanged from `rnn-three-match-001.npz`. Inference runs the same RNN twice: once with the original team assignment, once with team assignments swapped. The second prediction's H/A probabilities are swapped back and averaged with the first prediction. An offline Viterbi decoder then segments possession.

This guarantees exact invariance to exchanging team identities, rather than relying on the RNN to learn that symmetry from one training match. It introduces no new fitted parameter and needs no player names, jersey identities or possession labels at inference. It doubles recurrent inference work. A new checkpoint version makes older code reject the model rather than silently ignoring symmetry.

## Development and evidence boundaries

- Training: both J03WOH halves. No new recurrent fitting for the symmetric candidate.
- Validation/model selection: J03WPY, clean plus fixed-seed ball dropout. Its mean spell F1 rose only from 0.4542 to 0.4555; this tiny validation gain was treated as inconclusive.
- Failed experiments retained: transition-weighted training (boosts 3/6 within +/-12 training frames) and separate team-change versus stop/restart penalties. Neither improved the mean validation criterion. They were not promoted or run on the new test.
- Fresh confirmation: J03WN1. Candidate symmetry, penalty 24, reference checkpoint hash, baseline window 50 and dropout seed 300 were fixed before scoring. The same test conditions were used for every comparator.
- Earlier J03WMX results belong to the previous model. The symmetric candidate was NOT run on that already-viewed test, and its score must not be presented as 0.383 on J03WMX.
- J03WN1 has now also been viewed. Do not tune against either test or describe later reruns as fresh evidence. A new independent match is needed for future development.

## Run locally

Use Python 3.12 and pinned requirements. The notebook now defaults to this checkpoint; downloads, training, fresh-test scoring and competition inference remain disabled by default.

```sh
python -m possession_lens rnn-predict \
  --data-root data/raw/provided \
  --model models/rnn-symmetric-001.npz \
  --out work/rnn/predictions-symmetric-001
```

Reproduce the fresh evaluation (a rerun of a viewed test, not new evidence):

```sh
python -m controlswitch.rnn_fresh_test \
  --tracking "$HOME/Downloads/DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml" \
  --metadata "$HOME/Downloads/DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml" \
  --checkpoint models/rnn-symmetric-001.npz \
  --reference-checkpoint models/rnn-three-match-001.npz \
  --out work/fourth-match-reproduction
```

The complete JSON report includes all rejected validation trials, selected-model validation trials, source hashes, reference identity verification and fresh-test metrics. No data files or raw label arrays are published. Data: Bassek et al. (2025), IDSSE/Sportec, doi:10.1038/s41597-025-04505-y, CC BY 4.0; https://figshare.com/articles/dataset/28196177.

Viterbi is offline and uses future evidence. Synthetic ball dropout removes approximately 40% of observations in contiguous 0.8–4-second runs and does not cover every tracking failure. Team possession remains a model estimate. Individual outputs remain unverified tracklet proxies, not named-player percentages. This is one independent result for the new candidate, not a competition score or qualification guarantee.
