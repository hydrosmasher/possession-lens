# RNN switching experiment

The selected model is **the original 24-unit RNN plus an offline Viterbi decoder with transition penalty 6**. Recurrent weights were retrained reproducibly; augmented training was tested but was not selected because it performed worse under missing-ball conditions. This is a decoder improvement, not a new recurrent architecture.

| Model | Clean spell F1 | Ball-dropout spell F1 | Clean predicted spells |
|---|---:|---:|---:|
| Official baseline, validation-selected ±50-frame smoothing | 0.2690 | 0.2156 | 53 |
| Plain RNN, raw argmax | 0.3408 | 0.2787 | 136 |
| Selected RNN + decoder | **0.5138** | **0.4848** | **56** |

The reference contains 60 spells. Lower spell count alone is not success; spell F1 checks temporal overlap and distinguishes missed possessions from removed flicker. The selected decoder improves clean frame macro F1 from 0.9109 to 0.9139. Clean spell F1 improves by 0.1730, or about 51% relative to the raw RNN.

These are **validation-only** development results from the final 30% of J03WOH's first half, separated from training by a 200-frame purge. They do not establish cross-match generalisation. The previously viewed second half is not evaluated. Both baseline smoothing and RNN decoder/model selection use the same validation conditions. The selected RNN is chosen by mean clean/masked spell F1, not clean performance alone. Viterbi uses future evidence and is suitable for offline analysis, not real-time forecasting.

## Reproduce the existing pilot

```sh
python -m controlswitch.rnn_benchmark \
  --external-root work/external/cache \
  --out work/rnn-switching-new \
  --single-match-pilot --epochs 5
```

## Evaluate more matches

Download all three supported external matches with the existing command:

```sh
python -m possession_lens download-external --out work/external/raw
```

If that connection fails, use the original dataset archive: https://figshare.com/articles/dataset/28196177 (CC BY 4.0). Download the `04_03_positions_raw_observed` XML files for:

- J03WOH: training; a parsed cache is already available on the development machine.
- J03WPY: validation, 354.47 MB: https://figshare.com/ndownloader/files/51643526
- J03WMX: fresh test, 399.14 MB: https://figshare.com/ndownloader/files/51643514

Save complete XML files with their original filenames in `work/external/raw`. Do not rename partial `.crdownload` files to XML. The loader syntax-checks XML, validates frame alignment, and records source hashes. It uses each match's own pitch dimensions when building features.

```sh
python -m controlswitch.rnn_benchmark \
  --external-root work/external/raw work/external/cache \
  --out work/rnn-three-match-001 --epochs 5 --evaluate-test
```

This trains on both J03WOH halves, selects settings using J03WPY, then evaluates both J03WMX halves on clean data and synthetic ball dropout. Settings and checkpoint are written before test metrics are computed. Run the fresh-test command once after development decisions are complete; do not repeatedly tune against its results. Once viewed, J03WMX is no longer a fresh test for later changes. A new output folder is mandatory.

## Use the frozen improved checkpoint

```sh
python -m possession_lens rnn-predict \
  --data-root data/raw/provided \
  --model models/rnn-decoded-001.npz \
  --out work/rnn/predictions-decoded-001
```

The checkpoint includes the selected transition penalty; inference reads it without fitting anything. The original pilot checkpoint remains available. No competition ground truth or video was used in this experiment.
