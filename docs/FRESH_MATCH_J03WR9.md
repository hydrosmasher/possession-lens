# Frozen J03WR9 decision

The improved candidate did not beat the current boundary-aware RNN on the declared primary metric. Keep models/rnn-boundary-001.npz as the submission default. Candidate weights remain experimental. No provided-game predictions or video were used to make this comparison.

| Model | Clean spell F1 | Dropout spell F1 | Mean (primary) |
|---|---:|---:|---:|
| Baseline smooth50 | 0.3152 | 0.2289 | 0.2721 |
| Current boundary RNN | 0.5322 | 0.4504 | 0.4913 |
| Longer-trained candidate | 0.5217 | 0.4316 | 0.4767 |

Candidate minus current: -0.0146473. Clean misses/false positives: candidate 213/71, current 211/64. Dropout misses/false positives: candidate 230/123, current 238/93. Both produced zero spells shorter than 0.4 seconds. Clean merge proxy remained 53 for both; dropout merge proxies were candidate 50 vs current 53. These diagnostics show tradeoffs; they are not separately validated contact/possession-change event labels.

## Provenance
Preregistered match J03WR9, candidate rnn-capacity-002-1layer-experimental.npz, reference rnn-boundary-001.npz, baseline window 50, dropout seed 300. All were frozen before downloading/scoring. Mean clean/dropout spell F1 was declared primary. No parameter sweep or post-test tuning occurred. Training: J03WOH; validation: J03WPY. J03WR9 shares Fortuna Düsseldorf with both, so it is not an unseen-club evaluation. One match does not establish significance or universal superiority. This match is now viewed and cannot be reused as a new independent test.

The original Figshare file 51643532 was used because the mirror was slow. Its SHA-256 equals the mirror's published fingerprint: 70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269. The downloaded '(1).xml' was imported automatically without changing the original. Partial downloads were not parsed or scored. Raw XML, parsed cache and private runs stay outside Git. The initial scoring protocol is preserved verbatim in reports/external/rnn-fresh-J03WR9-001-protocol.json. Full metrics, fingerprints and model SHA-256 values are in the report JSON.

Source: Bassek et al. (2025), IDSSE/Sportec, https://doi.org/10.1038/s41597-025-04505-y, CC BY 4.0, DFL-authorised publication. Mirror file: https://huggingface.co/datasets/pysport/idsse-data/blob/main/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml . Original download: https://ndownloader.figshare.com/files/51643532 .

## Reproduce frozen scores
The test command intentionally refuses to start a second evaluation in the same folder. For independent reproduction, create a new folder, copy the published preregistered protocol into it as preregistered-protocol.json, then run:

```sh
python -m controlswitch.rnn_holdout --tracking data/external/raw/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml --metadata data/external/raw/DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml --candidate models/rnn-capacity-002-1layer-experimental.npz --reference models/rnn-boundary-001.npz --out local/your-reproduction-folder
```

Reproduction does not create new fresh evidence. Do not adjust settings based on this match. The dedicated loader verifies the source checksum and the frozen checkpoint hashes before scoring.
