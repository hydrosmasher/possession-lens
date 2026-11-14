# Evaluation evidence

Start with these aggregate external reports. Files describe the named split and protocol; they are not interchangeable accuracy measurements.

| Evidence | File | Status |
| --- | --- | --- |
| Paired-switch / compact / duration | `external/rnn-event-precision-010-report.json`, `external/rnn-event-precision-011-report.json`, `external/rnn-event-duration-012-report.json` | Reused validation; conservative0.5832 retains all27 recoveries and merges66, but FP115 exceeds112; no promotion |
| Crossfit + selective verifier | `external/rnn-crossfit-precision-008-report.json`, `external/rnn-selective-precision-009-report.json` | Reused validation; best0.5837, failed merge/retention guards; no promotion |
| Boundary precision | `external/rnn-boundary-precision-007-report.json` | Existing validation; verifier failed recovery/precision guards; no promotion |
| Spell-focused fine-tuning | `external/rnn-gated-recovery-005-report.json` and `external/rnn-gated-recovery-006-report.json` | Existing validation only; best0.5791 rejected by guards; no promotion |
| Latest decoder refinement | `external/rnn-gated-decoder-004-report.json` | Reused validation only;0.5554 spell F1; weights unchanged |
| GRU/LSTM architecture comparison | `external/rnn-gated-003-report.json` | Validation only;72 decoder/checkpoint trials |
| Original context fresh-match comparison | `external/rnn-europe-context-001-report.json` | Historical frozen test; separate match and supervision |
| Earlier full-match-model test | `external/rnn-fresh-J03WR9-001-report.json` | Historical frozen test; second division, outside new top-five-only training |

Adjacent `protocol`, `learning`, `validation-trials`, `freeze` and `checkpoint-verification` files preserve decisions and reproducibility. `external/rnn-gated-003-learning.png` visualises fitting curves. Original experiment JSON is retained rather than rewritten after later audits.

The decoder-004 diagnostics distinguish full-prediction short runs from reference-label-gap fragments. Geometric merge counts and player proximity involvement remain descriptive proxies. No supplied-match outputs or raw match files belong here. Older standalone HTML research pages stay in `archive/`; the user-facing experience is the root dashboard.

Full H/A/D validation: `external/deadball-014-*.json`; protocol, label audit, eight trials, scores and frozen settings. See `docs/external/DEADBALL_014.md`.

Reverse dead-ball robustness and CSV reproduction: `external/deadball-015-*.json`; the decoder is frozen from014, both external matches were previously viewed, and candidate CSV replay is byte-identical in the same dependency environment.

Missing-ball development: `external/gap-016-*.json` records the rejected repair/head variants; `external/control-017-*.json` records the hybrid correction, missed/merged spell diagnostics, masked-spell audit, frozen settings and exact saved-model replay. Both match directions reuse previously viewed labels.

Tracker robustness and submission freeze: `external/tracker-018-*.json`, `external/deadblend-019-*.json`, `external/submission-training-lineage.json`, and `external/release-audit-2026-11-14.json`.019 passes preregistered external guards;018 replacements were rejected. Reproduction proof uses a separate newly installed inference environment, same hardware.
