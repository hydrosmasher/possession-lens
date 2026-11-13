# Evaluation evidence

Start with these aggregate external reports. Files describe the named split and protocol; they are not interchangeable accuracy measurements.

| Evidence | File | Status |
| --- | --- | --- |
| Boundary precision | `external/rnn-boundary-precision-007-report.json` | Existing validation; verifier failed recovery/precision guards; no promotion |
| Spell-focused fine-tuning | `external/rnn-gated-recovery-005-report.json` and `external/rnn-gated-recovery-006-report.json` | Existing validation only; best0.5791 rejected by guards; no promotion |
| Latest decoder refinement | `external/rnn-gated-decoder-004-report.json` | Reused validation only;0.5554 spell F1; weights unchanged |
| GRU/LSTM architecture comparison | `external/rnn-gated-003-report.json` | Validation only;72 decoder/checkpoint trials |
| Original context fresh-match comparison | `external/rnn-europe-context-001-report.json` | Historical frozen test; separate match and supervision |
| Earlier full-match-model test | `external/rnn-fresh-J03WR9-001-report.json` | Historical frozen test; second division, outside new top-five-only training |

Adjacent `protocol`, `learning`, `validation-trials`, `freeze` and `checkpoint-verification` files preserve decisions and reproducibility. `external/rnn-gated-003-learning.png` visualises fitting curves. Original experiment JSON is retained rather than rewritten after later audits.

The decoder-004 diagnostics distinguish full-prediction short runs from reference-label-gap fragments. Geometric merge counts and player proximity involvement remain descriptive proxies. No supplied-match outputs or raw match files belong here. Older standalone HTML research pages stay in `archive/`; the user-facing experience is the root dashboard.
