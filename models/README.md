# Checkpoint index

| Role | File | Interpretation |
| --- | --- | --- |
| Full-match production default | `rnn-boundary-001.npz` | Existing frozen boundary-aware model; not replaced by recent partial-label experiments |
| Named external gallery | `rnn-europe-context-001-experimental.npz` | Frozen context model used to generate the bundled summaries |
| Latest validation-only candidate | `archive/rnn-gated-decoder-004-validation-only.npz` | Gated-003 weights with penalty20/strength1; validation spell F1 0.5554 |
| Gated architecture winner | `archive/rnn-gated-003-validation-only.npz` | Two-layer bidirectional GRU with boundary head; prior decoder penalty24 |
| Interaction reference | `archive/rnn-interaction-context-validation-only.npz` | Previous validation-only reference |

Load through `controlswitch.rnn_runtime.load_frozen` and `infer_frozen`. Gated checkpoints require the optional PyTorch environment; the existing NumPy models do not. Checkpoints are NumPy arrays with versioned metadata and are loaded without pickle. Inference accepts tracking features, never possession labels as model inputs.

Recent gated/decoder results use repeatedly reused, partially labelled external validation data. They are not fresh-test results and are not automatically promoted. Keep the gallery and production roles distinct. All earlier checkpoint filenames are retained to preserve historical reproduction commands. Earlier model notes are in `docs/RESEARCH_HISTORY.md` and the corresponding `docs/external/` study.

Attribution and source hashes are recorded in study reports and data notices. Sportec/IDSSE-derived models cite Bassek et al. (2025), DOI10.1038/s41597-025-04505-y, CC BY4.0. Historical SkillCorner data use its pinned MIT notice. Do not publish competition inputs or local predictions with the checkpoints.
