# Audited named-control continuity and interaction RNN

This update keeps the existing five training matches and PSG–Marseille validation. No challenge data or video is used; no viewed test is used to fit or select either method. The website continues to use the frozen context-001 team model, whose last fresh-test mean spell F1 was 0.6286.

## Named-control recovery

Two methods add only evidence-backed continuity:

1. An anonymous observation may inherit a roster ID only if it lies inside a contiguous provider `track_id` segment, is bracketed by the same known roster ID, has a compatible team tag and is within the selected maximum gap. Known IDs are never overwritten. This is provider-track continuity, not motion-based matching or a name guessed from a shirt/slot.
2. A short missing-ball gap may retain the same named controller only if both observed endpoints have that controller, the predicted team is constant, the player's position is observed throughout, and an interpolated ball corridor stays within the original 2m proximity radius with no observed competitor inside that radius. It is inferred continuity, not an observed touch. Existing direct named associations are preserved.

Parameters were selected by recovered count on the five training matches subject to proxy precision >=99.5%, then independently gated on masked PSG–Marseille observations with at least 500 recoveries. Identity gap candidates: 5, 10 and 20 native 10 Hz frames; ball gaps: 10, 20 and 30 common 25 Hz frames. Selected: **2 seconds for same-track ID continuity** and **0.8 seconds for ball-gap control**. The 1.2-second ball-gap setting failed the training precision requirement and was not used.

Validation proxy precision was **99.9807% for hidden provider IDs** (103,572/103,592 correct) and **99.8993% for hidden proximity associations** (1,984/1,986). These references are withheld automatic provider observations and the original proximity rule, not independent human identity/contact truth. Synthetic gaps can be easier than natural occlusions; provider identity errors may persist.

Actual gallery gains are small: approximately **0.02–0.28 percentage points per SkillCorner team**. The SkillCorner mapped range moves from 33.963–51.562% to 34.043–51.705%; the existing Sportec example remains unchanged at its earlier coverage. Most long occlusions remain unresolved. The update does not establish broad coverage beyond 59%. Direct, provider-ID-recovered and short-ball-gap seconds are recorded separately, and every named + unresolved ledger reconciles to team control. The 2m radius and denominators were not expanded to inflate coverage.

The gallery's inferred coverage is not an RNN input or a possession benchmark label. The legacy loader's recovery option defaults to false, leaving original RNN inference/training inputs unchanged. Displayed names remain roster joins; inferred time is explicitly marked and never presented as confirmed touches.

## Context RNN development

The new experiment adds 10 causal team-pooled ball/player interaction features to the previous 120 offline multiscale features: proximity, relative movement, ball trajectory changes, alignment and closing evidence. One 24-unit tanh recurrent layer is trained from scratch on the same five matches for up to 12 sampled iterations with dropout, L2, clipping and fixed augmentation. It reads PSG–Marseille for validation only, with no test loading.

Selection maximises mean clean/dropout validation spell F1 while requiring missed spells and possible merge counts to be no worse than context-001; false positives and short pulses retain the original European anchor safeguards. Iteration 6 and decoder penalty 36 were selected. **Validation spell F1: 0.5042 → 0.5126**. Pooled validation clean/dropout misses: **337 → 330**; possible merges: **72 → 68**; false positives: **135 → 135**; sub-0.4s pulses: **3 → 0**.

This is reused-validation evidence and may overfit it. There is **no new fresh-test result**, no claim of 0.8 F1, and no promotion to the gallery or production default. The checkpoint is archived as `models/archive/rnn-interaction-context-validation-only.npz`. Its 130-feature input is selected automatically by the frozen runtime version; labels never enter features.

## Reproduce

```sh
python -m controlswitch.control_recovery_audit --raw data/external/skillcorner-europe/raw \
  --model models/rnn-europe-context-001-experimental.npz --out local/new-control-audit
python -m controlswitch.interaction_context --raw data/external/skillcorner-europe/raw \
  --reference models/rnn-europe-context-001-experimental.npz \
  --anchor models/rnn-europe-001-experimental.npz --out local/new-interaction-run --iterations 12
```

Raw files remain local and ignored. The notebook contains disabled reproduction switches. Model and coverage methods remain separate so a better-looking player ledger cannot inflate possession-spell F1.
