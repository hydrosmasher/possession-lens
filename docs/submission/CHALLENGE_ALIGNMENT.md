# Possession Lens: submission alignment and three-day plan

Audit: 13 November 2026. Deadline published by organisers: 17 November 2026. Aim to finish on 16 November; exact closing time is unverified.

## Decision

Keep Possession Lens and its code, models, notebook, external benchmarks and dashboard. Pivot development priority to reproducible three-state possession segmentation. No restart or deletion is necessary. Novelty is a research direction, not a claim that no one has built it before.

## Evidence checked

Read participant Downloads CHALLENGE.md, DATA.md, EVALUATION.md, EVALUATION (1).md, SUBMISSION.md and RULES.md; checked current upstream raw documents. Inspected canonical Desktop pipeline, external loader, model features, README, reproduction instructions and existing local prediction summary.

The existing local/run/possession.csv passes the vendored official format validator: 199,319 sequential rows, valid H/A/D/X labels. This proves format only. It does not prove correctness, competitiveness or byte-identical regeneration. No competition ground truth or video was opened or scored in this audit.

Production already has a three-state recurrent inference path. Recent gated/context research mainly evaluates partial automatic H/A labels and does not establish dead-ball performance. Its roughly 0.58 spell F1 must not be compared directly with the organisers' full-match 0.2433 baseline.

Existing ML comparison remains in the separate working directory possession-lens-ml-013: boosted trees 0.5044, Extra Trees 0.5015 versus compact RNN 0.5832 on reused partial validation. It does not justify replacing the RNN. Its pending Desktop sync must preserve the current canonical files.

## Recommended submission direction

Possession Lens: recover team-control spells through occlusions and restarts.

Separate evidence for team control and play state, then combine them with the existing structured decoder:

1. Retain external-trained recurrent H/A evidence and team-permutation averaging.
2. Train or validate a play-state component using permitted external data with actual alive/dead labels. The existing Sportec loader reads BallStatus and BallPossession; audit the licence and top-five-first-division subset before any new fitting. Historical second-division checkpoints remain disclosed comparators, not silently rebranded first-division models.
3. Model observation reliability explicitly: absent ball is uncertain evidence, never sufficient evidence of dead play. Use ball kinematics, team movement, pitch-boundary geometry and observation masks jointly.
4. Evaluate automatic Smart Tagging as optional soft evidence for phases and restarts. Parse tags robustly, ignore N/A team tags, use the documented seconds-to-frame mapping. Phase spans are not possession labels; set-piece spans are not exact dead-ball intervals. Do not hand-adjust intervals or constants to this match.
5. Decode H/A/D per half. Preserve last controlled team through loose balls; allow D to split two spells of the same team. Use evidence for real turnovers instead of simply imposing long minimum spells.
6. Emit X outside published half boundaries; graded-frame abstention requires an explicit uncertainty policy. It is forward-filled by spell scoring, so it cannot repair missed boundaries by itself.

## Three working days

### Day 1: complete the measurable submission pipeline

- Preserve existing git state, checkpoints and local prediction; keep previous outputs unchanged.
- Check upstream scorer/validator compatibility with vendored version before replacing anything.
- Audit external alive/dead labels, licence, league, known-label coverage and historical split use. Reserve any actually untouched match; otherwise explicitly report reused validation.
- Establish a full H/A/D external comparison: official-style baseline, current production, then one factorised candidate. Use the official possession-spell metric, plus per-state F1, misses, false positives and merges. Add realistic missing-ball stress tests.
- Keep provided-game inference free of fitting and calibration. Freeze features and parameter candidates before challenge inference.

### Day 2: one focused improvement and freeze

- Evaluate play-state modelling and optional soft-tag fusion separately where external annotations support a fair comparison. Without suitable external tag data, keep tag fusion experimental/off by default rather than tuning it on the challenge.
- Select using external spell F1 with dead-ball and boundary diagnostics. Avoid expansive sweeps on the repeatedly reused validation match.
- Freeze one complete pipeline, checkpoint hashes, deterministic settings and dependencies. Generate final challenge CSV locally twice into separate folders; require identical SHA-256 hashes and official validation success.
- Refresh notebook to reproduce the actual submitted command. Keep the dashboard as a useful local viewer, not a separate scoring target.

### Day 3: deliver, with buffer

- Run from a fresh source checkout with the licensed assets supplied separately. Check row indexing, half boundaries, H/A/D/X handling, byte-identical output and no unintended network calls.
- Write a short technical document: exact method, assets used/dropped, all training sources and label provenance, control/restart rules, limitations and reproduction command.
- Audit public source and its history for restricted match data, reconstructive derived files and secrets. Prepare public release; change repository visibility only after the review/approval step.
- Use registered team name for submission filenames. Confirm team name and exact closing time before final upload. The form requires CSV, public repository URL and short PDF/Markdown document. Keep 17 November as contingency.

## Scope held for later

Keep named-player external demos, verified-event charts, forecasting, video ingestion and substitution features. Do not spend the remaining submission time expanding them. Provided tracking slots are not real player identities, and manually mapping the supplied match is prohibited.

## Rule precedence

The previously supplied organisers' email explicitly takes precedence over other guidance. Continue external-only training/tuning; do not check predictions against the provided video, send provided assets to AI services, create manual challenge annotations, or use human-produced event/identity data about this game. Public challenge labels must not choose models, thresholds or checkpoints. This plan does not rely on the repository's broader allowance for self-supervised adaptation of the provided match.

## References

- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md
- https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md
