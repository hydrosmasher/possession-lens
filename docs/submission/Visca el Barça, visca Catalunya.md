# Possession Lens: possession through missing and noisy tracking

**Team:** Visca el Barça, visca Catalunya
**Authors:** Hrishikesh Karmalkar, Basel, Michail and Omar
**Submitted candidate:** 053
**Code:** https://github.com/hydrosmasher/possession-lens
**Updated:** 16 November 2026

## Approach

We estimate team possession for every tactical-video frame using the supplied automatic tracking. The submitted candidate is frozen model 053. It separates live-team control from live/dead play, then decodes a coherent H/A/D sequence. Its 508 identity-free features combine ball-player proximity, relative movement, ball-trajectory changes and collective team shape. These are control proxies, not verified touches.

## Inputs and frame alignment

Metadata XML supplies team membership; tracking TXT supplies player/ball positions and speeds. Coordinates are converted to centred pitch metres. Tracking IDs 1-199319 map to CSV frames 0-199318. Halves are processed separately: frames 11185-90651 and 110808-190197 inclusive. X is written outside those blocks. The frozen model does not use the provided video, homography, Smart Tagging or manual player mappings.

## Model and temporal context

Conditional H/A evidence combines 50% two-layer bidirectional GRU, 37.5% joint-motion tree model and 12.5% retained context RNN. The GRU has a 64-dimensional projection, 64 units per direction, LayerNorm/GELU and locked dropout 0.25. Training and inference use 256-frame cores with 64 context frames per side. This is offline estimation, not a live turnover forecast.

## Possession rules

A close opponent or momentary challenge does not automatically change possession: trajectory, relative movement and temporal evidence also contribute. Pass flight can remain with the same team. There is no hard minimum spell length. A learned boundary verifier adapts H/A switch cost from base 12; stop/restart cost is 3. Clean/gap and noisy-tracking experts estimate live/dead play, with an observation-quality gate. Dead-ball is a model estimate, not a manually entered restart timeline.

## Missing observations

Missing ball is not automatically dead ball. Team-shape features retain evidence when ball coordinates are absent. Gaps of at most 50 frames may be interpolated only between visible endpoints with implied speed at most 50 m/s; original visibility remains a feature. There is no extrapolation or interpolation across halves. Tracking slots are not verified player identities.

## Label sources and amount

The final forward model fits J03WMX from the first-division Bundesliga IDSSE/Sportec release, authorised by DFL under CC BY 4.0. Provider BallStatus/BallPossession labels map mechanically to H/A/D: 145,967 known 25 Hz frames, comprising 85,369 live and 60,598 dead. Reverse development fits J03WN1 and evaluates WMX; forward evaluates WN1. Provider annotations are used as supplied, without team-created manual relabelling. Source: Bassek et al. (2025), doi:10.1038/s41597-025-04505-y.

## Retained historical supervision

The retained context RNN fits five historical SkillCorner first-division games: 4039, 3749, 3518, 2068 and 2440. The historical MIT release is pinned at revision 1ed50b775f93075c2a6d00d0d795f13b7b89709e. Provider possession metadata supplies 187,825 known native 10 Hz H/A frames, resampled to 25 Hz without filling unknown labels; there is no verified dead-ball truth. Four other previously seen historical games are diagnostic only. These partial labels are not equivalent to full H/A/D references.

## Training recipe and lineage

053 continues 052 for 12 fixed iterations of 512 windows; half the anchors target known-live spells up to 3 seconds and those frames receive doubled spell weights. AdamW uses learning rate 0.0001, weight decay 0.001, clipping 1 and seed 153. Training-only scaling, spatial reflections, team swaps and synthetic tracking errors are used. CE excludes dead/unknown/padding/context; auxiliary narrow turnover BCE has weight 0.1. The GRU history totals 52 iterations and 26,624 sampled windows, not independent new labels.

## Other fitted components

The retained joint tree uses 136,592 repeated augmented live examples; clean/gap and noise dead experts use 58,388 and 116,776 exposures. The quality gate uses 116,776 automatically generated corruption-category exposures, not possession truth. The boundary head uses 14,896 balanced exposures. Exact component recipes, source roles and hashes are in reports/external/submission-training-lineage.json. Repeated frames and corruption variants do not increase the number of independent matches.

## Evidence and limitations

Mean official possession-spell F1 over two heavily reused Bundesliga games and four conditions is 0.628641, versus 0.615370 for 052. Different synthetic-error draws after freezing score 0.622752 versus 0.606253. These are development results, not fresh-match accuracy or competition scores. Official spell F1 averages one-to-one matching across IoU thresholds 0.3-0.7.

## Research after freezing

Later receipt, fragmentation and GRU refinements remain research only. The latest fine-tuning replays the same five historical games and adds eight error views of the two Bundesliga matches; no new match was acquired. Best research mean reaches 0.632989 but loses six previously matched short possessions and regresses on a noisy fold. Protecting those short spells increases false positives. Neither replaces 053.

## Known failure modes

Missing/noisy tracking, short-possession boundaries and fragmented longer control remain difficult. 053 has 1,603 misses and 1,101 false-positive spells across repeated conditions; these counts repeat physical possessions. A separate previously seen partial-label PSG-Marseille diagnostic regresses from 0.300976 to 0.278404 for 052 to 053, exposing transfer risk. Those censored labels lack live/dead truth and cannot be equated to full H/A/D evaluation. Competition rank is unknown.

## Compliance

No provided-game possession labels, video or public score are used to train, tune or select the submitted algorithm. No provided data is uploaded to AI services. Inputs are obtained separately by judges; private tracking, videos, feature caches and challenge prediction files are excluded from the public repository.

## Reproduction

Use Python 3.12. CPU inference is pinned to PyTorch 2.9.1 with one thread and deterministic algorithms; no GPU or hosted service is needed. Config files pin all model assets by SHA-256. Obtain licensed organiser files separately.

```sh
git clone https://github.com/hydrosmasher/possession-lens.git
cd possession-lens
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gru-inference-lock.txt
python -m controlswitch.recurrent_pipeline \
  --data-root /path/to/provided-data \
  --out local/submission-053 \
  --config reports/external/short-spell-053-config.json \
  --filename "Visca el Barça, visca Catalunya.csv"
python evaluation/official/evaluation/validate_submission.py \
  "local/submission-053/Visca el Barça, visca Catalunya.csv"
```

Expected CSV: 199,319 data rows; SHA-256 `2f5a445d2c3c10b3126b5bda3a5b9b11050d9a87410f19b7cbc5aaa88b928eeb`.

The frozen file was reproduced byte-for-byte in two isolated source/dependency environments on the same machine. Cross-platform bitwise identity is not claimed. The compact notebook keeps optional inference and research training disabled.

References: https://doi.org/10.1038/s41597-025-04505-y ; https://github.com/SkillCorner/opendata ; https://github.com/metrica-sports/more-than-a-hack-2027
