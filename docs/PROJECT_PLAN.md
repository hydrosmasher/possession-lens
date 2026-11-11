# Team decision: recurrent possession model

## Required

1. Train a genuine recurrent neural network on licensed external matches only.
2. Apply frozen weights to the organiser's automatic tracking to label every tactical frame H/A/D/X.
3. Estimate team possession as H/(H+A) and A/(H+A). Report dead and unknown time separately.
4. Reproduce everything locally with Python and pinned dependencies; retain the judge website.

## Individual contributions

The delivered track slots are fragmented and may change identity. They are not individual players or jersey numbers. The initial output is a **tracklet proximity proxy**, not an RNN attribution or named-player possession percentage. Its denominator is total predicted team possession time; ball flight and unavailable/ambiguous observations remain unassigned. A person's share cannot be reconstructed by simply adding or renaming slots. Video availability alone does not supply reliable identities. Manual mappings on this match are prohibited by the organiser clarification.

Automatic identity association would be an additional research component requiring independent validation. Do not delay the required possession submission to promise unsupported individual statistics.

## Optional

- Classify team possession usage into progression/attack versus recycling/backward movement. Define direction of attack and validate the categories on external data first. Backward movement is not automatically tactically negative.
- Analyse individual usage only after reliable identities are available.
- Retain interaction detection and turnover forecasting as experimental modules; they are not required submission outputs.

## Delivery by 17 November 2026

- 10 November: data layout, recurrent implementation and first external training run.
- 11–12 November: fresh external validation, recurrent architecture comparison and freezing settings.
- 13–14 November: reproducible inference, output validation, technical document and repository preparation.
- 15–17 November: packaging, submission checks and final upload.

Existing classical Possession Lens experiments are preserved as comparators, not presented as RNN results.
