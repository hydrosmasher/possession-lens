# Barcelona context and pass-continuity experiment 076

Date: 16 November 2026. Decision: **keep frozen model 053**. No submission CSV, model weights, production defaults, or challenge evaluation changed.

## Context reviewed

These official reports describe other matches, not the supplied challenge game. They are qualitative context, not frame-level possession labels. No video was watched or downloaded, no player mapping was inferred, and no text was treated as supervision.

| Match | Report observation | Testable hypothesis, not an established tracking result |
|---|---|---|
| Valencia–Barcelona, 17 August 2024 | The report describes central overloads, wide combinations, and more controlled play after taking the lead. | A fixed Barcelona possession percentage or one game-state assumption would be unreliable; retain team-symmetric features. |
| Rayo–Barcelona, 27 August 2024 | The equaliser involved Pedri and Raphinha exchanging passes; the report describes a stronger second half. | Nearby defenders should not interrupt same-team pass continuity without control evidence. |
| Barcelona–Valladolid, 31 August 2024 | The first two goals followed long passes; a later goal followed a counterattack from an opposition corner. | Preserve long pass flights while allowing genuine short transitions; proximity alone is insufficient. |

Sources:
- https://www.fcbarcelona.com/en/news/4086449/valencia-1-2-barca-flick-starts-with-win-as-lewy-scores-a-brace
- https://www.fcbarcelona.com/en/football/first-team/news/4098302
- https://www.fcbarcelona.com/en/football/first-team/news/4102104/fc-barcelona-7-0-real-valladolid-supersonic/featured

These are article-based hypotheses, not a systematic tactical analysis of full matches. Public viewing does not confer download/training rights. No anonymous licensed 2024/25 Barcelona tracking with possession labels was found; StatsBomb's public competition catalogue lacks that LaLiga season, and the local SkillCorner Barcelona matches date from 2019/20.

## Fixed experiment

A conservative geometric veto asks whether an isolated short H/A pulse is actually a straight, steady, fast pass between two different players of the surrounding team. The closest opposing player must not move with the ball. Complete visible ball evidence is required; dead-ball labels and missing-ball frames are immutable. Parameters and the two variants were written to the protocol before scoring, with no post-result threshold sweep.

The veto was applied separately to frozen 053 and to receipt-supported recovery070. It uses only ball coordinates, player coordinates, team membership and existing predictions. No Barcelona identity/score/game clock enters it. It is an offline observation filter, not verified touch detection and not an RNN retraining run.

Evaluation uses only the existing CC BY4.0 IDSSE/Sportec first-division Bundesliga WMX/WN1 data: opposite-match folds, both halves, clean/short-gap/long-gap/noisy tracking. These matches and degradation draws are heavily reused development data, not fresh independent tests. Official possession-spell matching is used through the existing evaluation helpers.

## Results

| Method | Mean spell F1 | Missed spells | False positives | Matched <=3s spells |
|---|---:|---:|---:|---:|
| Current 053 | 0.628641 | 1603 | 1101 | 787 |
| Recovery070, previous rejected candidate | 0.631917 | 1561 | 1137 | 824 |
| 053 + pass veto | 0.628520 | 1607 | 1095 | 784 |
| Recovery070 + pass veto | 0.631806 | 1565 | 1131 | 821 |

The veto fired five times across all repeated cases. It reduced false positives by six but also increased misses by four versus its respective input. All 46 spells newly recovered by070 were retained, but some previously detected spells were lost. The same five pulse vetoes occurred for both inputs, so this did not establish a filter specifically separating new false recovery proposals.

The recovery variant beats053 on the aggregate mean, but scores below070 and regresses on the WMX noisy fold: 0.526092 to0.523860. Therefore neither variant passes the predeclared acceptance guards. No different-error-draw promotion audit was run for rejected variants. No model replacement is justified.

## Reproduce

In the Python3.12 research environment, from the project root:

```sh
python -m controlswitch.pass_transit_experiment --root . --cache-root /path/to/licensed/external/cache
```

This requires private ignored research caches from053/069/070 plus licensed external WMX/WN1 arrays. It refuses to overwrite `local/pass-transit-076`. Public readers can inspect the committed protocol/results and run synthetic tests without private data:

```sh
python -m unittest discover -s tests -p test_pass_transit.py -v
```

Full raw predictions and per-case arrays stay in ignored local storage. The public report contains aggregate scores and frame-numbered research veto diagnostics, not raw tracking, roster names, or challenge predictions.

Next defensible step: learn or test richer receipt evidence from external data, with an explicit guard preserving already-correct baseline spells; do not loosen this physics threshold against the reused validation scores.
