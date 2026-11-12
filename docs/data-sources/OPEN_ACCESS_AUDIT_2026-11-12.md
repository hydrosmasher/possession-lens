# Fresh top-five-league data audit — 12 November 2026

No additional anonymously downloadable dataset was verified in this search with all four requirements: top-five domestic league, continuous player/ball tracking, usable possession supervision, and a confirmed licence permitting the intended use. This is a search finding, not a claim that none exists anywhere.

| Source | Finding | Decision |
|---|---|---|
| SkillCorner original nine-match MIT release | All nine are now training, validation or viewed tests. | Keep existing split history; none can be called fresh again. |
| IDSSE / Sportec, DOI 10.1038/s41597-025-04505-y | Two first-division Bundesliga matches plus five second-division matches. Both first-division matches have already been viewed in this project. | No new qualifying fresh match; exclude second division under the user's scope. |
| peggy44/PremierLeague25-26 and PremierLeague24-25 on Hugging Face | Dataset cards describe SkillCorner 10 Hz tracking and events, CC BY-NC 4.0 research/education use. Both require login, information sharing and acceptance of access conditions. Anonymous raw access returned 401. Possession-label semantics and hackathon-use permission have not been audited. | Free but gated, not anonymously open. Do not bypass access conditions or upload assets. No files imported or used. |
| Soccermatics-Skillcorner | Public analysis code; data require the course's deliberately unpublished Dropbox link or an entitled SkillCorner account. | Code availability does not grant data access. No files used. |
| Driblab open-data | README describes ten 2025 tracking matches, but the current root lists README/images/.gitattributes, no dataset-1 folder or licence. README does not establish possession-label supervision. | Availability, licence and labels unverified; no files used. |
| StatsBomb / Wyscout open events | Event-level records and occasional freeze frames are not continuous player/ball tracking. | Do not turn event gaps into fabricated frame-level truth for this RNN. |
| Current SkillCorner A-League release | Open data but outside the requested leagues. | Excluded. |

Primary references:

- https://huggingface.co/datasets/peggy44/PremierLeague25-26
- https://huggingface.co/datasets/peggy44/PremierLeague24-25
- https://github.com/soccermatics/Soccermatics-Skillcorner
- https://github.com/driblab/open-data
- https://www.nature.com/articles/s41597-025-04505-y
- https://github.com/SkillCorner/opendata

The user explicitly selected **anonymous downloads only**. Free-but-gated sources are therefore excluded from the active search and are not a pending login request.

No new independent test result is claimed. The last frozen fresh-test score remains 0.6286 mean possession-spell F1 for the context RNN on 2841. A finer validation-only decoder search over penalties 18/24/30/36/42/48/54 retained penalty 48: no eligible improvement over 0.5042 validation spell F1. It reads only PSG–Marseille development data and does not reopen the viewed tests for tuning. Full report: reports/external/rnn-decoder-refinement-001-report.json.

Before a future fresh evaluation, obtain permitted raw tracking and labels, audit the development matches, preregister new match-level splits, keep the test labels sealed until checkpoint/decoder freeze, and verify there is no overlap with the nine historical matches or the two viewed Bundesliga matches. Publicly accessible files do not by themselves establish a licence.
