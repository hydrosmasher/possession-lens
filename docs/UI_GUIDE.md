# Using the match centre

Open `dashboard.html` or the Mac launcher. The offline dashboard starts with Barcelona–Real Madrid and includes ten external named-player examples.

1. Choose a match in the library.
2. Select **Team control** for possession, the timeline and the interactive control explorer. Choose a half and 1-, 5- or 10-minute windows; click a window or possession segment to inspect it.
3. Select **Player comparisons** to compare provider-mapped names. Player totals cover the full match; unresolved control stays in the denominator. Search and CSV export preserve that denominator.
4. Select **Passes & shots** to open Bochum–Leverkusen, the bundled verified-event example. Inspect event comparisons, expand exact totals, or filter the goals/substitution timeline.
5. Open **Model evidence** for evaluation results and limitations. Validation improvements and historical test results remain separate.

The availability line explains what each match supports. Event charts disappear when no verified feed exists; missing counts are never shown as zero. Touch counts are unavailable. Player control is estimated tracking proximity involvement, not confirmed contacts or causal attacking value. Only provider-roster-mapped names are displayed; missing identities are never guessed.

The control explorer uses concatenated recording-period time. Bins restart at period boundaries; selected-window spells are clipped to the window. These charts do not measure territory or attacking pressure. Half/window filters apply to team analysis, while player comparisons explicitly remain full-match totals.

**Open local analysis** reads JSON/CSV in the browser without uploading it. Provided-match data stays local. **Reset named demo** restores the external example.

The Barcelona-inspired design uses blue, claret and gold with an original PL shield. It does not claim club affiliation. Tabs support arrow keys, Home and End; focus outlines and reduced-motion styles are included.

## Verification

On 13 November, native Chrome review confirmed the desktop layout, player shortcut and full-match notice, verified-event shortcut, correct source totals, and collapsed exact-total disclosure. Automated checks cover gallery switching, chart calculations, player selection, availability and filter visibility. Small-screen CSS is included but this review did not visually verify every screen size.

The library includes development matches and previously viewed tests; it is not ten independent held-out evaluations. The latest decoder result remains validation-only and the 0.8 spell-F1 target is unmet.
