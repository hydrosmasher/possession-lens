# Ownership and handover issue log

## Confirmed findings

- The new actor component improved ambiguous provider-team association on its component holdout (174 frames, 94.8% versus 85.6% nearest-player). Provider labels are weak; the match was used elsewhere. This does not establish possession accuracy or verified touches.
- Prototype 062 reduced false spells but increased missed and merged possessions. Rejected.
- Root bug: zero ownership evidence still allowed the new decoder to change team labels. Prototype 063 preserves existing labels exactly on every zero-support frame and resets the ledger around these gaps. All external cases satisfy this invariant.
- Fixing that bug reduced regressions, but 063 still fails overall acceptance. Neither ledger replaces model 053.
- Association can confidently favour the wrong possession team while the ball travels. Receipt evidence needs to be distinguished from proximity before giving it authority over the team model.

## Next isolated issue

Study receipt versus ball-in-flight using external training observations only. Build a behaviour test for an opponent near a passing trajectory who never receives the ball. Keep existing dead frames and unsupported-frame labels immutable. Do not add a global minimum possession duration: short real interceptions must remain possible.

## Verification

222 automated tests passed. Full external comparisons cover the existing two Bundesliga matches under four tracking conditions; these are reused development data, not independent tests. Candidate 053, prior fallback 052, challenge output and evaluation files are preserved. No challenge labels or video were used for these changes.

## Receipt comparison 064

Completed a matched distance-only versus trajectory scorer using external possession-compatible candidate labels. Frozen different-error draws confirm lower opponent-decoy acceptance (83.94% to57.24%) at owning-candidate retention95.03% versus94.23%. A different-provider check lowers decoy acceptance82% to18% but loses4.38points of owner retention. No verified receipt labels or independent project test. The bounded correction changes no possession predictions; model053 remains deployed. See receipt-064-summary.md.226 tests pass. Next isolate safe handover integration and transfer false negatives.

## Missing-ball and reliability issues065–067

Audit:1,201 of1,603 missed external spell instances last<=3seconds. Gap-specific player-motion/endpoint specialist066 underperforms053:mean.624543versus.628641,misses1638versus1603;rejected. Preserve seen-ball predictions andDmask;no new challenge inference. New exports include observation-quality.json with observation-only flags (not calibrated confidence). Synthetic export test proves possession labels unchanged. Notebook and aggregate reports updated. Nextshort1–3second possessions merged into overlong predictions.

## Short-possession diagnosis and recovery068–071

Only142of1201missed<=3sspells recover under rawH/Aargmax with Dmask fixed;globalunsmoothing dropsF1to.487254/FP3927. Weak549 andfragmented/misaligned387dominate. LocalTCN069recovers75additionalshortmatchesbutFP+129. Boundedreceipt-supported070recovers37shortmatches,mean.631917versus.628641,misses1561versus1603,butFP1137versus1101andnoiseguardfails:REJECTED. Strongeragreement071short+27/F1.630321/FP1128alsoREJECTED. AllD/missingframeinvariantsretainedinboundedvariants. No candidatepromoted,newchallengeCSVoraudit. Nextcalibrateproposalprecisiononexternalfittingdata;donotlowerglobalpenalty.

## External calibration073 — end-of-day

Broad candidate-pool logistic scorer trainedfirsthalffittingmatch,thresholdcalibratedsecondhalf;oppositematchnotusedforfit/cutoff. Originalmissed-onlypilot0728negativeexamples/0positives stopped. Broaderpool192/176fittingcandidates,111/132genuine;calib132/142candidates83/99genuine. Elevenfeaturesexclude baseline-correctnessshortcut. FrozenfiltersF1.631617/misses1569/FP1131/short818;070F1.631917/miss1561/FP1137/short824. Retains38of46sameprototype-recoveredspells,loses8=>REJECTED;baseline053unchanged. Newportableartifactloaderblocksresearchbydefault;replays/exportsverified;254testsPASS. No challengeinputs/newCSV/publicpush. Stopfortodayafterfinalsave;tomorrowprioritisesubmissionreproducibility/docs/bundleoverunsafeprototyperelease.
