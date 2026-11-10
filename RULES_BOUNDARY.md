# Competition data boundary

The participant's latest organiser email, received during this session, takes precedence over older repository documentation.

- No output-versus-video checks on the provided match.
- No manually created labels, corrections or other human-produced information about that match.
- No uploading the provided dataset, video or extracts to AI services.
- No training, fitting, calibration or parameter tuning on provided possession ground truth or video.
- Automatic checks using supplied data are permitted. Viewing inputs to understand the task is distinct from comparing predicted outputs with video.
- External datasets may be used when their licences permit it; all training and tuning here use external matches.

Implemented separation:

- `baseline` reads provided automatic tracking and applies the unmodified organiser baseline. It validates format without ground truth.
- `benchmark` accepts only supported external Sportec match files with independently checked team IDs. Features never accept possession labels as arguments; labels are used only for supervision and external evaluation.
- `develop` preserves the original test report and evaluates additions only on the external validation split.
- No command in this project scores predictions against competition ground truth or trains on competition video.
- The judge demo is synthetic. Competition-derived predictions stay in ignored local run folders. External reports cite their CC BY 4.0 source.
