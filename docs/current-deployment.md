# Release verification — 16 November 2026

The frozen release candidate completed **1,670 tests: 1,666 passed and 4 skipped** before installation. This is the staged application's result, using its isolated regression environment. The public fixture-package validation is recorded separately below; neither result establishes all-night detection reliability.

The release includes bounded latest-frame inference, source-age diagnostics, optional GPU-only entrance verification, route-scoped context recovery, rolling waiting-area confirmation, visit-continuation safeguards, contrasting barrier evidence, lane-specific audio, the redesigned dashboard and the local vehicle appearance report.

Recorded positive and negative replays cover selected incidents and controls. Negative coverage is sampled, not a continuous survey of every lighting or traffic condition. No field false-alarm rate or guaranteed detection time is claimed. The private images, local configurations, calibration and event logs used for incident review are excluded from this public source package.

The first-sighting release was installed on Windows after the 1,670-test suite. Its initial 65-sample live check reported ready detection, frame ages of 0–0.1 seconds and AI ages of 0.1–0.5 seconds; retained calibration templates matched their pre-installation hashes. This is a bounded observation window, not proof of continuous future health.

A subsequent diagnostics-only follow-up separates nested barrier timing from sequential detection timing and records bounded source-correlated whole-frame durations. It was installed after 95 focused tests passed; the earlier 1,670-test count predates it. A subsequent 15-sample live interval contained six alignment-fault samples during large manual camera changes, which the operator confirmed. That interval is retained as imperfect verification evidence and is not presented as continuously healthy.

The operator subsequently changed the active view to **ANPR Entrance**. The application switched to that view's own six-zone profile and 0.5-second AI interval. The optional YOLOX-S entrance pass is disabled for that view, as intended. The new PTZ first-sighting and entrance-verifier settings remain saved in the PTZ profile; they must not be described as currently active ANPR behavior. Camera-label recognition improvements remain a separate pending change at this checkpoint.

Known limitations remain: the recorded 04:53 incident was **not rescued** by the evaluated first-sighting counterfactual, and immediate detection of a truck entering at the lower left remains unresolved. The release does not claim to fix either case. The new timing diagnostics can help investigate future processing stalls; they do not retroactively identify an unrecorded stall or improve recognition by themselves.

The public source package, including the timing follow-up, completed its isolated main suite: **1,680 tests run, 1,632 passed and 48 skipped in 131.399 seconds**. Its separate vehicle-report suite passed **26 of 26 tests in 3.683 seconds**. Camera, model and audio operations were faked; local HTTP fixtures used their own ephemeral ports. The Windows worker ran at below-normal priority with OpenCV/library CPU threads limited to one.

The 48 main-suite skips are explicit: 42 private labelled calibration checks, two private PTZ incident checks, one optional private reference, one model smoke-image check and two opt-in real-browser lifecycle checks. Their assertions remain in the repository; no private images are distributed merely to make these tests run.

An initial public run exited 0 but did not retain a summary because of a Windows hidden-child output-forwarding defect. That run is not counted as passing evidence. The runner was corrected to pass stdout/stderr handles explicitly, and the complete results above come from the subsequent captured run.
