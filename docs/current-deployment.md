# Source publication and QA — 22 November 2026

This is a publication of the implementation already present after the 21 November releases, plus independent QA and research. **No application fixes or new deployment occurred during this audit.** Seven reproduced defects remain open, including a P1 missed-follower alarm. Read the [QA findings](QA-2026-11-22.md) before treating test success as readiness.

## Work included

- Earlier tracking-continuity, guided camera setup, stabilization and pedestrian-attention work; early light warnings and source-freshness protections.
- Bounded evidence/history work, recording deadlines and clean incident-export support.
- The 21 November UI authoring source and bundled dashboard, including Setup, Health, lane status, startup feedback and the dark native title bar.
- The evening reliability changes: retry after slow returning optional inference, display-only arm snapshots during ordinary observation overlap, calibration/learning contention safeguards, steady display crop and completed-start receipt checks.
- The latest barrier UP timers, last-observation display, bell controls and 60-second/10-second reminder defaults in regression fixtures. Reminder interaction defects remain documented; inclusion is not a claim that they are fixed.

## Verification completed for this publication

| Check | Result | Scope |
|---|---|---|
| Frozen installed-source baseline | 2,001 run; 1,997 passed; 4 skipped; 325.593 s | Isolated copied source, config and state; zero snapshot drift |
| Sanitized public main suite | 2,001 run; 1,953 passed; 48 skipped; 297.565 s | Public source and normalized fixtures copied into disposable state |
| Public vehicle-report suite | 26 passed; 6.433 s | Separate appearance-report module, fake records |
| Focused independent detection/barrier regressions | 73 / 221 passed | Overlap the baseline; do not add to its total |
| Adversarial QA | Seven defects reproduced | Actual decision methods with synthetic data and fake services; failures remain unfixed |
| UI authoring parity | Exact PAGE_TEMPLATE match | 189,707 characters; no live rebuild |

The public runner uses fake inference/audio, local HTTP fixtures on ephemeral ports and isolated configuration/state. OpenCV and numeric CPU pools are limited to one thread and the Windows child runs at below-normal priority. No real speaker, camera driver, barrier actuation or live OFF/F4 exercise is implied.

Public skip reasons:

- 42: Labelled campus scene fixtures unavailable.
- 1: Optional private reference fixture is not bundled.
- 2: Private labelled PTZ incident evidence unavailable.
- 1: Set C500_ML_SMOKE_IMAGE to official YOLOX assets/dog.jpg.
- 2: set BARRIERWATCHER_APP_TESTS=1 to run the app lifecycle tests.

Private evidence remains excluded rather than publishing recordings or calibration to eliminate skips. All 1,981 existing test definitions and 5,938 assertion calls were retained; reviewed assertion-expression substitutions concern synthetic device/address examples only. The separate appearance suite retains its 76 assertion calls.

At 00:26 a read-only operational sample showed an armed system, both AI checks ready, both arms freshly DOWN and 0.1-second frame/loop ages. A later sample found alerts disarmed; the live log records a dashboard-origin disarm at 00:30:38. This audit issued no live control command and did not establish the caller's identity. The observed paused state was left in place. Production integrity checks found all 259 monitored source/config/assets unchanged.

## Open limits

The current test suite misses the interaction defects documented in the QA report. Historical native-call stalls, real-world recall through glare/weather, full-night operation and physical barrier-state accuracy remain unproven. Optional inference is still synchronous before primary publication. No new trained model or model replacement was installed.

The publication includes source, public build helpers, documentation and normalized non-deployable fixtures. It excludes active configuration, credentials, models, recordings, calibration images/arrays, raw operational logs, authorization notes and vehicle records. The private copy manifest and synthetic reproduction receipts are retained locally.

Recommended follow-up starts with the demonstrated defects, then a labelled independent-window benchmark, primary-first optional inference experiments and confidence-staged tracking comparisons. See [research and acceptance criteria](UPGRADE-RESEARCH-2026-11-22.md).

---

## Historical verification — 16 November 2026

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
