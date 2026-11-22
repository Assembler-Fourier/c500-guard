# Deployed QA repairs — 22 November 2026

The seven reproduced QA defects are now repaired. Read the [release evidence and limits](FIXES-2026-11-22.md). The following-car fix preserves distinct observed visits while retaining the ten-second acknowledgement policy.

| Check | Result | Scope |
|---|---|---|
| Frozen complete application suite | 2,062 run; 2,058 passed; 4 skipped; 485.945s | Isolated state; no source drift |
| Sanitized public package | 2,062 run; 2,014 passed; 48 skipped; 465.107s | No private scene fixtures distributed |
| Separate vehicle-report suite | 26 passed; 6.407s | Synthetic appearance records; unchanged module |
| Accelerated traffic | 60,000 episodes; 52,500 vehicles; 22,500 people; 7,500 negative episodes; no assertion failures | 1,665,000 observation packets; 2,000 virtual hours; 350.57s wall time |
| Two-arm stress regression | 10,000 complete cycles passed | Included in the full suite, not an additional test-method count |
| Recorded-observation comparison | Four clips, 432 observations; exact IDs and first-event times preserved | Cached real-model boxes, not new inference |
| Post-install health | 12 consecutive healthy samples | Actual restarted watcher, camera, AI, input and served UI |

The initial complete private/public runs each had eight failures in JavaScript test harnesses. Six extracted a newly added event listener without its DOM; two omitted a new transport global. Function extraction boundaries and test globals were corrected; all 14 existing methods and 92 JavaScript assertion fragments in the two affected files were retained. No application code changed between the large traffic run and the final full suites. The initial failed logs remain local. The final complete reruns above passed.

Public skips retain the original optional/private-fixture assertions: 42 labelled scene fixtures, two private PTZ incidents, one private reference, one model smoke-image opt-in, and two real app-lifecycle opt-ins. Private-suite skips are its four optional smoke/reference/app checks as recorded by that runner. No live OFF/F4 exercise is claimed.

Installed 12 allowlisted source/test files with verified backups and a supervised refresh. Installed SHA-256 hashes match the tested stage. Alerts remained **armed**, matching the pre-refresh state. The final operational sample had fresh camera/input/AI data and no fault; the one-second input guard and ten-second quiet period were unchanged. The generated reminder WAV is 10 seconds, matching the configured period. Active configuration and nine of ten monitored runtime archives retained their hashes, including every barrier template and the primary alignment reference. The existing startup learner refreshed one lighting-reference cache at 01:16:18; its logged update, unchanged learning methods and six valid reference arrays were reviewed and its exact new hash was accepted. The initial blanket hash-check failure is retained. Served dashboard/settings pages contain the new retry flow.

The two-second entrance test was accepted, playback was observed starting and stopping without an error, and speaker volume returned to its prior value. Human audibility was not confirmed.

The editable UI source is synchronized to the tested bundle before publication; future builds must use `tools/build_ui.py` from a development checkout. Physical vehicle arrivals and speaker audibility are not established by simulated alarms or an accepted test-tone request.

---

## Previous audit checkpoint

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
