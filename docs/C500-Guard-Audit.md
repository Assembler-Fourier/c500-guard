# C500 Guard — full audit and upgrade (11 September 2026, evening)

Scope: every component on this PC: watcher, alignment, local AI, input helper, supervisor, bridge/relay, dashboard, startup, logs and the live camera at night. The earlier acknowledgement audit remains in [ACKNOWLEDGEMENT-AUDIT.md](ACKNOWLEDGEMENT-AUDIT.md).

## Critical: the guard could not run at night (fixed)

At night the monitor picture is lit differently from the daytime alignment reference. Landmark matches fell from about 370 per frame in daylight to 16–48 after dark, and **every** frame was rejected. While ON, that meant a fault tone within seconds and all zone and AI detection paused. The event log shows this happened four times between 20:03 and 20:23; each time monitoring had to be turned off.

The fix keeps the safety checks and adds two things:

- **Confirmation tier.** When evidence is too weak to measure a new camera position, it may still *confirm* the camera has not moved (within 5 pixels, with a precise fit and the scene still recognisably the same). Zones then stay at the last verified position; a weak estimate never moves them.
- **Lighting references.** Up to four extra references in the same zone coordinates: saved with **Save current lighting** while OFF, or learned automatically when only weak evidence is available. A new reference is kept only if it strongly realigns at least three of the next five frames at the verified pose (0.75 px).

Tested on live frames: a night reference aligned 211 of 226 frames strongly (median 455 matches), within 0.2 px of the daytime pose (1.7 px worst). A camera bump, a different monitor picture or a dark monitor still faults.

## Other problems found and fixed

| Problem | Evidence | Fix |
|---|---|---|
| Every rejected alignment frame reset the background model, blinding detection for about 2 s | Recorded runs had 1.3–1.6 s rejection episodes; the fault trips at 2 s | The last verified position is held for up to 10 s, with no reset on recovery |
| Typing silenced new alarms before anyone heard them | Several alarms acknowledged in under a second | Keys and mouse clicks count only after 1 s of siren; the dashboard button is instant (your choice) |
| A stuck detector loop would leave the dashboard answering but nothing detecting | No heartbeat existed | 20 s watchdog: fault tone, then restart; the supervisor also restarts after 45 s of stall |
| One exception in frame processing would stop the watcher | No per-frame guard | Bad frames are skipped; a persisting error becomes a visible "detector" fault after 3 s |
| Windows can silently drop low-level keyboard/mouse hooks under load | Documented Windows behaviour; no detection | Hooks re-register every 5 minutes (new hook before old is removed) |
| A muted speaker or dead keyboard helper was invisible | Not checked anywhere | System health card, red banner and `ctl.py doctor` checks (speaker read-only via Windows Core Audio) |
| No record of what triggered an alarm | Event text only | Recent alerts with the picture from each alarm, how and when it was silenced; history survives restarts |
| Browser disconnect tracebacks filled the log; logs grew without limit | 171 traceback lines | Routine disconnects are no longer logged; logs rotate at 5 MB |

The parked-vehicle behaviour is unchanged by your choice: a vehicle that stays in a zone alarms again after each five-second quiet period.

## Verification

- **202 automated checks pass** (172 before), none skipped, including dashboard JavaScript in Node, real model inference and this camera's alignment image.
- Installed at 20:44 through the Windows admin prompt. Monitoring stayed OFF through the restart; `ctl.py doctor` passed every check, including speaker 92% unmuted, keyboard helper connected and detector running.
- 20:55: **Save current lighting** stored a night reference. You armed the system at 20:56. Over the next armed minute alignment was locked for 143 of 143 samples, with no faults, frame age at most 0.2 s, detector heartbeat at most 0.2 s, and 39 ms median processing. Total CPU was 28–45%.
- Alarms at 20:56 were silenced by mouse after about 2 s. Their saved pictures show vehicle headlights and tail lights in the zones.
- No new errors in the watcher log since the restart.

## Remaining limits

- **Dawn** has not yet been observed live. The automatic learning and the daytime reference should cover it; check Alignment and Recent alerts in the morning.
- **Headlight sweeps** at night can trigger motion zones even if the vehicle stays outside them.
- A **frozen NVR picture** on the monitor cannot be detected: glare hides the NVR's own clock at this framing. A frozen C500 feed is detected.
- The guard starts at **sign-in**. After a Windows Update restart it does not run until someone signs in.
- Updates need administrator approval because the helpers run elevated.

## Changed files and rollback

`src/watch.py`, `scene_alignment.py`, `platform_support.py`, `ctl.py`, `service.py`, `bridge.py`, `hotkey_win.py`, `dashboard_ui.py`, and new or updated tests. The installer keeps the previous version in `C:\c500-guard\backups\` for rollback.

During the first install attempt (20:39) the non-elevated shell could not stop the watcher. The supervisor was paused for about 80 seconds until it was resumed; the watcher kept running and nothing was changed.
