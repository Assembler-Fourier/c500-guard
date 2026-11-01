# C500 Guard audit and acknowledgement upgrade

Audit date: 11 September 2026. Scope: this PC's camera watcher, local AI, Windows input helper, dashboard, startup and recovery paths.

## Changes

- **Five-second acknowledgement:** silences the alarm before writing its log, then suppresses every alert channel for five seconds. Detection continues. Current conditions are checked again at expiry; cleared events are not replayed. Repeated input while quiet does not extend the deadline.
- **Any key or mouse click:** the Windows helper acknowledges an active alarm without consuming ordinary input. Keyboard/mouse processing, ON/OFF control and network requests run separately so a slow request cannot stall input or the OFF hotkey. No typed text or pointer positions are recorded. Mouse movement, scrolling and injected software input are ignored by this global helper.
- **Stale input protection:** each alarm has a unique identifier. A delayed acknowledgement cannot silence a different alarm or an alarm from a restarted watcher.
- **Dashboard:** an acknowledgement countdown, visible state-save warnings, and polling recovery after browser Back navigation. Acknowledgement controls are disabled while quiet or when there is no active alarm.
- **AI recovery:** model loading and inference failures retry with a bounded two-to-sixty-second delay. Results from before a failure or long gap cannot complete a new confirmation.
- **Alignment:** replaced a rigid grid count with the coverage of landmarks that have nearby supporting matches. This avoids rejecting valid geometry solely because matches sit across grid boundaries. Movement limits, scene-appearance checks and required-zone visibility checks remain in place.
- **Process recovery:** Windows checks exact executable/script identity before stopping a process and keeps its process handle open during termination. Unknown or recycled process IDs are not terminated. The supervisor requires three failed API checks before attempting to recover a verified unresponsive watcher.
- **Saved switch state:** a failed ON/OFF state write is now visible. OFF still silences immediately; retrying the same command can save the setting after storage recovers.

## Audit observations

The video and AI queues are bounded. Decoder stall detection, reconnects, zone validation and atomic saving are present. Camera and control endpoints remain bound to loopback, with local Host/Origin and control-header checks.

Two limitations remain: logs have no automatic retention yet (under 700 KB total at audit time), and a native AI inference call that never returns would need separate process isolation for a hard timeout. AI freshness is visible; motion and camera-fault detection continue independently. A real on-site check in daylight and at night remains useful.

## Validation

All **172 automated checks passed** with the real model and camera-image fixtures enabled. Native Windows testing confirmed that the disposable test process had no console, an incorrect process identity could not be stopped, and the correctly identified launcher and child were stopped together. Both installed startup tasks run, and the input-helper log confirms successful keyboard and mouse hook registration.

The final alignment rule also passed **300 raw frames and 59 recorded camera cases**, including traffic, plus regressions for grid-boundary sensitivity, insufficient support and wrong-scene rejection.

The final installed one-minute live check produced **598 new frames**, zero stale samples and zero decoder/watcher restarts. Median local processing was **46.3 ms**, with a 75.8 ms 95th percentile; this excludes upstream camera delay. Its strict zero-pause criterion did **not** pass: alignment rejected five of 226 samples in one brief episode when reliable matches became concentrated in a small part of the view. Alignment and AI recovered automatically. This conservative pause remains a practical limit; the system does not continue scene detection using an unsupported transform. Monitoring remained OFF throughout the check.

The silent test dashboard displayed the five-second countdown and resumed an alarm for persistent simulated activity afterward. Returning from the zone editor restored current polling. Automated checks cover exact timing, all alert channels, stale IDs, repeated input, current-versus-cleared activity, slow log writes, saved-state failures, global input pass-through, hook cleanup, AI retries and guarded process termination.

See the [quick guide](C500-Guard-Quick-Guide.md) for daily operation. The dashboard's current ON/OFF setting is authoritative.
