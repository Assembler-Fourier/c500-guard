# Architecture and alarm policy

`watch.py` owns capture, the current aligned frame, route state, the local HTTP dashboard and lifecycle. `semantic_monitor.py` has one bounded pending-frame queue and one worker. A newer pending frame replaces an older waiting frame; acquisition timestamps survive inference and delivery. Generation checks reject work that belongs to a reset or an old view. Timing and per-view object diagnostics contain no pixels.

`object_detector.py` validates model bytes and input shape. The primary YOLOX-Tiny detector uses focused crops and optional whole-picture context. Route-specific context recovery is opt-in and merges observations from the same source once. A separate optional YOLOX-S entrance pass runs serially in the existing worker, uses the whole picture, filters to entrance geometry/classes, and cannot become a CPU fallback workload. Missing or slow optional inference leaves primary results available.

`object_tracking.py` applies class, confidence, route direction, confirmation area and distinct-source evidence rules. A rolling confirmation window and a two-strong-observation stop-line proof are explicit per-zone settings. Weak diagnostic boxes alone are not confirmation. `visit_memory.py` groups observations into visits so an early route warning and later stop-line confirmation do not automatically become two alarms. It distinguishes acknowledged visits, queued followers, clear evidence and narrowly checked continuation after a tracking gap.

`barrier_monitor.py`, `arm_angle.py` and `barrier_workflow.py` handle physical-arm evidence separately from vehicle detection. A conflicting current angle or background match can yield Unknown. Automatic acknowledgement requires current evidence for the relevant lane; a movement in another lane cannot acknowledge the current visit.

`alarm_audio.py` builds lane-specific tone patterns while preserving soft onset and evidence-dependent escalation. `desk_activity.py` and `hotkey_win.py` supply local input activity. The event lifecycle controls whether a tone may continue; stale evidence must not acquire renewed authority merely because the audio asset exists.

`alarm_recorder.py` writes bounded incident clips asynchronously. When opted in by the watcher, it pairs clean aligned JPEGs with rendered incident frames and records exact source offsets in a ZIP for offline replay. This uses the same incident owner and retention limits; it does not create continuous camera recording or a second reader.

`vehicle_sightings.py` exports bounded clean vehicle crops for the optional `car-memory` process. That process produces an appearance-suggestion report, not proof of identity or occupancy. The report heartbeat and version allow the dashboard to show stale/error state and refresh rows while preserving search, focus and scroll.

The dashboard uses local HTML, CSS, JavaScript and SVG. State freshness guards disable unsafe controls; lane controls remain stable across polls. Camera framing guidance has its own setup view rather than covering the monitored road.
