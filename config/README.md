# Local configuration

Copy `zones.example.json` to `zones.json` and draw zones for your own camera. The example uses arbitrary rectangles, disables alignment and presence capture, and enables neither the optional entrance verifier nor camera-specific motion cues. It is not a validated alarm installation.

`source.json` is local-only and has the form `{"source":"cam:0"}` for a selected capture source. On Windows, inspect FFmpeg DirectShow devices with `ffmpeg -list_devices true -f dshow -i dummy`; the device-list command normally exits with an error after listing. Network sources and credentials belong only in this local file, never in Git. Match the physical view, route direction and classes before enabling alerts.

View profiles, empty-road references, barrier up/down calibration and alignment templates must be collected locally. Do not copy the test fixtures into a real installation. The fixture test runner creates its own temporary configuration tree.
