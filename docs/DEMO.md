# Interactive browser demo

[Open the public lab](https://assembler-fourier.github.io/c500-guard/).

The public site is a standalone, interactive explanation of BarrierWatcher. Its original Canvas campus, traffic, tracks, alerts and gate movements are **synthetic**. It does not access a camera, run a detector, connect to an installation, store visitors' activity, or control physical barriers. The real application remains separate and does not actuate barriers either.

## Explore

- **Guided tour:** repeating scripted traffic with explicitly labelled demo-operator actions.
- **Entrance / Exit:** a vehicle contacts an early approach zone. Acknowledge the alert, then use **Open & serve** to let that simulated visit through.
- **Following cars:** acknowledge the first visit. A separate follower keeps its own identity and alerts after the ten-second quiet period; serving one vehicle does not resolve another.
- **Pedestrian:** an alert stays latched even after the person leaves the scene, until acknowledged.
- **Occlusion:** a scripted hidden interval preserves the same track and visit.
- **Light change:** a subtle exposure effect without a scripted vehicle; this illustrates the intended distinction and is not a real classifier benchmark.

Use day/night, zones, pause, restart and 0.5× / 1× / 2× speed controls. The `A` shortcut acknowledges while focus is outside an editable field. All controls also work through normal keyboard navigation. Sound is OFF by default and must be explicitly enabled. Demo tones use limited-gain sine pulses, not the installed alarm or its escalation curve. Muting, pausing, leaving the page or hiding the tab stops the tones. Returning from a hidden tab leaves the simulation paused. Reduced-motion preference starts it paused.

## Implementation

- Plain HTML, CSS and JavaScript ES modules; no framework, runtime CDN, analytics or build dependency.
- `demo/scene.mjs`: original responsive Canvas 2D illustration.
- `demo/simulation.mjs`: deterministic scripted state machine, separate from production logic.
- `demo/app.mjs`: controls, accessible text equivalent, Web Audio opt-in and tab lifecycle.
- The technology section describes the **installed** Python/OpenCV/YOLOX/ONNX Runtime pipeline and optional components; those models do not execute in this demo.

Serving a vehicle is simplified for demonstration. It selects the foremost unserved visit in that lane, releases that visit, then closes the animated arm five seconds after served traffic clears. It does not validate real barrier classification or automatic-service policy. Scripted early contact and simulated alert timing are not measured camera-to-sound latency.

## Run and check locally

From the repository root:

```sh
python -m http.server 8899 --bind 127.0.0.1 --directory demo
node --test demo/tests/*.test.mjs
```

Open `http://127.0.0.1:8899/`. A current browser with ES modules and Canvas 2D is required. Use Node 20 or newer for the dependency-free tests. No Python application dependencies or camera configuration are needed for this preview.

Behavioral tests cover early contacts, per-visit acknowledgement, a distinct follower after ten-second quiet, separate lane service, gate clearance, pedestrian latching, occlusion, lighting negatives, pause/speed/reset, immutable snapshots and long-running tour retention. They test this demo's logic only. Production release results and their limits are documented in [the repair report](FIXES-2026-11-22.md).

## Publish

`.github/workflows/demo-pages.yml` runs the demo tests, packages an explicit seven-file asset allow-list, and deploys to GitHub Pages on relevant pushes to `main` or manual workflow dispatch. GitHub Pages must use **GitHub Actions** as its publishing source. The Pages artifact contains no production source, camera settings, recordings, model weights, calibration, logs or vehicle data. Hosting is public; footage and installation-specific configuration stay private.

The workflow stamps asset URLs with the deployed commit, including imported modules. This prevents a refreshed HTML page from mixing in a browser-cached version of the previous controls. The placeholder query string also works unchanged on the local preview server.

To update the experience, edit `demo/`, run its tests, inspect desktop/mobile layouts and keyboard/audio behavior, then push. Changes to the demo do not deploy or restart the installed alarm application.
