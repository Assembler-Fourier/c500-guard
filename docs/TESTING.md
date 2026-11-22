# Isolated regression tests

Install `requirements.txt` in a development virtual environment. Node.js must be on PATH for JavaScript behavior tests. Tests for unavailable platform features may skip; inspect the final summary and skip reasons.

```powershell
python tools\run_tests.py
python tools\run_tests.py --pattern test_ai_context.py
python tools\run_tests.py --car-memory
```

The runner copies this source package into a new temporary checkout, puts normalized policy fixtures at the config paths expected by existing regressions, and redirects application and temporary state under that scratch directory. It does not copy your active `config/` or connect to a camera. Fixtures retain test geometry for route, exclusion, separate-vehicle, ACK and continuation behavior; they are not installable calibration. Do not run tests from a production directory.

The main suite includes synthetic images, fake inference and audio, bounded HTTP fixtures and Node DOM behavior checks. Model inference and live sound are not part of these commands. The optional `car-memory` suite is separate.

Labelled barrier replay tests require private camera evidence and skip when it is absent. An explicitly supplied `BARRIERWATCHER_TEST_EVIDENCE` directory can provide `templates-before-incident.npz`, `templates-user-225605.npz` and `labelled-down/scene-0.jpg` through `scene-5.jpg` for two recent incident checks. Older calibration/lighting tests accept an explicit `BARRIERWATCHER_CALIBRATION_FIXTURES` directory containing their labelled images. Such images and calibration are never included in this repository. These tests retain their assertions; the public suite will have more skips than a private run with authorized evidence available. Other skips depend on the platform and installed optional tools.

Release validation and public-package validation are distinct. The original staged release suite result does not by itself prove that the sanitized package, its example config and its fixture runner work. See `current-deployment.md` for the separate recorded results and limitations.

## Dashboard authoring

The public authoring source is in `ui/`. From a disposable development checkout:

```powershell
python tools/build_ui.py .
```

This regenerates `src/dashboard_ui.py`'s bundled page. Do not run it in an active installation.
The private live-preview helper is not distributed. UI bundle reconstruction was checked against
the published source for this release; authoring source and the bundled page match exactly.

## Adversarial QA

The [22 November audit](QA-2026-11-22.md) records seven failures that the original suite missed.
The subsequent [repair release](FIXES-2026-11-22.md) adds reproductions and fixes all seven, with
complete reruns and accelerated decision tests. Passing tests does not establish real-camera
recall. The ten-second quiet period is intentional and remains unchanged.


## Accelerated traffic decision checks

From a disposable development checkout with the test dependencies installed:

```powershell
python tools/simulate_traffic.py --episodes 60000 --seed 20261122 --output outputs/traffic-simulation.json
```

This standalone process supplies synthetic detector packets to the actual tracker, visit memory and alarm/acknowledgement methods. It uses isolated test fixtures and fake output services; do not point it at an active installation. Eight scenario families cover arrivals, followers, quiet-period boundaries, short occlusions, pedestrians and rejected observations. Idle timestamps are skipped: reported virtual hours are not hours of continuous video inference. Run each source comparison in a separate process because Python imports are cached. The complete unittest suite separately tests input guard, barrier, auto-acknowledgement and UI behavior. See [release validation](FIXES-2026-11-22.md).
