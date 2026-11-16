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
