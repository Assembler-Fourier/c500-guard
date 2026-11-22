# BarrierWatcher / c500-guard

A local camera dashboard and attention alarm for a supervised entrance, exit and pedestrian route. Python processes the camera feed; a browser or optional Windows WebView2 window displays the controls. Camera calibration, models and recordings remain on the installation machine.

The application combines object observations, configured route geometry, image-change cues and separately calibrated barrier evidence. It provides lane-specific alarm tones, acknowledgement, visit grouping, stale-camera warnings, local incident recording and an optional vehicle appearance report. It does not operate a physical barrier. Detection and appearance matching can be wrong or late; they assist an operator and do not establish identity or guarantee that a vehicle was detected.

**22 November audit:** this source snapshot includes earlier implemented work and documents seven open defects. No fixes were applied during the audit. In particular, a separately observed follower can inherit an acknowledged visit; see the QA report before assessing deployment readiness.

## Start here

- [Windows setup and operation](docs/WINDOWS.md)
- [Architecture and alarm policy](docs/ARCHITECTURE.md)
- [Models and optional components](docs/MODELS.md)
- [Running isolated regression tests](docs/TESTING.md)
- [Private data and retention](docs/PRIVACY.md)
- [Current release verification](docs/current-deployment.md)
- [Independent QA: seven open findings](docs/QA-2026-11-22.md)
- [Comparison research and proposed upgrades](docs/UPGRADE-RESEARCH-2026-11-22.md)

Python 3.10 or newer and FFmpeg are required. The Windows integrations use scheduled tasks, native input hooks and optional Intel graphics acceleration. Core detector code also has CPU support. The native window source is in `app/`; a browser remains available when that optional executable is absent.

The repository intentionally contains no active camera configuration, credentials, calibration templates, model weights, recordings, vehicle database or site screenshots. `config/zones.example.json` is a generic starting point. The normalized geometry under `tests/fixtures/` exists only for repeatable regression tests and must not be deployed as camera calibration.

For a synthetic, silent preview after installing the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\watch.py watch selftest --zones config\zones.example.json --silent
```

Open the local dashboard at `http://127.0.0.1:8787`. This checks the visual path using generated input; it does not validate real detection, audible volume or camera alignment.

The project source is MIT licensed. Separately obtained runtimes, bridge software and model weights retain their own licenses; see [models and components](docs/MODELS.md).
