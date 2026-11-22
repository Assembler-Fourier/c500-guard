# BarrierWatcher interface — 21 Nov 2026 revamp

Source of the page served at `/` (and inside the app window). From a development checkout,
build with `python tools/build_ui.py .`; it writes the bundled page into
`src/dashboard_ui.py` (`PAGE_TEMPLATE`). Do not run the builder against an active installation.
The private design-preview helper is not distributed; see `docs/TESTING.md` for isolated checks.

## Who it is for
A night controller who glances over: *am I protected, is anything happening, where?* Calm while watching,
unmistakable when alarming, nothing to read that does not help.

## Structure
| Rail | Page | Contents |
|---|---|---|
| Daily | **Live** | Camera (fits one screen), three lane tiles (status, zone count/objects, own barrier UP/DOWN), status card that becomes the alarm card, health chips, recent events |
| | **Events** | Filter (All/Entrance/Exit/Footpath/System), rows with thumbnail, outcome, time; detail dialog with video and Real/Wrong |
| | **Vehicles** | Embedded car-memory report + status bar |
| Occasional | **Setup** | Tabs: Camera (guided framing check), Zones (editor), Barriers (UP/DOWN examples, empty road) |
| | **Settings** | Embedded settings page (same tokens; duplicate status chip hidden when embedded) |
| | **Health** | Camera / AI / alignment / keys tiles, processing chart with the 100 ms budget, sound tests, program facts, Full OFF |
| Bottom | Full OFF | Confirmation dialog |

Top bar on every page: name, one status pill, **Pause / Resume alerts**, clock. Alarm adds a red line under the bar.

## Removed debris
Preview "scenario" simulator and its popover, "DESIGN PREVIEW" footer, Signal lab (analysis drawer), Appearance
options (surface/accent/density/motion), desk-camera line, Analyze button, Camera setup / Barrier setup buttons on
the Live screen, duplicate status texts, lane "Hear" buttons (now in Health › Sound).

## Kept exactly
Status derivation (`state.js` `deriveStatus`), transport and command receipts (`runtime.js`), zone editor
(`zones.js`), barrier helpers, camera-setup flow, vehicle status polling, event/feedback logic, health labels,
page auto-reload on a new `ui_version` (never during an alarm, in Setup/Settings or with a dialog open),
`?app=1` close confirmation. Old page names in a URL hash (`#monitor`, `#system`, …) map to the new ones.

## Tokens
`--bg #090c10`, `--surface #10151b`, `--surface-raised #151b22`, `--line #222a34`, `--text #e8edf2`,
`--muted #8d98a6`, `--accent #5ad1e6`, `--good #41d18f`, `--warn #f2b44c`, `--danger #ff5a55`,
lanes `--entry #f0b45e`, `--exit #5fd3b3`, `--path #b39ef2`. Radius 16/11/8, Segoe UI Variable.
