#!/usr/bin/env bash
# c500-guard installer for macOS and Linux. Safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ok()   { printf '  \033[32m[ OK ]\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$1"; }
die()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; exit 1; }

echo; echo "c500-guard installer"; echo "===================="
echo "  Full runbook: $ROOT/docs/WINDOWS.md"
echo "  Section 4 covers every error seen so far, with its fix."

command -v python3 >/dev/null || die "python3 not found. Install Python 3.10+."
python3 - <<'PY' || exit 1
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"  Python 3.10+ required, found {sys.version.split()[0]}")
PY
ok "python3 $(python3 -c 'import sys;print(sys.version.split()[0])')"

if ! command -v ffmpeg >/dev/null; then
  if command -v brew >/dev/null; then
    echo "  installing ffmpeg via brew..."; brew install ffmpeg
  elif command -v apt-get >/dev/null; then
    echo "  installing ffmpeg via apt..."; sudo apt-get update && sudo apt-get install -y ffmpeg
  else
    die "ffmpeg not found and no brew/apt available. Install ffmpeg, then re-run."
  fi
fi
ok "ffmpeg $(command -v ffmpeg)"

[ -d .venv ] || python3 -m venv .venv
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install --quiet numpy pillow
ok "python packages installed"

./.venv/bin/python src/bridge.py install
./.venv/bin/python src/bridge.py init
ok "go2rtc installed and checksum-verified"

[ -f config/zones.json ] || cp config/zones.example.json config/zones.json
ok "config/zones.json ready"

echo
./.venv/bin/python src/ctl.py doctor || true
cat <<'NEXT'

NEXT STEPS
  1. Connect the camera (once):
       ./.venv/bin/python src/bridge.py start
       ./.venv/bin/python src/bridge.py login-details
     Open the printed URL, expand Xiaomi, sign in with your NUMERIC Mi Account
     ID (not your email), pick your region, then add the camera.

  2. Draw zones for your view:
       ./.venv/bin/python src/watch.py snapshot rtsp://127.0.0.1:8554/c500 --out snapshot.png

  3. Run it:
       ./.venv/bin/python src/ctl.py start
     Then open http://127.0.0.1:8787

  On macOS/Linux use the Turn ON/OFF button or 'ctl.py toggle'.
  The Ctrl+Alt+F4 global hotkey is Windows-only.

IF SOMETHING GOES WRONG
  docs/WINDOWS.md - setup and troubleshooting.
  The two that cost the most time:
    - Xiaomi login needs your NUMERIC Mi Account ID, not your email (error 70016)
    - Login can succeed and still list no devices if the region is wrong
NEXT
