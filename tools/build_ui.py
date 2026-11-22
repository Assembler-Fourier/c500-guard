"""Build the dashboard page from ui/ and write it into src/dashboard_ui.py's PAGE_TEMPLATE (21 Nov 2026 revamp).

Modules keep the existing bundle format: each file defines __c500Modules["<name>"] = (() => {...})(); and app.js
reads its dependencies from that table. Usage: python build_ui.py <stage folder>
"""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve()
ui = root / "ui"
MODULES = ["zones.js", "state.js", "runtime.js", "app.js"]

bundle = "\n(() => {\n'use strict';\nconst __c500Modules = Object.create(null);\n\n"
for name in MODULES:
    text = (ui / name).read_text(encoding="utf-8")
    if not text.lstrip().startswith(f'__c500Modules["{name}"] = (() => {{'):
        raise SystemExit(f"{name}: not in bundle module form")
    bundle += f"// Module: {name}\n{text.rstrip()}\n\n"
bundle += "})();\n"
if "</script" in bundle.lower():
    raise SystemExit("a module contains </script")

page = (ui / "index.html").read_text(encoding="utf-8")
for key, value in (("__THEME_CSS__", (ui / "theme.css").read_text(encoding="utf-8")),
                   ("__ZONES_CSS__", (ui / "zones.css").read_text(encoding="utf-8")),
                   ("__SCRIPT__", bundle)):
    if page.count(key) != 1:
        raise SystemExit(f"index.html needs exactly one {key}")
    page = page.replace(key, value)

target = root / "src" / "dashboard_ui.py"
raw = target.read_bytes().decode("utf-8")
newline = "\r\n" if "\r\n" in raw else "\n"
text = raw.replace("\r\n", "\n")
lines = text.split("\n")
hits = [i for i, line in enumerate(lines) if line.startswith("PAGE_TEMPLATE = ")]
if len(hits) != 1:
    raise SystemExit(f"expected one PAGE_TEMPLATE line, found {len(hits)}")
lines[hits[0]] = "PAGE_TEMPLATE = " + repr(page)
out = "\n".join(lines)
target.write_bytes((out.replace("\n", "\r\n") if newline == "\r\n" else out).encode("utf-8"))
print(f"page {len(page)} chars ({len(bundle)} script) -> {target}")
