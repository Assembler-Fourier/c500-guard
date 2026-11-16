"""Dispatch a fixed, independent Windows shutdown task; never a watcher child."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import platform_support as ps


def request_full_off() -> None:
    if not ps.WINDOWS:
        raise RuntimeError('Full power control is installed on Windows only.')
    scheduler = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'schtasks.exe'
    try:
        result = subprocess.run(
            [str(scheduler), '/Run', '/TN', 'C500 Guard OFF'],
            capture_output=True, text=True, timeout=8,
            **ps.background_subprocess_kwargs())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('Shutdown could not be confirmed. Use BarrierWatcher OFF on the desktop, or press F4.') from exc
    if result.returncode:
        raise RuntimeError('Windows could not start Full OFF. Use BarrierWatcher OFF on the desktop; check the power-control installation if it also fails.')

