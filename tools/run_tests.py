"""Run public regressions in a disposable source/config/state tree."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='test_*.py')
    parser.add_argument('--car-memory', action='store_true', help='Run the separate appearance-report tests')
    parser.add_argument('--temp-root', type=Path, help='Optional parent for the disposable test checkout')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if args.temp_root:
        args.temp_root = args.temp_root.resolve()
        args.temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='barrierwatcher-tests-', dir=args.temp_root) as temporary:
        scratch = Path(temporary)
        checkout = scratch / 'source'
        checkout.mkdir()
        # Explicit source tree only: never copy a local active config or runtime data.
        for folder in ('src', 'tests', 'tools', 'car-memory', 'assets', 'app'):
            original = source / folder
            if original.is_dir():
                shutil.copytree(original, checkout / folder, ignore=shutil.ignore_patterns(
                    '__pycache__', '*.pyc', '.venv', 'node_modules', 'data', 'models',
                    '*.exe', '*.dll', '*.db', '*.sqlite*', 'report.html', 'worker-status.json',
                    'AUTHORISATION*.txt', 'AUTHORIZATION*.txt'))
        for name in ('install.ps1', 'install.sh', 'requirements.txt', 'requirements-ml.txt',
                     'requirements-alignment.txt', 'Realign camera.cmd'):
            if (source / name).is_file():
                shutil.copyfile(source / name, checkout / name)
        shutil.copytree(source / 'tests/fixtures/config', checkout / 'config')
        shutil.copyfile(source / 'config/zones.example.json', checkout / 'config/zones.example.json')
        state = scratch / 'state'
        temp = scratch / 'temp'
        state.mkdir(); temp.mkdir()
        env = dict(os.environ)
        env.update({
            'BARRIERWATCHER_STATE': str(state), 'LOCALAPPDATA': str(state),
            'XDG_STATE_HOME': str(state), 'APPDATA': str(state),
            'TEMP': str(temp), 'TMP': str(temp), 'TMPDIR': str(temp),
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPATH': os.pathsep.join(str(checkout / name) for name in ('src', 'tests', 'car-memory')),
        })
        # Tests supply fake audio/inference. Constrain OpenCV's independent CPU pool.
        command = (
            'import cv2,sys,unittest; cv2.setNumThreads(1); '
            'suite=unittest.defaultTestLoader.discover(sys.argv[1],pattern=sys.argv[2]); '
            'result=unittest.TextTestRunner(verbosity=2).run(suite); '
            'raise SystemExit(0 if result.wasSuccessful() else 1)'
        )
        folder = 'car-memory' if args.car_memory else 'tests'
        print(f'Running {folder}/{args.pattern} in an isolated temporary checkout.', flush=True)
        flags = (subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else 0
        return subprocess.call([sys.executable, '-B', '-c', command, folder, args.pattern],
                               cwd=checkout, env=env, creationflags=flags,
                               stdout=sys.stdout, stderr=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
