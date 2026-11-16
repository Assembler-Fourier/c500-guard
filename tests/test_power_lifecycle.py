"""Run the mock-only controller contract checks in Windows PowerShell 5.1."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class PowerLifecycleTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'win32', 'Windows PowerShell 5.1')
    def test_mocked_lifecycle(self):
        executable = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
        result = subprocess.run([str(executable), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
            str(Path(__file__).with_name('power_lifecycle.ps1'))],
            capture_output=True, text=True, timeout=30, creationflags=0x08000000)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('mocked controller checks passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
