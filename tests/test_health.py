"""Speaker check and log rotation helpers. The speaker check never changes settings."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import platform_support as ps


class HealthHelperTests(unittest.TestCase):
    def test_rotate_log_keeps_one_previous_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "watcher.log"
            log.write_bytes(b"x" * 20)
            ps.rotate_log(log, max_bytes=10)
            self.assertFalse(log.exists())
            self.assertEqual((Path(folder) / "watcher.log.1").read_bytes(), b"x" * 20)
            log.write_bytes(b"y" * 5)
            ps.rotate_log(log, max_bytes=10)
            self.assertEqual(log.read_bytes(), b"y" * 5)
            ps.rotate_log(Path(folder) / "missing.log")

    @unittest.skipUnless(sys.platform.startswith("win"), "Windows Core Audio")
    def test_speaker_status_is_well_formed_on_any_thread(self):
        results = [ps.audio_output_status()]
        worker = threading.Thread(target=lambda: results.append(ps.audio_output_status()))
        worker.start()
        worker.join(10)
        for status in results:
            if not status["available"]:
                self.skipTest(status.get("detail", "no speaker"))
            self.assertIsInstance(status["muted"], bool)
            self.assertGreaterEqual(status["volume"], 0)
            self.assertLessEqual(status["volume"], 1)
            self.assertIsInstance(status["device"], str)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
