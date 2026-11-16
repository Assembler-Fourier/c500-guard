"""The landmark budget is tunable, and a silly value can never break tracking."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_watch import watcher  # noqa: E402
from scene_alignment import AlignmentConfig  # noqa: E402


class AlignerBudgetTests(unittest.TestCase):
    def build(self, alignment):
        obj = watcher()
        obj.alignment_config = alignment
        return obj

    def test_default_is_the_library_default(self):
        self.assertEqual(self.build({})._aligner_config().max_features,
                         AlignmentConfig().max_features)

    def test_a_configured_budget_is_used(self):
        self.assertEqual(self.build({"max_features": 900})._aligner_config().max_features, 900)

    def test_too_few_landmarks_to_track_with_is_refused(self):
        """Tracking needs hundreds of landmarks; 5 would silently stop alignment."""
        self.assertEqual(self.build({"max_features": 5})._aligner_config().max_features, 400)

    def test_an_absurd_budget_is_capped_rather_than_obeyed(self):
        self.assertEqual(self.build({"max_features": 500000})._aligner_config().max_features, 3000)

    def test_nonsense_values_leave_the_default_in_place(self):
        for value in ("many", None, [1, 2], {"a": 1}):
            self.assertEqual(self.build({"max_features": value})._aligner_config().max_features,
                             AlignmentConfig().max_features)

    def test_a_missing_alignment_block_is_fine(self):
        obj = watcher()
        obj.alignment_config = None
        self.assertEqual(obj._aligner_config().max_features, AlignmentConfig().max_features)


if __name__ == "__main__":
    unittest.main()
