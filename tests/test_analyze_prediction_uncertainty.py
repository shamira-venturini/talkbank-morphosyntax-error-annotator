import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_prediction_uncertainty import exact_tag_set_correct, summarize_items


class AnalyzePredictionUncertaintyTests(unittest.TestCase):
    def test_exact_tag_set_correct_ignores_surface_recon_differences(self):
        row = {
            "human_gold": "he go [:: goes] [* m:03s:a] home .",
            "model_prediction": "he go [: goes] [* m:03s:a] home .",
        }
        self.assertEqual(exact_tag_set_correct(row), 1)

    def test_summary_prefers_higher_confidence_for_correct_items(self):
        items = [
            {"uncertainty_mean_token_logprob": -0.2, "exact_tag_set_correct": 1},
            {"uncertainty_mean_token_logprob": -0.3, "exact_tag_set_correct": 1},
            {"uncertainty_mean_token_logprob": -1.8, "exact_tag_set_correct": 0},
            {"uncertainty_mean_token_logprob": -2.0, "exact_tag_set_correct": 0},
        ]
        summary = summarize_items(
            items,
            metric="uncertainty_mean_token_logprob",
            correctness_key="exact_tag_set_correct",
            n_bins=2,
        )
        self.assertEqual(summary["n_items"], 4)
        self.assertGreater(summary["auc"], 0.9)
        self.assertGreater(summary["top_quartile_accuracy"], summary["bottom_quartile_accuracy"])


if __name__ == "__main__":
    unittest.main()
