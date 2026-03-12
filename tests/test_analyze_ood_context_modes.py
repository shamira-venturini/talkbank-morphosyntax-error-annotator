import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_ood_context_modes import main as analyze_main


class AnalyzeOodContextModesTests(unittest.TestCase):
    def test_pairwise_outputs_are_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pred_dir = root / "pred"
            out_dir = root / "out"
            pred_dir.mkdir(parents=True, exist_ok=True)

            base_rows = [
                {
                    "row_id": 1,
                    "file_name": "a.cha",
                    "speaker": "PAR0",
                    "line_no": 10,
                    "input": "he go .",
                    "model_prediction": "he go [:: goes] [* m:03s:a] .",
                    "pred_tag_count": 1,
                    "context_utterance_count": 0,
                    "context_char_count": 0,
                    "uncertainty_mean_token_logprob": -0.2,
                    "uncertainty_seq_logprob": -1.0,
                    "uncertainty_min_token_logprob": -0.7,
                    "uncertainty_mean_token_margin": 5.0,
                    "uncertainty_min_token_margin": 1.0,
                },
                {
                    "row_id": 2,
                    "file_name": "a.cha",
                    "speaker": "PAR0",
                    "line_no": 11,
                    "input": "they swim .",
                    "model_prediction": "they swim .",
                    "pred_tag_count": 0,
                    "context_utterance_count": 0,
                    "context_char_count": 0,
                    "uncertainty_mean_token_logprob": -0.1,
                    "uncertainty_seq_logprob": -0.6,
                    "uncertainty_min_token_logprob": -0.4,
                    "uncertainty_mean_token_margin": 6.0,
                    "uncertainty_min_token_margin": 1.2,
                },
            ]
            local_rows = [
                {
                    **base_rows[0],
                    "model_prediction": "he go [:: goes] [* m:03s:a] .",
                    "context_utterance_count": 1,
                    "context_char_count": 30,
                    "uncertainty_mean_token_logprob": -0.15,
                },
                {
                    **base_rows[1],
                    "model_prediction": "they swim [:: swims] [* m:03s:a] .",
                    "pred_tag_count": 1,
                    "context_utterance_count": 1,
                    "context_char_count": 30,
                    "uncertainty_mean_token_logprob": -0.25,
                },
            ]

            with (pred_dir / "predictions_utterance_only.jsonl").open("w", encoding="utf-8") as handle:
                for row in base_rows:
                    handle.write(json.dumps(row) + "\n")
            with (pred_dir / "predictions_local_prev.jsonl").open("w", encoding="utf-8") as handle:
                for row in local_rows:
                    handle.write(json.dumps(row) + "\n")

            argv = sys.argv[:]
            try:
                sys.argv = [
                    "analyze_ood_context_modes.py",
                    "--predictions-dir",
                    str(pred_dir),
                    "--baseline-mode",
                    "utterance_only",
                    "--out-dir",
                    str(out_dir),
                ]
                analyze_main()
            finally:
                sys.argv = argv

            self.assertTrue((out_dir / "context_mode_summary.csv").exists())
            self.assertTrue((out_dir / "pairwise_vs_baseline.csv").exists())
            self.assertTrue((out_dir / "changed_items.csv").exists())
            self.assertTrue((out_dir / "manual_review_changed_outputs.csv").exists())

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["shared_rows"], 2)
            self.assertGreaterEqual(summary["changed_item_rows"], 1)


if __name__ == "__main__":
    unittest.main()
