import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_blinded_review import analyze_review_bundle, parse_review_decision


class AnalyzeBlindedReviewTests(unittest.TestCase):
    def test_parse_review_decision_handles_single_and_multiple_marks(self):
        self.assertEqual(parse_review_decision({"score_correct": "1"}), "correct")
        self.assertEqual(parse_review_decision({"score_ambiguous": "x"}), "ambiguous")
        self.assertEqual(
            parse_review_decision({"score_correct": "1", "score_incorrect": "1"}),
            "invalid_multiple",
        )
        self.assertEqual(parse_review_decision({}), "unscored")

    def test_analyze_review_bundle_summarizes_model_vs_gold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            review_sheet = tmp / "blinded_review_sheet.csv"
            answer_key = tmp / "answer_key.csv"
            out_dir = tmp / "analysis"

            review_sheet.write_text(
                "\n".join(
                    [
                        "review_id,utterance_id,input,candidate_annotation,score_correct,score_incorrect,score_ambiguous,score_unsure,notes",
                        "R001,U001,she go home .,she go [:: goes] [* m:03s:a] home .,1,,,,",
                        "R002,U001,she go home .,she go [:: went] [* m:0ed] home .,,1,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            answer_key.write_text(
                "\n".join(
                    [
                        "review_id,utterance_id,row_id,source,original_category",
                        "R001,U001,101,final_model,wrong_label",
                        "R002,U001,101,gold,gold_reference",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = analyze_review_bundle(review_sheet, answer_key, out_dir)

            self.assertEqual(summary["n_review_rows"], 2)
            self.assertEqual(summary["decision_counts"]["correct"], 1)
            self.assertEqual(summary["decision_counts"]["incorrect"], 1)
            self.assertEqual(summary["head_to_head_pair_outcomes"]["model_preferred"], 1)
            self.assertTrue((out_dir / "posthoc_review_gold_comparison_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
