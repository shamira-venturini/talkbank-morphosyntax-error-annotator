import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "study_04_context_windows" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_enni_context_ablation import choose_alignment, index_by_structural_key


class AnalyzeEnniContextAblationTests(unittest.TestCase):
    def test_index_by_structural_key_uses_file_speaker_line_and_utterance_index(self):
        rows = [
            {
                "row_id": 10,
                "file_name": "a.cha",
                "speaker": "CHI",
                "line_no": 11,
                "utterance_index_raw": 2,
                "prev_same_speaker_text": "context",
            }
        ]

        indexed = index_by_structural_key(rows)

        self.assertIn(("a.cha", "CHI", 11, 2), indexed)
        self.assertEqual(indexed[("a.cha", "CHI", 11, 2)]["prev_same_speaker_text"], "context")

    def test_choose_alignment_prefers_structural_keys_when_row_ids_collide(self):
        utterance_only_rows = [
            {
                "row_id": 10,
                "file_name": "a.cha",
                "speaker": "CHI",
                "line_no": 11,
                "utterance_index_raw": 2,
            }
        ]
        prev_rows = [
            {
                "row_id": 10,
                "file_name": "b.cha",
                "speaker": "CHI",
                "line_no": 22,
                "utterance_index_raw": 3,
            },
            {
                "row_id": 99,
                "file_name": "a.cha",
                "speaker": "CHI",
                "line_no": 11,
                "utterance_index_raw": 2,
            },
        ]

        strategy, matches, diagnostics = choose_alignment(utterance_only_rows, prev_rows)

        self.assertEqual(strategy, "structural_key")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], 10)
        self.assertEqual(matches[0][2], 99)
        self.assertEqual(diagnostics["shared_row_ids"], 1)
        self.assertEqual(diagnostics["row_id_consistent_keys"], 0)
        self.assertEqual(diagnostics["shared_structural_keys"], 1)

    def test_choose_alignment_uses_row_ids_when_keys_match(self):
        utterance_only_rows = [
            {
                "row_id": 10,
                "file_name": "a.cha",
                "speaker": "CHI",
                "line_no": 11,
                "utterance_index_raw": 2,
            }
        ]
        prev_rows = [
            {
                "row_id": 10,
                "file_name": "a.cha",
                "speaker": "CHI",
                "line_no": 11,
                "utterance_index_raw": 2,
            }
        ]

        strategy, matches, diagnostics = choose_alignment(utterance_only_rows, prev_rows)

        self.assertEqual(strategy, "row_id")
        self.assertEqual(len(matches), 1)
        self.assertEqual(diagnostics["shared_row_ids"], 1)
        self.assertEqual(diagnostics["row_id_consistent_keys"], 1)
        self.assertEqual(diagnostics["shared_structural_keys"], 1)


if __name__ == "__main__":
    unittest.main()
