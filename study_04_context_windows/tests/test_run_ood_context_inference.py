import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "study_04_context_windows" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_ood_context_inference import build_augmented_input


class RunOodContextInferenceTests(unittest.TestCase):
    def test_prev_same_speaker_uses_precomputed_field_when_available(self):
        row = {
            "row_id": 1,
            "speaker": "CHI",
            "input": "he go .",
            "prev_same_speaker_text": "they are eating .",
            "prev_same_speaker_input": "PRECOMPUTED",
        }
        context_map = {1: {"prev_rows": [], "pool_rows": []}}

        prepared = build_augmented_input(
            row=row,
            context_mode="prev_same_speaker",
            context_map=context_map,
            local_prev_k=1,
            max_context_chars=1000,
        )

        self.assertEqual(prepared["input_text"], "PRECOMPUTED")
        self.assertEqual(prepared["context_text"], "they are eating .")
        self.assertEqual(prepared["context_utterance_count"], 1)

    def test_prev_same_speaker_falls_back_to_latest_same_speaker_row(self):
        row = {
            "row_id": 2,
            "speaker": "CHI",
            "input": "he go .",
        }
        context_map = {
            2: {
                "prev_rows": [
                    {"row_id": 1, "speaker": "CHI", "utterance_index_raw": 7, "input": "they are eating ."},
                ],
                "pool_rows": [],
            }
        }

        prepared = build_augmented_input(
            row=row,
            context_mode="prev_same_speaker",
            context_map=context_map,
            local_prev_k=1,
            max_context_chars=1000,
        )

        self.assertIn("Target utterance to annotate", prepared["input_text"])
        self.assertIn("they are eating .", prepared["input_text"])
        self.assertEqual(prepared["context_utterance_count"], 1)


if __name__ == "__main__":
    unittest.main()
