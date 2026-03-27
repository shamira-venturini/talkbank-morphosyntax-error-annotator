import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY4_SCRIPTS = ROOT / "study_04_context_windows" / "scripts"
if str(STUDY4_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STUDY4_SCRIPTS))

from prepare_enni_context_ablation_input import (
    build_prev_same_speaker_input,
    load_file_manifest,
    previous_same_speaker,
)


class PrepareEnniContextAblationInputTests(unittest.TestCase):
    def test_previous_same_speaker_finds_latest_matching_speaker(self):
        utterances = [
            {"speaker": "EXA", "text": "prompt", "line_no": 1, "raw_index": 0},
            {"speaker": "CHI", "text": "first", "line_no": 2, "raw_index": 1},
            {"speaker": "EXA", "text": "prompt two", "line_no": 3, "raw_index": 2},
            {"speaker": "CHI", "text": "second", "line_no": 4, "raw_index": 3},
            {"speaker": "CHI", "text": "target", "line_no": 5, "raw_index": 4},
        ]

        prev = previous_same_speaker(utterances, target_index=4, speaker="CHI")
        self.assertIsNotNone(prev)
        self.assertEqual(prev["text"], "second")

    def test_build_prev_same_speaker_input_falls_back_to_target_when_missing(self):
        self.assertEqual(build_prev_same_speaker_input("", "he go ."), "he go .")

    def test_build_prev_same_speaker_input_wraps_context_and_target(self):
        actual = build_prev_same_speaker_input("they are eating .", "he go .")
        self.assertIn("Context (for disambiguation only", actual)
        self.assertIn("[PREV_SAME_SPEAKER] they are eating .", actual)
        self.assertTrue(actual.endswith("he go ."))

    def test_load_file_manifest_normalizes_to_basenames_and_skips_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "files.txt"
            manifest.write_text(
                "# pilot selection\n"
                "studies/04_context_windows_pilot/ENNI/TD/B/716.cha\n"
                "903.cha\n"
                "\n",
                encoding="utf-8",
            )

            actual = load_file_manifest(manifest)

        self.assertEqual(actual, {"716.cha", "903.cha"})


if __name__ == "__main__":
    unittest.main()
