import sys
import tempfile
import unittest
import subprocess
import json
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

    def test_prepare_script_uses_fallback_transcript_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_jsonl = tmp / "input.jsonl"
            primary_dir = tmp / "clean"
            fallback_dir = tmp / "fallback"
            out_jsonl = tmp / "out.jsonl"
            out_summary = tmp / "summary.json"

            (fallback_dir / "TD" / "B").mkdir(parents=True)
            (fallback_dir / "TD" / "B" / "999.cha").write_text(
                "@UTF8\n"
                "@Participants:\tCHI Target_Child, EXA Investigator\n"
                "*CHI:\tfirst .\n"
                "*EXA:\tokay .\n"
                "*CHI:\ttarget .\n",
                encoding="utf-8",
            )
            input_jsonl.write_text(
                json.dumps(
                    {
                        "row_id": 1,
                        "source_dataset": "ENNI_OOD",
                        "review_is_reviewed": True,
                        "file_name": "999.cha",
                        "speaker": "CHI",
                        "line_no": 5,
                        "utterance_index_raw": 2,
                        "input": "target .",
                        "output": "target [* m:0ed] .",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(STUDY4_SCRIPTS / "prepare_enni_context_ablation_input.py"),
                    "--input-jsonl",
                    str(input_jsonl),
                    "--enni-dir",
                    str(primary_dir),
                    "--fallback-enni-dir",
                    str(fallback_dir),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-summary",
                    str(out_summary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(out_summary.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(result.returncode, 0)
        self.assertEqual(summary["rows_exported"], 1)
        self.assertEqual(summary["rows_resolved_via_fallback"], 1)
        self.assertEqual(summary["fallback_files_used"], ["999.cha"])
        self.assertEqual(rows[0]["prev_same_speaker_text"], "first .")


if __name__ == "__main__":
    unittest.main()
