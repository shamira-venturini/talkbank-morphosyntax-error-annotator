import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ood_chat_utils import parse_chat_file, select_speakers


class OodChatUtilsTests(unittest.TestCase):
    def test_parse_chat_file_ignores_mor_gra_and_cleans_timestamps(self):
        content = "\n".join(
            [
                "@UTF8",
                "@Participants:\tPAR0 Participant, PAR1 Participant",
                "*PAR0:\thello there . \x15100_200\x15",
                "%mor:\tintj|hello adv|there .",
                "%gra:\t1|0|ROOT",
                "*PAR1:\tI [/] I go .",
                "\tand then stop . \x15210_240\x15",
                "@End",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "sample.cha"
            p.write_text(content + "\n", encoding="utf-8")
            parsed = parse_chat_file(p)

        self.assertEqual(parsed["participants"], ["PAR0", "PAR1"])
        self.assertEqual(len(parsed["utterances"]), 2)
        self.assertEqual(parsed["utterances"][0]["speaker"], "PAR0")
        self.assertEqual(parsed["utterances"][0]["text"], "hello there .")
        self.assertEqual(parsed["utterances"][1]["text"], "I [/] I go . and then stop .")

    def test_select_speakers_policy(self):
        utterances = [
            {"speaker": "PAR0"},
            {"speaker": "PAR0"},
            {"speaker": "PAR1"},
        ]
        participants = ["PAR1", "PAR0"]

        self.assertEqual(select_speakers(utterances, participants, "first_participant", []), ["PAR1"])
        self.assertEqual(select_speakers(utterances, participants, "dominant", []), ["PAR0"])
        self.assertEqual(select_speakers(utterances, participants, "all", []), ["PAR0", "PAR1"])
        self.assertEqual(select_speakers(utterances, participants, "dominant", ["PAR1"]), ["PAR1"])


if __name__ == "__main__":
    unittest.main()
