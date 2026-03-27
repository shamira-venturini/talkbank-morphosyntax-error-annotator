import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_ood_gillam_dataset import infer_group_and_age, main as prepare_main


class PrepareOodGillamDatasetTests(unittest.TestCase):
    def test_infer_group_age_from_nested_path(self):
        corpus = Path("/tmp/Gillam")
        p = corpus / "SLI" / "7f" / "sample.cha"
        meta = infer_group_and_age(p, corpus)
        self.assertEqual(meta["group"], "SLI")
        self.assertEqual(meta["age_dir"], "7f")
        self.assertEqual(meta["age_years"], "7")

    def test_infer_group_age_from_flat_filename(self):
        corpus = Path("/tmp/Gillam")
        p = corpus / "TD_10m_46733nj.cha"
        meta = infer_group_and_age(p, corpus)
        self.assertEqual(meta["group"], "TD")
        self.assertEqual(meta["age_dir"], "10m")
        self.assertEqual(meta["age_years"], "10")

    def test_main_writes_gillam_outputs(self):
        content = "\n".join(
            [
                "@UTF8",
                "@Participants:\tCHI Target_Child, INV Investigator",
                "*CHI:\the go there .",
                "*INV:\twhat happened ?",
                "*CHI:\tyesterday he go park .",
                "@End",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            corpus_dir = root / "Gillam"
            out_dir = root / "prepared"
            corpus_dir.mkdir(parents=True, exist_ok=True)
            (corpus_dir / "SLI_8f_sample.cha").write_text(content + "\n", encoding="utf-8")

            argv = sys.argv[:]
            try:
                sys.argv = [
                    "prepare_ood_gillam_dataset.py",
                    "--corpus-dir",
                    str(corpus_dir),
                    "--out-dir",
                    str(out_dir),
                    "--speaker-policy",
                    "dominant",
                    "--min-word-count",
                    "1",
                ]
                prepare_main()
            finally:
                sys.argv = argv

            prepared = out_dir / "gillam_utterances.jsonl"
            self.assertTrue(prepared.exists())
            rows = [json.loads(line) for line in prepared.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["corpus"], "Gillam")
            self.assertEqual(rows[0]["group"], "SLI")
            self.assertEqual(rows[0]["age_years"], "8")


if __name__ == "__main__":
    unittest.main()
