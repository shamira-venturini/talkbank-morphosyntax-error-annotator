import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_acl_splits import build_chat_tokens
from set_experiment_prompts import build_prompt


class CompositionalAblationTests(unittest.TestCase):
    def test_hybrid_tokens_include_full_detailed_labels(self):
        records = [{"output": "foo [* m:++ed:i] bar"}]
        tokens = build_chat_tokens(records, strategy="hybrid")
        self.assertIn("[* m:++ed:i]", tokens)

    def test_component_tokens_drop_full_detailed_labels(self):
        records = [{"output": "foo [* m:++ed:i] bar"}]
        tokens = build_chat_tokens(records, strategy="components")
        self.assertNotIn("[* m:++ed:i]", tokens)
        self.assertIn("[* m:++", tokens)
        self.assertIn(":ed]", tokens)
        self.assertIn(":i]", tokens)

    def test_compositional_prompt_adds_scheme_constraints(self):
        prompt = build_prompt(["[* m:03s:a]"], reconstruction_mode="nonword_only", prompt_style="compositional")
        self.assertIn("Build each CHAT error tag compositionally", prompt)
        self.assertIn("Use m:* only", prompt)
        self.assertIn("Use :a only", prompt)
        self.assertIn("Use :i only", prompt)

    def test_standard_prompt_remains_simple(self):
        prompt = build_prompt(["[* m:03s:a]"], reconstruction_mode="nonword_only", prompt_style="standard")
        self.assertNotIn("Build each CHAT error tag compositionally", prompt)
        self.assertIn("Use reconstruction markers only for nonword corrections", prompt)


if __name__ == "__main__":
    unittest.main()
