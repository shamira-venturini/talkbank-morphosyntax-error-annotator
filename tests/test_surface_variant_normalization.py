import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from add_error_count import align_input_surface_variants, normalize_surface_variants


class SurfaceVariantNormalizationTests(unittest.TestCase):
    def test_normalizes_selected_variants_in_plain_text(self):
        text = "then rabbit do nt want a balloon cause of a other story and nother sticker ."
        normalized, counts = normalize_surface_variants(text)
        self.assertIn("do n(o)t", normalized)
        self.assertIn("a(n)other story", normalized)
        self.assertIn("(a)nother sticker", normalized)
        self.assertIn("(be)cause", normalized)
        self.assertGreaterEqual(counts["token_nt"], 1)
        self.assertGreaterEqual(counts["token_a_other"], 1)
        self.assertGreaterEqual(counts["token_nother"], 1)
        self.assertGreaterEqual(counts["token_cause"], 1)

    def test_does_not_modify_bracketed_chunks(self):
        text = "do nt [:: not] [* m:03s:a] and cause [:: because] [* s:r:prep] ."
        normalized, counts = normalize_surface_variants(text)
        self.assertIn("[:: not]", normalized)
        self.assertIn("[:: because]", normalized)
        self.assertIn("do n(o)t", normalized)
        self.assertIn("(be)cause", normalized)
        self.assertEqual(normalized.count("[:: because]"), 1)
        self.assertEqual(normalized.count("[:: not]"), 1)
        self.assertGreaterEqual(sum(counts.values()), 2)

    def test_does_not_double_normalize_parenthesized_cause(self):
        text = "she said (be)cause it was late ."
        normalized, counts = normalize_surface_variants(text)
        self.assertEqual(normalized, text)
        self.assertEqual(counts["token_cause"], 0)

    def test_does_not_double_normalize_parenthesized_nother(self):
        text = "now (a)nother sticker ."
        normalized, counts = normalize_surface_variants(text)
        self.assertEqual(normalized, text)
        self.assertEqual(counts["token_nother"], 0)

    def test_does_not_normalize_substrings_in_lengthened_tokens(self):
        text = "and then they we:nt and a::nother kid ran ."
        normalized, counts = normalize_surface_variants(text)
        self.assertEqual(normalized, text)
        self.assertEqual(counts["token_nt"], 0)
        self.assertEqual(counts["token_nother"], 0)

    def test_aligns_optional_g_from_output_surface(self):
        input_text = "lookin at the balloons ."
        output_text = "lookin(g) at the balloons ."
        aligned, counts = align_input_surface_variants(input_text, output_text)
        self.assertEqual(aligned, "lookin(g) at the balloons .")
        self.assertEqual(counts["align_optional_g"], 1)

    def test_aligns_known_variant_from_output_surface(self):
        input_text = "then rabbit do nt want a balloon ."
        output_text = "then rabbit do n(o)t want a balloon ."
        aligned, counts = align_input_surface_variants(input_text, output_text)
        self.assertEqual(aligned, "then rabbit do n(o)t want a balloon .")
        self.assertEqual(counts["align_nt_to_norm"], 1)

    def test_aligns_back_to_raw_when_output_has_raw_a_other(self):
        input_text = "then suddenly he wanted a(n)other one ."
        output_text = "then suddenly he wanted a [* m:allo] other one ."
        aligned, counts = align_input_surface_variants(input_text, output_text)
        self.assertEqual(aligned, "then suddenly he wanted a other one .")
        self.assertEqual(counts["align_a_other_to_raw"], 1)

    def test_does_not_force_grammatical_rewrite_alignment(self):
        input_text = "and close hims eyes ."
        output_text = "and close his eyes ."
        aligned, counts = align_input_surface_variants(input_text, output_text)
        self.assertEqual(aligned, input_text)
        self.assertEqual(sum(counts.values()), 0)


if __name__ == "__main__":
    unittest.main()
