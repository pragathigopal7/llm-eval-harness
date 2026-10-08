import unittest

from evalharness.metrics import (
    contains,
    exact_match,
    json_valid,
    normalize,
    numeric_match,
    regex_match,
    score_metrics,
    token_f1,
)
from evalharness.models import Example


def ex(expected=None, **metadata):
    return Example(id="x", input="q", expected=expected, metadata=metadata)


class TestMetrics(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("  The Quick, brown FOX! "), "quick brown fox")

    def test_exact_match(self):
        self.assertEqual(exact_match("Canberra.", "canberra", ex()), 1.0)
        self.assertEqual(exact_match("Sydney", "Canberra", ex()), 0.0)

    def test_contains(self):
        self.assertEqual(contains("It is O(log n) time.", "O(log n)", ex()), 1.0)
        self.assertEqual(contains("O(n)", "O(log n)", ex()), 0.0)

    def test_token_f1(self):
        self.assertAlmostEqual(token_f1("Gabriel Garcia Marquez", "Gabriel Garcia Marquez", ex()), 1.0)
        self.assertAlmostEqual(token_f1("Garcia Marquez", "Gabriel Garcia Marquez", ex()), 0.8)
        self.assertEqual(token_f1("nothing here", "Gabriel", ex()), 0.0)

    def test_numeric_match_uses_last_number(self):
        self.assertEqual(numeric_match("7 pens cost $21, so change is 29", "29", ex()), 1.0)
        self.assertEqual(numeric_match("The answer is 1,200.", "1200", ex()), 1.0)
        self.assertEqual(numeric_match("72.0 km/h", "72", ex()), 1.0)
        self.assertEqual(numeric_match("no numbers", "3", ex()), 0.0)

    def test_numeric_match_leading_decimal(self):
        self.assertEqual(numeric_match("The probability is .5", "0.5", ex()), 1.0)
        self.assertEqual(numeric_match("Slope: -.25", "-0.25", ex()), 1.0)
        self.assertEqual(numeric_match("Done.", "0", ex()), 0.0)

    def test_regex_match_prefers_metadata_pattern(self):
        self.assertEqual(regex_match("date: 2025-01-01", "zzz", ex(pattern=r"\d{4}-\d{2}-\d{2}")), 1.0)
        self.assertEqual(regex_match("hello", "HEL", ex()), 1.0)

    def test_json_valid(self):
        self.assertEqual(json_valid('{"a": 1}', None, ex()), 1.0)
        self.assertEqual(json_valid('```json\n{"a": 1}\n```', None, ex()), 1.0)
        self.assertEqual(json_valid("{a: 1}", None, ex()), 0.0)

    def test_score_metrics_skips_metrics_without_reference(self):
        scores = score_metrics('{"a": 1}', ex(expected=None), ["exact_match", "json_valid"])
        self.assertEqual(scores, {"json_valid": 1.0})

    def test_unknown_metric(self):
        with self.assertRaises(KeyError):
            score_metrics("x", ex("x"), ["nope"])


if __name__ == "__main__":
    unittest.main()
