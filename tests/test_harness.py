import json
import os
import tempfile
import unittest

from evalharness import (
    Completion,
    Example,
    LLMJudge,
    MockProvider,
    Pricing,
    compare_runs,
    load_jsonl,
    run_eval,
    summarize,
)
from evalharness.cli import main
from evalharness.judge import parse_judgment
from evalharness.report import load_run, render_markdown, save_run, wilson_interval

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "..", "data", "sample.jsonl")


class FixedProvider:
    """Returns a fixed reply; records prompts it saw."""

    name = "fixed"

    def __init__(self, reply, model="fixed", in_tok=10, out_tok=5):
        self.reply, self.model, self.in_tok, self.out_tok = reply, model, in_tok, out_tok
        self.prompts = []

    def complete(self, prompt, system=None):
        self.prompts.append((prompt, system))
        return Completion(self.reply, self.in_tok, self.out_tok)


class BrokenProvider:
    name, model = "broken", "broken"

    def complete(self, prompt, system=None):
        raise RuntimeError("boom")


def examples():
    return [
        Example(id="a", input="Capital of France?", expected="Paris", metadata={"category": "geo"}),
        Example(id="b", input="2+2?", expected="4", metrics=["numeric_match"], metadata={"category": "math"}),
    ]


class TestDataset(unittest.TestCase):
    def test_sample_dataset_loads(self):
        data = load_jsonl(SAMPLE)
        self.assertGreaterEqual(len(data), 10)
        self.assertEqual(len({e.id for e in data}), len(data))

    def test_duplicate_ids_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write('{"id": 1, "input": "a"}\n{"id": 1, "input": "b"}\n')
        try:
            with self.assertRaises(ValueError):
                load_jsonl(fh.name)
        finally:
            os.unlink(fh.name)


class TestRunner(unittest.TestCase):
    def test_perfect_mock_run(self):
        data = examples()
        provider = MockProvider(known={e.input: e.expected for e in data}, accuracy=1.0)
        run = run_eval(data, provider, metrics=["exact_match"])
        self.assertTrue(all(c.passed for c in run.cases))
        self.assertEqual(summarize(run)["pass_rate"], 1.0)

    def test_per_example_metrics_override(self):
        run = run_eval(examples(), FixedProvider("The answer is 4"), metrics=["exact_match"])
        by_id = {c.example_id: c for c in run.cases}
        self.assertEqual(set(by_id["b"].scores), {"numeric_match"})
        self.assertTrue(by_id["b"].passed)
        self.assertFalse(by_id["a"].passed)

    def test_provider_errors_are_captured(self):
        run = run_eval(examples(), BrokenProvider())
        self.assertTrue(all(not c.passed and "boom" in c.error for c in run.cases))
        self.assertEqual(summarize(run)["errors"], 2)

    def test_cost_and_template_and_system(self):
        provider = FixedProvider("Paris", in_tok=1_000_000, out_tok=1_000_000)
        run = run_eval(
            examples()[:1],
            provider,
            pricing=Pricing(3.0, 15.0),
            system="be brief",
            prompt_template="Q: {input}\nA:",
        )
        self.assertAlmostEqual(run.cases[0].cost_usd, 18.0)
        self.assertEqual(provider.prompts[0], ("Q: Capital of France?\nA:", "be brief"))

    def test_judge_requested_without_judge_is_an_error(self):
        data = [Example(id="j", input="Explain", expected="x", metrics=["llm_judge"])]
        run = run_eval(data, FixedProvider("text"))
        self.assertIn("no judge configured", run.cases[0].error)


class TestJudge(unittest.TestCase):
    def test_parse_json(self):
        self.assertEqual(parse_judgment('{"reasoning": "good", "score": 4}'), (4, "good"))

    def test_parse_with_surrounding_text(self):
        score, _ = parse_judgment('Sure.\n{"reasoning": "ok", "score": 5}\nDone')
        self.assertEqual(score, 5)

    def test_parse_fallback_and_failure(self):
        self.assertEqual(parse_judgment("score: 2")[0], 2)
        self.assertIsNone(parse_judgment("no idea")[0])
        self.assertIsNone(parse_judgment('{"score": 9}')[0])

    def test_judge_scores_are_normalized(self):
        judge = LLMJudge(FixedProvider('{"reasoning": "fine", "score": 5}'))
        data = [Example(id="j", input="Explain X", expected="ref", metrics=["llm_judge"])]
        run = run_eval(data, FixedProvider("an answer"), judge=judge)
        self.assertEqual(run.cases[0].scores, {"llm_judge": 1.0})
        self.assertTrue(run.cases[0].passed)
        self.assertEqual(run.cases[0].judge_reasoning, "fine")

    def test_unparseable_judge_output_fails_case(self):
        judge = LLMJudge(FixedProvider("garbage"))
        data = [Example(id="j", input="Explain X", metrics=["llm_judge"])]
        run = run_eval(data, FixedProvider("an answer"), judge=judge)
        self.assertFalse(run.cases[0].passed)
        self.assertIn("could not be parsed", run.cases[0].error)


class TestReport(unittest.TestCase):
    def test_wilson_interval(self):
        lo, hi = wilson_interval(8, 10)
        self.assertTrue(0.4 < lo < 0.8 < hi < 1.0)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_compare_detects_regression(self):
        data = examples()
        good = run_eval(data, MockProvider({e.input: e.expected for e in data}, accuracy=1.0), name="good")
        bad = run_eval(data, FixedProvider("nope"), name="bad")
        result = compare_runs(good, bad)
        self.assertTrue(result["regressed"])
        self.assertEqual(sorted(result["newly_failing"]), ["a", "b"])
        self.assertFalse(compare_runs(bad, good)["regressed"])
        self.assertFalse(compare_runs(good, bad, max_drop=1.0)["regressed"])

    def test_save_load_roundtrip_and_markdown(self):
        run = run_eval(examples(), FixedProvider("Paris"), name="rt")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.json")
            save_run(run, path)
            with open(path) as fh:
                self.assertIn("summary", json.load(fh))
            loaded = load_run(path)
        self.assertEqual(summarize(loaded)["pass_rate"], summarize(run)["pass_rate"])
        md = render_markdown(loaded, compare_runs(run, loaded))
        self.assertIn("# Eval report: rt", md)
        self.assertIn("Failures", md)


class TestCLI(unittest.TestCase):
    def test_end_to_end_with_regression_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "base.json")
            cand = os.path.join(tmp, "cand.json")
            common = ["run", "--dataset", SAMPLE, "--provider", "mock", "--metrics", "exact_match"]
            self.assertEqual(main(common + ["--mock-accuracy", "1.0", "--out", base]), 0)
            code = main(
                common
                + ["--mock-accuracy", "0.3", "--out", cand, "--baseline", base, "--fail-on-regression",
                   "--report", os.path.join(tmp, "report.md")]
            )
            self.assertEqual(code, 1)
            self.assertTrue(os.path.exists(os.path.join(tmp, "report.md")))
            self.assertEqual(main(["compare", base, cand]), 0)
            self.assertEqual(main(["compare", base, cand, "--fail-on-regression"]), 1)


if __name__ == "__main__":
    unittest.main()
