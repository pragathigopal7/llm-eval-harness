import json
import os
import tempfile
import threading
import unittest

from evalharness import Completion, Example, MockProvider, run_eval, summarize
from evalharness.cli import main
from evalharness.report import load_run, render_markdown, save_run

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "..", "data", "sample.jsonl")


class ScriptedProvider:
    """Replies with the next item of ``script`` on every call (cycling).

    An item that is an Exception is raised instead of returned.
    """

    name, model = "scripted", "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, prompt, system=None):
        with self._lock:
            item = self.script[self.calls % len(self.script)]
            self.calls += 1
        if isinstance(item, Exception):
            raise item
        return Completion(item, 10, 5)


def one_case():
    return [Example(id="a", input="Capital of France?", expected="Paris")]


def run_scripted(script, samples, policy="majority"):
    # concurrency=1 keeps the call order, and so the sample order, fixed.
    return run_eval(one_case(), ScriptedProvider(script), samples=samples, pass_policy=policy, concurrency=1)


class TestPassPolicies(unittest.TestCase):
    def test_policies_on_two_of_three(self):
        script = ["Paris", "nope", "Paris"]
        self.assertTrue(run_scripted(script, 3, "majority").cases[0].passed)
        self.assertFalse(run_scripted(script, 3, "all").cases[0].passed)
        self.assertTrue(run_scripted(script, 3, "any").cases[0].passed)

    def test_majority_needs_more_than_half(self):
        self.assertFalse(run_scripted(["Paris", "nope"], 4, "majority").cases[0].passed)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            run_eval(one_case(), ScriptedProvider(["Paris"]), samples=0)
        with self.assertRaises(ValueError):
            run_eval(one_case(), ScriptedProvider(["Paris"]), samples=2, pass_policy="most")


class TestAggregation(unittest.TestCase):
    def test_single_sample_is_unchanged(self):
        run = run_scripted(["Paris"], 1)
        case = run.cases[0]
        self.assertIsNone(case.sample_passes)
        self.assertIsNone(case.sample_outputs)
        self.assertNotIn("sampling", summarize(run))
        self.assertNotIn("Consistency", render_markdown(run))

    def test_scores_tokens_and_samples(self):
        case = run_scripted(["Paris", "nope", "Paris"], 3).cases[0]
        self.assertEqual(case.sample_passes, [True, False, True])
        self.assertEqual(case.sample_outputs, ["Paris", "nope", "Paris"])
        self.assertAlmostEqual(case.scores["exact_match"], 2 / 3)
        self.assertAlmostEqual(case.pass_fraction, 2 / 3)
        self.assertEqual((case.input_tokens, case.output_tokens), (30, 15))

    def test_failed_case_shows_a_failing_sample(self):
        case = run_scripted(["Paris", "nope", "Paris"], 3, "all").cases[0]
        self.assertEqual(case.output, "nope")

    def test_some_failed_samples_are_counted_not_fatal(self):
        case = run_scripted(["Paris", RuntimeError("boom"), "Paris"], 3).cases[0]
        self.assertIsNone(case.error)
        self.assertEqual(case.sample_errors, 1)
        self.assertTrue(case.passed)

    def test_all_failed_samples_error_the_case(self):
        run = run_scripted([RuntimeError("boom")], 3)
        case = run.cases[0]
        self.assertIn("boom", case.error)
        self.assertFalse(case.passed)
        self.assertEqual(summarize(run)["errors"], 1)


class TestSamplingSummary(unittest.TestCase):
    def test_pass_at_k_metrics_and_flaky_cases(self):
        data = [
            Example(id="stable", input="q1", expected="A"),
            Example(id="flaky", input="q2", expected="B"),
            Example(id="broken", input="q3", expected="C"),
        ]
        replies = {"q1": ["A", "A", "A", "A"], "q2": ["B", "x", "B", "y"], "q3": ["z", "z", "z", "z"]}

        class PerPrompt:
            name, model = "per-prompt", "per-prompt"

            def __init__(self):
                self.seen = {}

            def complete(self, prompt, system=None):
                i = self.seen.get(prompt, 0)
                self.seen[prompt] = i + 1
                return Completion(replies[prompt][i])

        run = run_eval(data, PerPrompt(), samples=4, concurrency=1)
        s = summarize(run)["sampling"]
        self.assertEqual(s["samples"], 4)
        self.assertAlmostEqual(s["pass_at_1"], (1 + 0.5 + 0) / 3)
        self.assertAlmostEqual(s["pass_at_k"], 2 / 3)
        self.assertAlmostEqual(s["pass_all_k"], 1 / 3)
        self.assertAlmostEqual(s["output_agreement"], (1 + 0.5 + 1) / 3)
        self.assertEqual(s["flaky"], [{"id": "flaky", "passed": 2, "samples": 4}])

        md = render_markdown(run)
        self.assertIn("## Consistency (4 samples per case, pass policy: majority)", md)
        self.assertIn("`flaky` (2/4)", md)
        self.assertIn("(passed 0/4 samples)", md)

    def test_agreement_ignores_case_and_whitespace(self):
        run = run_scripted(["Paris", " paris ", "PARIS"], 3)
        self.assertEqual(summarize(run)["sampling"]["output_agreement"], 1.0)

    def test_save_load_roundtrip_keeps_samples(self):
        run = run_scripted(["Paris", "nope", "Paris"], 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.json")
            save_run(run, path)
            loaded = load_run(path)
        self.assertEqual(loaded.cases[0].sample_passes, [True, False, True])
        self.assertEqual(summarize(loaded)["sampling"], summarize(run)["sampling"])


class TestStochasticMock(unittest.TestCase):
    def test_default_mock_is_deterministic_per_prompt(self):
        mock = MockProvider({"q": "A"}, accuracy=0.5, seed=3)
        self.assertEqual(len({mock.complete("q").text for _ in range(20)}), 1)

    def test_stochastic_mock_varies_but_repeats_across_runs(self):
        def answers():
            mock = MockProvider({"q": "A"}, accuracy=0.5, seed=3, stochastic=True)
            return [mock.complete("q").text for _ in range(20)]

        first = answers()
        self.assertEqual(set(first), {"A", "I don't know."})
        self.assertEqual(first, answers())

    def test_sampled_run_is_repeatable_under_concurrency(self):
        data = [Example(id=str(i), input=f"q{i}", expected=f"a{i}") for i in range(10)]

        def sampled():
            mock = MockProvider({e.input: e.expected for e in data}, accuracy=0.6, stochastic=True)
            return summarize(run_eval(data, mock, samples=5, concurrency=8))["sampling"]

        self.assertEqual(sampled(), sampled())


class TestSamplingCLI(unittest.TestCase):
    def test_cli_samples_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.json")
            args = ["run", "--dataset", SAMPLE, "--provider", "mock", "--metrics", "exact_match",
                    "--mock-accuracy", "0.7", "--mock-stochastic", "--samples", "3",
                    "--pass-policy", "all", "--out", out]
            self.assertEqual(main(args), 0)
            with open(out) as fh:
                data = json.load(fh)
        self.assertEqual(data["config"]["samples"], 3)
        self.assertEqual(data["config"]["pass_policy"], "all")
        self.assertEqual(data["summary"]["sampling"]["samples"], 3)
        self.assertTrue(all(len(c["sample_passes"]) == 3 for c in data["cases"]))

    def test_cli_rejects_zero_samples(self):
        with self.assertRaises(SystemExit):
            main(["run", "--dataset", SAMPLE, "--provider", "mock", "--samples", "0"])


if __name__ == "__main__":
    unittest.main()
