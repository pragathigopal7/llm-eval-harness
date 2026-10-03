# llm-eval-harness

A small, **dependency-free** Python harness for evaluating LLM systems. It runs a model over a test set, scores every answer, tracks latency, tokens and cost, and fails CI when a prompt or model change makes things worse.

```
dataset (JSONL) ──► provider (Anthropic / OpenAI-compatible / mock) ──► scorers ──► run.json + report.md
                                                                          │            │
                                     exact_match, token_f1, numeric,  ◄───┘            └──► compare vs baseline
                                     regex, json_valid, LLM-as-judge                         (regression gate)
```

## Features

- **Reference metrics:** `exact_match`, `contains`, `token_f1`, `numeric_match`, `regex_match`, `json_valid`
- **LLM-as-judge:** a second model grades answers 1–5 against a rubric and a reference; output is parsed robustly, and unparseable verdicts count as errors, not passes
- **Per-example metrics:** each test case can choose its own scorers, so factual, math, format and open-ended cases live in one dataset
- **Statistics that hold up on small sets:** pass rate with a 95% Wilson confidence interval, plus a per-category breakdown
- **Ops metrics:** mean/p50/p95 latency, token counts, and cost from your per-million-token prices
- **Regression detection:** compares against a saved baseline, lists newly failing and newly passing cases, and exits non-zero for CI
- **Providers:** Anthropic Messages API, any OpenAI-compatible server (OpenAI, vLLM, Ollama, LM Studio), and a deterministic mock for offline runs
- **Robust runs:** concurrent execution, retries with backoff on 429/5xx, and provider failures recorded per case instead of crashing the run
- **Standard library only:** Python 3.9+, nothing to install

## Quick start

```bash
git clone https://github.com/pragathigopal7/llm-eval-harness.git
cd llm-eval-harness

# Offline demo, no API key needed
python -m evalharness run --dataset data/sample.jsonl --provider mock --mock-accuracy 0.8

# Real model, with an LLM judge for open-ended answers
export ANTHROPIC_API_KEY=...
python -m evalharness run \
  --dataset data/sample.jsonl \
  --provider anthropic --model claude-haiku-4-5 --temperature 0 \
  --judge-model claude-sonnet-4-5 \
  --input-price 1 --output-price 5 \
  --out runs/baseline.json --report reports/baseline.md
```

Optionally, install it as a command with `pip install -e .`, which gives you `evalharness run ...`.

### Local or open models (vLLM, Ollama)

```bash
python -m evalharness run --dataset data/sample.jsonl \
  --provider openai --base-url http://localhost:11434/v1 --model llama3.1
```

## Dataset format

One JSON object per line:

```json
{"id": "math-002", "input": "What is 17 * 24? End your answer with the number.", "expected": "408", "metrics": ["numeric_match"], "metadata": {"category": "math"}}
{"id": "fmt-003", "input": "Give the ISO date for 1 Jan 2025.", "metrics": ["regex_match"], "metadata": {"pattern": "^\\s*2025-01-01\\s*$"}}
{"id": "reason-001", "input": "Explain why the sky is blue.", "expected": "Rayleigh scattering ...", "metrics": ["llm_judge"]}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique, stable id; regressions are tracked by it |
| `input` | yes | Prompt; inserted into `--prompt-template` as `{input}` |
| `expected` | no | Reference answer for reference-based metrics and the judge |
| `metrics` | no | Overrides the run-level `--metrics` for this case |
| `metadata` | no | `category` for breakdowns, `pattern` for `regex_match`, `rubric` for a per-case judge rubric |

A case **passes** when it was scored without errors and every metric reaches `--threshold` (default 0.5). Judge scores of 1–5 are mapped to 0–1, so the default threshold means a judge score of 3 or higher.

## Catching regressions

```bash
# Save a baseline from main
python -m evalharness run --dataset data/sample.jsonl --provider anthropic --model claude-haiku-4-5 \
  --out baselines/main.json

# On a change, compare and gate. This exits 1 if the pass rate drops more than 5 points.
python -m evalharness run --dataset data/sample.jsonl --provider anthropic --model claude-haiku-4-5 \
  --system-file prompts/new_system.txt \
  --baseline baselines/main.json --max-drop 0.05 --fail-on-regression

# Or compare two saved runs
python -m evalharness compare runs/baseline.json runs/candidate.json --fail-on-regression
```

The report shows the pass-rate delta, the delta for each metric, and exactly which case ids started failing.

## CI

`.github/workflows/ci.yml` runs the unit tests and an offline smoke eval on every push. On pull requests it also runs a real-model eval when an `ANTHROPIC_API_KEY` repository secret is set. That eval writes its report to the job summary and gates on `baselines/main.json` if you commit one. You can set the models with the `EVAL_MODEL` and `JUDGE_MODEL` repository variables.

## Python API

```python
from evalharness import AnthropicProvider, LLMJudge, Pricing, load_jsonl, run_eval, summarize, compare_runs

examples = load_jsonl("data/sample.jsonl")
model = AnthropicProvider("claude-haiku-4-5", temperature=0)
judge = LLMJudge(AnthropicProvider("claude-sonnet-4-5", temperature=0))

run = run_eval(examples, model, metrics=["exact_match"], judge=judge,
               pricing=Pricing(input_per_mtok=1, output_per_mtok=5), concurrency=8)
print(summarize(run)["pass_rate"])
```

Any object with `name`, `model` and `complete(prompt, system=None) -> Completion` works as a provider, so you can evaluate a full RAG pipeline or agent, not just a bare model.

## Project layout

```
evalharness/
  models.py     dataclasses: Example, Completion, CaseResult, RunResult
  dataset.py    JSONL loading and validation
  metrics.py    deterministic scorers
  judge.py      LLM-as-judge prompt and verdict parsing
  providers.py  Anthropic, OpenAI-compatible and mock providers (urllib, retries)
  runner.py     concurrent execution and scoring
  report.py     summary stats, Wilson CI, regression comparison, Markdown report
  cli.py        `run`, `compare`, `metrics`
data/sample.jsonl   16 cases across factual, math, format, coding and open-ended
tests/              unittest suite (no network)
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Roadmap

- Pairwise (A/B) judging with position swapping to reduce judge bias
- Repeated sampling per case to measure answer variance
- Judge calibration against a small human-labeled set
- HTML dashboard of run history

## License

MIT
