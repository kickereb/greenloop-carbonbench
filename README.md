# GreenLoop CarbonBench local pilot

This repository contains a reproducible experiment for local Ollama inference on an Apple Silicon Mac. It was built and smoke-tested on an Apple M5 Pro with 18 CPU cores and 24 GB unified memory.

The experiment tests two claims:

1. A telemetry tool can measure the relative resource use of repeatable local LLM requests.
2. A simple energy formula can transfer from one open-model family to model families that it did not see during fitting.

The experiment does **not** prove the energy or carbon footprint of a closed cloud model. `powermetrics` reports estimated CPU, GPU, and ANE power for the whole SoC. It does not report wall-socket power. It also omits some device components. The tool keeps this distinction in every output.

The detailed preregistration is in [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md). The field definitions and energy method are in [docs/TELEMETRY_AND_CARBON.md](docs/TELEMETRY_AND_CARBON.md). The completed local smoke test is in [docs/VALIDATION_LOG.md](docs/VALIDATION_LOG.md).

The first instrumented 60-request shakedown is documented in [docs/results/SHAKEDOWN_RUN_1.md](docs/results/SHAKEDOWN_RUN_1.md). It completed without request or telemetry failures, but did not pass the preregistered repeatability and memory-pressure gates; the full pilot has therefore not started.

## What the tool does

- Downloads a fixed 100-prompt suite from pinned GSM8K and SQuAD revisions.
- Generates controlled prefill and decode probes.
- Verifies every downloaded file with SHA-256.
- Pulls six Ollama models from four model families.
- Uses streamed Ollama responses to record time to first token.
- Records input tokens, cached input tokens, output tokens, and all Ollama phase durations.
- Records model digests and model metadata.
- Collects continuous CPU, GPU, ANE, and thermal data from `powermetrics`.
- Preserves the raw NUL-delimited plist power stream.
- Integrates power only across the exact overlap with each request.
- Uses loaded-model baselines before and after each model block.
- Keeps gross and baseline-corrected energy estimates.
- Grades deterministic prompts without another LLM.
- Stores all results in a resumable SQLite database.
- Checks matrix completeness, failures, duplicate cells, cache hits, reloads, and power coverage.
- Fits a transparent formula on one family and tests it on locked prompts from held-out families.
- Produces CSV, JSON, Markdown, and an SVG quality-energy plot.

The implementation uses only the Python standard library. Python 3.9 or later is sufficient.

## Repository map

```text
configs/
  smoke.json              Fast API and storage check for the installed Gemma 4 model
  smoke_truncation.json   The exact 64-token guardrail test recorded in the validation log
  shakedown.json          60-request repeatability gate before the core run
  pilot.json              Six-model, 100-prompt, three-round experiment
docs/
  EXPERIMENT_PROTOCOL.md  Hypotheses, controls, statistics, and decision rules
  TELEMETRY_AND_CARBON.md Exact energy method and claim limits
  VALIDATION_LOG.md       Tests already run on this M5 Pro
prompts/
  smoke.jsonl             Four deterministic smoke prompts
  pilot_100.jsonl         Frozen generated pilot suite
  THIRD_PARTY_NOTICES.md  Dataset attribution and licence notices
src/carbonbench/          Telemetry, runner, storage, grading, and analysis code
tests/                    Dependency-free unit and integration tests
```

## Quick start

Run these commands from this repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check the host and Ollama:

```bash
carbonbench doctor
```

Rebuild the exact prompt suite if needed:

```bash
carbonbench fetch-prompts \
  --output prompts/pilot_100.jsonl \
  --raw-directory data/raw
```

The command pins the upstream source commits. It stops if a checksum differs. The generated file has 100 prompts:

| Stratum | Count | Main purpose |
|---|---:|---|
| GSM8K | 35 | Deterministic mathematical quality |
| SQuAD v1.1 | 45 | Grounded answer quality and input variation |
| Controlled long context | 10 | Prefill scaling at 128–2,048 target words |
| Controlled decode | 10 | Output caps of 32, 64, 128, and 256 tokens |

Every stratum has a frozen development/locked split. The total split is 80/20.

Inspect the exact run count:

```bash
carbonbench plan --config configs/pilot.json
```

The supplied core plan contains:

- 6 models;
- 100 prompts;
- 3 full rounds;
- 1,800 measured requests;
- 36 warm-up requests;
- 18 model blocks;
- 15-second loaded-idle windows before and after every block.

## Model download

Preview the pull commands:

```bash
carbonbench pull --config configs/pilot.json --dry-run
```

Pull the models:

```bash
carbonbench pull --config configs/pilot.json
```

The default matrix is:

| Model | Nominal parameters | Role |
|---|---:|---|
| `qwen2.5:3b` | 3.09B | Formula fit |
| `qwen2.5:7b` | 7.62B | Formula fit |
| `qwen2.5:14b` | 14.8B | Formula fit |
| `llama3.2:3b` | 3.21B | Held-out family |
| `gemma3:12b` | 12.2B | Held-out family |
| `granite3.1-moe:3b` | 3.3B total | Descriptive MoE challenge |

The current Q4 model files total about 28 GB on disk. Only one model is used at a time. The largest core file is about 9 GB. The runner records the resolved digest because a mutable model tag is not enough for reproduction.

Do not add a 24B or larger stress model to the core matrix until you confirm that it does not use swap. Model weights are not the only memory use. Ollama, KV cache, and macOS share the 24 GB memory pool.

## Run without power telemetry

Use this first. It checks Ollama, grading, SQLite, resume behavior, and reporting.

```bash
carbonbench run \
  --config configs/smoke.json \
  --database results/smoke-128.sqlite \
  --energy none

carbonbench verify --database results/smoke-128.sqlite

carbonbench analyze \
  --database results/smoke-128.sqlite \
  --output results/smoke-128-report
```

An interrupted run is safe. Run the same command again. Successful cells have deterministic keys and are skipped.

The validation log also records a deliberate 64-token truncation test. Its quality gate fails by design. The ordinary `smoke.json` uses a 128-token cap.

Use `--max-runs 5` for a short shakedown. Use `--model qwen2.5:3b` to isolate one configured model. A partial matrix will correctly fail the completeness check.

## Run with Apple power telemetry

`powermetrics` needs elevated permission. Give permission only to the collector. Do not run the Python process or Ollama as root.

```bash
caffeinate -dimsu carbonbench run \
  --config configs/pilot.json \
  --database results/pilot.sqlite
```

CarbonBench runs `sudo -v` on the controlling terminal and asks for your normal macOS login password before starting `sudo -n /usr/bin/powermetrics`. Nothing is displayed while you type the password. CarbonBench never reads or stores it, and the Python process and Ollama remain unprivileged.

For a first instrumented check, copy `configs/pilot.json`, keep one small model, use 6–10 prompts, and use 10 repeats. Do not start the full matrix until the within-cell coefficient of variation is acceptable.

The repository includes the preregistered 60-request gate:

```bash
carbonbench plan --config configs/shakedown.json
caffeinate -dimsu carbonbench run \
  --config configs/shakedown.json \
  --database results/shakedown.sqlite
carbonbench verify --database results/shakedown.sqlite
carbonbench analyze \
  --database results/shakedown.sqlite \
  --output results/shakedown-report
```

It runs three representative models against one long-prefill probe and one long-decode probe for ten rounds.

Keep these conditions fixed:

- Use native Ollama on macOS, not Docker Desktop.
- Run one request at a time.
- Close GPU-heavy and CPU-heavy applications.
- Stop model downloads and software updates.
- Keep the display and external devices in one fixed state.
- Use one power source and one macOS power mode.
- Do not measure wall power while the battery is charging.
- Check memory pressure and swap before every large-model block.
- Let the Mac cool between blocks if thermal pressure changes.

## Verify and analyze

```bash
carbonbench verify --database results/pilot.sqlite

carbonbench analyze \
  --database results/pilot.sqlite \
  --output results/pilot-report
```

Expected report files:

```text
results/pilot-report/
  report.md
  summary.json
  model_summary.csv
  runs.csv
  quality_energy.svg
```

The raw power stream is next to the SQLite file. Inspect it independently:

```bash
carbonbench inspect-power results/pilot-<session-id>.powermetrics.pliststream
```

## Result interpretation

Use this order:

1. Check request and telemetry integrity.
2. Check repeatability.
3. Check quality.
4. Compare energy only after the first three checks pass.
5. Inspect each held-out family separately.
6. Treat any carbon value as a scenario calculation unless wall power was measured.

The main local success rule is:

> On one frozen Apple platform, telemetry is repeatable; a phase-aware formula predicts two unseen open-model families inside the declared error; and a lower-energy model keeps quality inside a five-percentage-point non-inferiority margin.

Do not state that CarbonBench measured the carbon emissions of ChatGPT or another cloud service. That claim needs server hardware, batching, PUE, location, grid intensity, hidden-token, and whole-system power data that the local test does not contain.

## Test the code

```bash
python -m unittest discover -s tests -v
```

The suite tests plist parsing, arbitrary stream boundaries, exact power-window overlap, negative baseline corrections, graders, config validation, streamed Ollama usage, database completeness, and held-out estimator fitting.

## Publish to GitHub

The project is already a local Git repository on branch `main`. Generated results, raw telemetry, downloaded source data, virtual environments, and Python build products are excluded by `.gitignore`.

You can create and push a repository directly from Terminal with GitHub CLI; creating it in the GitHub website first is optional. Check the account currently selected for `github.com`:

```bash
gh auth status --hostname github.com
```

Create a private repository under the authenticated account and push the existing `main` branch:

```bash
GH_HOST=github.com gh repo create greenloop-carbonbench \
  --private \
  --source=. \
  --remote=origin \
  --push
```

Use `--public` instead of `--private` only if you intend to publish the code and bundled prompt material publicly.

If you prefer to create the empty repository on GitHub first, do not initialize it with a README, licence, or `.gitignore`, because those files already exist locally. Then connect and push it, replacing `YOUR_GITHUB_USERNAME` with your account name:

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/greenloop-carbonbench.git
git push -u origin main
```

Before every later push:

```bash
git status
git add <files-you-changed>
git commit -m "Describe the change"
git push
```

## Primary references

- [Ollama API introduction](https://docs.ollama.com/api/introduction)
- [Ollama generate endpoint](https://docs.ollama.com/api/generate)
- [Ollama usage fields](https://docs.ollama.com/api/usage)
- [Ollama context length](https://docs.ollama.com/context-length)
- [Ollama Apple Metal support](https://docs.ollama.com/gpu#metal-apple-gpus)
- [powermetrics manual](https://manp.gs/mac/1/powermetrics)
- [GSM8K source](https://github.com/openai/grade-school-math)
- [SQuAD source](https://github.com/rajpurkar/SQuAD-explorer)
