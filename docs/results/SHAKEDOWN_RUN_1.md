# Instrumented shakedown 1

**Date:** 2026-09-30  
**Host:** Apple M5 Pro, 18 CPU cores, 24 GB unified memory  
**Session:** `a7724a5f-de14-40ba-aa22-7645c6e00ad9`  
**Configuration hash:** `2b23c378c90e7661a834c2742a6fcc0ed18a16f56e3fbf56efa40e5e76ec1cad`

## Outcome

The first instrumented shakedown completed all 60 measured requests without a failed request, retry, duplicate cell, prompt-cache hit, unexpected model load, invalid power sample, low-coverage request, or quality-task truncation. Median power-window coverage was 100%, and every model block had both pre- and post-block observations.

The run did **not** pass the preregistered gate for proceeding to the full 1,800-request pilot:

- median within-cell energy coefficient of variation was 17.6%, above the 10% threshold;
- only 2 of 6 model–prompt cells had coefficient of variation at or below 10%;
- the system-wide pageout counter increased during 28 of 30 model blocks.

The full pilot should not be run until memory pressure and measurement variability are reduced and the shakedown is repeated.

## Preliminary descriptive results

| Model | Role | Runs | Deterministic quality | Mean incremental SoC energy | Median latency | Median decode rate |
|---|---|---:|---:|---:|---:|---:|
| `qwen2.5:3b` | Fit | 20 | 1.000 | 14.13 J | 0.803 s | 108.85 tokens/s |
| `gemma3:12b` | Held-out family | 20 | 1.000 | 86.60 J | 2.954 s | 34.15 tokens/s |
| `qwen2.5:14b` | Fit | 20 | 1.000 | 106.79 J | 3.268 s | 30.28 tokens/s |

These values are descriptive only because the repeatability gate failed. The energy values are baseline-corrected estimates for Apple CPU, GPU, and ANE SoC components. They are not wall-socket electricity measurements or validated carbon emissions.

`controlled-decode-000` is intentionally ungraded; it exists to control output length and exercise decode. The deterministic retrieval probe scored 1.000 in every accepted run.

## Integrity checks

| Check | Result |
|---|---:|
| Complete 60-request matrix | Pass |
| Failed attempts | 0 |
| Duplicate accepted cells | 0 |
| Median power coverage at least 90% | Pass (100%) |
| Unexpected model loads | 0 |
| Prompt-cache hits | 0 |
| Invalid or low-coverage power windows | 0 |
| Truncated quality requests | 0 |
| Complete pre/post block observations | Pass (30/30) |
| No pageouts during model blocks | **Fail (28/30 had increases)** |
| Repeatability hypothesis H1 | **Fail** |

## Local artifacts

The following generated artifacts remain local and are excluded by `.gitignore`:

```text
results/shakedown.sqlite
results/shakedown-a7724a5f-de14-40ba-aa22-7645c6e00ad9.powermetrics.pliststream
results/shakedown-report/
```

The raw `powermetrics` stream is approximately 105 MB. It is deliberately not stored in the Git repository.

Regenerate the verification and compact report from the local database with:

```bash
carbonbench verify --database results/shakedown.sqlite
carbonbench analyze \
  --database results/shakedown.sqlite \
  --output results/shakedown-report
```

## Next run

Before repeating the shakedown:

1. Reboot the Mac to clear accumulated swap and memory pressure.
2. Connect stable AC power and stop unrelated CPU-, GPU-, and memory-intensive applications.
3. Disable downloads, indexing-heavy work, and background model activity during measurement.
4. Confirm that no unrelated Ollama model is resident.
5. Repeat the same frozen configuration in a **new database** and require both the repeatability and pageout gates to pass before the core pilot. Reusing `results/shakedown.sqlite` would resume the completed first matrix and skip every existing cell.

After restarting, use:

```bash
cd /Users/evam/Documents/ChatGPT/Greenloop
source .venv/bin/activate
carbonbench doctor

caffeinate -dimsu carbonbench run \
  --config configs/shakedown.json \
  --database results/shakedown-2.sqlite

carbonbench verify --database results/shakedown-2.sqlite

carbonbench analyze \
  --database results/shakedown-2.sqlite \
  --output results/shakedown-2-report
```
