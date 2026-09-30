# Instrumented shakedown 2

**Date:** 2026-09-30  
**Host:** Apple M5 Pro, 18 CPU cores, 24 GB unified memory  
**Session:** `21f7fd66-9ce2-4292-8340-8f7381dc2303`  
**Ollama:** 0.35.0  
**Configuration hash:** `2b23c378c90e7661a834c2742a6fcc0ed18a16f56e3fbf56efa40e5e76ec1cad`

## Outcome

The Mac was restarted before this run and began with zero swap. All 60 physical
requests completed successfully, but only 20 were accepted into the analysis.
CarbonBench correctly rejected 40 Qwen attempts after Ollama reported that it
had reused prompt tokens.

| Model | Attempts | Accepted | Prompt-cache hits | Cached-token range |
|---|---:|---:|---:|---:|
| `gemma3:12b` | 20 | 20 | 0 | 0 |
| `qwen2.5:3b` | 20 | 0 | 20 | 24–28 |
| `qwen2.5:14b` | 20 | 0 | 20 | 24–28 |

The matrix is therefore incomplete and cannot be used for repeatability or
cross-model energy conclusions. `controlled-decode-000` also reached its
intentional output cap in 20 rejected Qwen attempts and 10 accepted Gemma
attempts; this is expected for that probe and is not a quality-task failure.

## Memory result

Despite beginning at zero swap, the system reached approximately 5.29 GiB of
swap during the run, and the pageout counter increased in 29 of 30 model
blocks. This is stronger evidence than shakedown 1 that the memory-pressure
failure is created during the workload rather than merely inherited from the
pre-run desktop state.

The observed behavior matches Ollama's llama-server historical prompt cache.
Ollama documents `prompt_eval_cached_count` as the number of input tokens read
from cache. A current Ollama issue documents an 8 GiB default RAM budget and
the `LLAMA_ARG_CACHE_RAM=0` workaround:

- <https://docs.ollama.com/api/usage>
- <https://github.com/ollama/ollama/issues/18264>

## Root cause and protocol correction

Two distinct caches matter:

1. The active inference slot retains the most recent request's KV state.
2. The historical RAM cache can restore older prompt states after that slot has
   been replaced.

Disabling the historical cache alone does not eliminate the fixed chat-template
prefix in the active slot. CarbonBench now handles both layers without changing
the measured model prompt:

- a short raw request replaces the active slot immediately before every
  measured request;
- the real request still uses Ollama's normal model-specific chat template;
- an unmeasured preflight checks that the historical cache cannot restore the
  earlier template prefix;
- the run stops before `powermetrics` starts if any cached prompt tokens remain;
- verification reports flags across all successful attempts, including
  attempts rejected from the primary matrix.

This behavior was tested locally against two Ollama 0.35 servers. The ordinary
server restored 24 Qwen template tokens after the slot scrub; a server started
with `LLAMA_ARG_CACHE_RAM=0` restored zero.

## Local artifacts

The database and raw power stream remain local and are excluded by
`.gitignore`:

```text
results/shakedown-2.sqlite
results/shakedown-2-21f7fd66-9ce2-4292-8340-8f7381dc2303.powermetrics.pliststream
results/shakedown-2-report/
```

Regenerate the corrected verification output with:

```bash
carbonbench verify --database results/shakedown-2.sqlite
```

The corrected verifier reports 40 cache-hit attempts, 20 accepted cells, and
29 blocks with pageouts.

## Next run

Restart the Mac again to clear the swap created by this run. Before reopening
Ollama, set `LLAMA_ARG_CACHE_RAM=0` as described in the README. Use a new
database (`results/shakedown-3.sqlite`); shakedown 2 is retained as an invalid
but diagnostically useful run.
