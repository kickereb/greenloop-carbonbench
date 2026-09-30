# Local validation log

**Date:** 28 September 2026  
**Host:** Apple M5 Pro, 18 CPU cores, 24 GB unified memory  
**macOS:** 26.5.2, build 25F84  
**Ollama:** 0.33.1  
**Installed model:** `gemma4:26b`, Q4_K_M, 17,987,581,215 bytes  
**Model digest:** `5571076f3d70050487b26b341705799e0ab29b808164f90d20d4cf84f699d251`

## Code tests

Command:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Result:

```text
20 tests passed
```

The tests cover:

- plist document parsing;
- NUL-delimited power streams;
- missing power rails;
- invalid samples;
- actual interval overlap;
- negative baseline correction;
- prompt graders;
- config validation;
- streamed Ollama output and usage fields;
- complete synthetic result matrices;
- held-out-family estimator fitting;
- report creation.

## Frozen prompt suite

The downloader fetched and verified:

- GSM8K test JSONL at commit `3101c7d5072418e28b9008a6636bde82a006892c`;
- SQuAD v1.1 development JSON at commit `eee5fdbf62f8613a7812b03419e6b29617b74fd1`.

Generated prompt file:

```text
prompts/pilot_100.jsonl
```

SHA-256:

```text
74f2f26bfb839b8d001147ac51081e2623cc20bec9242070a28950ecb090e966
```

Checks:

- 100 rows;
- 100 unique IDs;
- 80 development prompts;
- 20 locked prompts;
- locked prompts present inside every task category;
- controlled input size no larger than 2,048 target words at a 4,096-token context;
- controlled output caps at 32, 64, 128, and 256 tokens.

## Live Ollama smoke experiment

Command:

```bash
PYTHONPATH=src python3 -m carbonbench run \
  --config configs/smoke_truncation.json \
  --database results/smoke.sqlite
```

This run did not use privileged power telemetry. It tested the full API-to-database-to-report path.

| Prompt | Input tokens | Output tokens | Wall time | TTFT | Stop reason | Quality |
|---|---:|---:|---:|---:|---|---:|
| Evidence abstention | 87 | 10 | 24.50 s | 15.81 s | stop | 1.0 |
| Grounded QA | 94 | 22 | 38.87 s | 22.10 s | stop | 1.0 |
| Arithmetic | 87 | 64 | 88.04 s | 27.71 s | length | 0.0 |
| Controlled retrieval | 95 | 12 | 29.56 s | 23.37 s | stop | 1.0 |

The arithmetic response reached the configured 64-token cap before it emitted the required final answer. The tool correctly stored:

```text
done_reason = length
flags = ["output_truncated"]
```

This is a useful validation. The tool did not treat a shorter, incomplete response as a quality success.

Database verification result:

- expected measured requests: 4;
- successful requests: 4;
- failed requests: 0;
- duplicate cells: 0;
- unexpected measured model loads: 0;
- cache hits reported by Ollama: 0;
- complete matrix: yes.

The infrastructure checks passed. The combined verification gate failed as intended because `no_truncated_quality_requests` was false. This confirms that a completed HTTP request is not automatically treated as a valid benchmark result.

The generated report is in `results/smoke-report/`. The `results/` directory is ignored by Git.

## Checks not yet run

These checks need user interaction or more time and were not claimed as complete:

- live `powermetrics` acquisition, because it needs `sudo -v`;
- sampler-overhead test at 100, 250, 500, and 1,000 ms;
- 60-request repeatability shakedown;
- six-model downloads;
- 1,800-request core experiment;
- external wall-meter calibration;
- time-aligned South Australian grid-carbon calculation.

Run these in the order given in the protocol.
