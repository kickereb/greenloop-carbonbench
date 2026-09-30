# Telemetry and carbon method

## 1. Measurement names

Use the exact labels in this table.

| Source | Correct label | Do not call it |
|---|---|---|
| Ollama API | Request token and duration telemetry | Energy |
| `powermetrics` CPU/GPU/ANE | Estimated system-wide SoC component power | Whole-Mac power |
| Integrated `powermetrics` | Estimated SoC component energy | Wall energy |
| Baseline correction | Incremental estimated SoC component energy above loaded idle | Exact query energy |
| External logging meter | Measured AC wall energy | Ollama-only energy |
| Grid factor × wall energy | Operational carbon estimate | Full life-cycle carbon |
| Activity Monitor Energy Impact | Relative diagnostic score | Joules or watts |

Apple states that `powermetrics` average power is estimated, may be inaccurate, and should not be used for comparisons between devices. Same-device controlled comparisons are the intended use here.

## 2. Why the tool uses plist

The collector runs:

```bash
sudo -n /usr/bin/powermetrics \
  --samplers cpu_power,gpu_power,ane_power,thermal \
  --format plist \
  --buffer-size 0 \
  --sample-rate 500 \
  --poweravg 0 \
  --handle-invalid-values
```

The plist stream is machine-readable. Documents are separated by a NUL byte. The tool saves the original bytes before it derives request-level values.

Text parsing is not used. Human-readable labels can change and do not expose a stable interval field.

## 3. Exact-window integration

`powermetrics` reports an actual `elapsed_ns` for each sample. The requested 500 ms cadence is not used as the integration duration.

For a component with power in milliwatts:

```text
sample_energy_J = power_mW × overlap_seconds / 1000
```

The overlap is the intersection of:

- the actual sample interval; and
- the monotonic request start/end interval.

For the SoC component sum:

```text
E_soc_J = sum(combined_power_mW × overlap_seconds / 1000)
```

If `combined_power` is absent, the tool sums the available CPU, GPU, and ANE values. A missing value stays missing. It is not silently changed to zero.

Invalid samples do not contribute. The run keeps an invalid-sample count and a coverage ratio.

## 4. Coverage

Coverage is:

```text
union of valid sample/request overlap seconds
------------------------------------------------
request wall duration
```

The core minimum is 90%. A request below this value is flagged.

Very short requests can have poor coverage at 500 ms. Use a serialized workload block that lasts at least 30 seconds if a model responds too quickly. Do not claim precise per-request energy from one or two samples.

## 5. Loaded-idle baseline

For each model block:

1. Load and warm the model.
2. Collect 15 seconds of loaded idle.
3. Run the prompt block.
4. Collect 15 seconds of loaded idle.
5. Take the median power in each window.
6. Use the mean of the two medians.

Then:

```text
E_incremental_J = E_gross_J - baseline_mW × covered_seconds / 1000
```

Negative results stay negative. They indicate that the workload is below the resolution/noise of the baseline method. Setting negative results to zero would bias the experiment upward.

Gross energy remains in the database. Baseline correction does not erase it.

## 6. What `combined_power` omits

Treat `combined_power` as CPU + GPU + ANE. It does not establish whole-Mac electricity. It can omit:

- DRAM and parts of the memory fabric;
- display;
- SSD;
- fans;
- external devices;
- battery charging;
- adapter conversion loss.

This is important for LLM inference because Apple Silicon uses unified memory.

## 7. External wall-meter validation

Use a logging AC meter for the primary operational energy subset.

Requirements:

- at least about 1 Hz logging, or an accurate block Wh counter;
- synchronized timestamps;
- battery full and not charging;
- fixed display and peripheral state;
- prompt blocks that last several minutes;
- matched loaded-idle blocks;
- at least ten repeats for calibration conditions.

For each block:

```text
incremental_wall_J = measured_wall_J - loaded_idle_wall_W × block_seconds
```

Compare the SoC proxy with wall energy. Report concordance, bias, and limits of agreement. Do not fit a calibration and test it on the same blocks.

## 8. Battery fallback

Battery data can support a coarse whole-system check over long blocks. It is not suitable for short query estimates.

If no wall meter is available:

- disconnect AC;
- run 10–30 minute blocks;
- record start and end battery state;
- keep display and workload conditions fixed;
- do not mix AC and battery observations.

Do not calculate per-query joules from one instantaneous battery current and voltage reading.

## 9. Carbon conversion

The tool accepts a fixed `grid_intensity_g_per_kwh` only when the user supplies it in the experiment config. The default is `null`.

For a component-energy scenario:

```text
component_energy_carbon_proxy_g = gross_soc_Wh / 1000 × grid_intensity_g_per_kWh
```

This remains a proxy. It is not whole-device carbon.

For a calibrated wall measurement:

```text
operational_gCO2e = wall_Wh / 1000 × grid_intensity_gCO2e_per_kWh
```

Record:

- grid data provider;
- region;
- local timestamp and UTC timestamp;
- marginal or average intensity;
- update interval;
- missing-data rule;
- uncertainty;
- whether renewable certificates or market accounting are included.

## 10. Closed-source extrapolation

The local formula can create a transfer test. It cannot reveal a cloud provider’s true energy.

A closed-source estimate needs uncertainty for:

- hardware type and count;
- quantization;
- active versus total parameters;
- prompt and output tokens;
- hidden reasoning tokens;
- batching and utilization;
- model routing;
- caching;
- datacentre PUE;
- server location;
- grid intensity;
- idle-energy allocation.

Report a range or distribution. Do not report one precise number without a sensitivity analysis.

## 11. SQLite audit data

The database contains four main tables.

### `experiments`

Stores the canonical config JSON, config hash, prompt hash, host snapshot, Ollama version, and installed model list.

### `models`

Stores declared family, parameter counts, quantization, fit/holdout role, and the full Ollama manifest snapshot.

### `prompts`

Stores prompt text, byte and character counts, source, licence, grader, and development/locked metadata.

### `runs`

Stores one row per warm-up or measured request. Important fields include:

- deterministic run key;
- session, round, and schedule position;
- requested and actual output counts;
- API and wall durations;
- time to first token;
- load duration;
- cached prompt tokens;
- stop reason;
- response hash and optional response text;
- quality result;
- gross and incremental component energy;
- sample coverage;
- carbon proxy, when configured;
- machine-readable flags.

### `power_samples`

Stores each parsed plist interval with its actual elapsed time, rail values, validity, and thermal pressure. The raw stream remains the source of truth.

### `block_observations`

Stores pre/post model residency, memory pressure, swap use, VM page-out counters, and power-source state. Verification fails the page-out guard when the counter rises inside a measured model block.

## 12. Privacy

The tool is local. It sends prompts only to the configured Ollama URL.

For public benchmark prompts, storing response text is useful for audit. For private prompts, set:

```json
"redact_responses": true
```

The tool then keeps the response SHA-256 and metrics, not the response text. Prompt text is still stored in the `prompts` table. A later browser plugin should default to local aggregation and avoid storing user content.

## 13. Primary technical references

- [Ollama usage fields](https://docs.ollama.com/api/usage)
- [Ollama generate API](https://docs.ollama.com/api/generate)
- [Ollama model details API](https://docs.ollama.com/api-reference/show-model-details)
- [Ollama keep-alive behavior](https://docs.ollama.com/faq#how-can-i-preload-a-model-into-ollama-to-get-faster-response-times)
- [Ollama concurrency behavior](https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests)
- [powermetrics manual](https://manp.gs/mac/1/powermetrics)
- [Apple energy-efficiency guide](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/MonitoringEnergyUsage.html)
