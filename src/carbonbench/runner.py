from __future__ import annotations

import hashlib
import os
import platform
import random
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ExperimentConfig, ModelSpec
from .db import Store
from .energy import EnergyError, NullCollector, PowerMetricsCollector, summarize_window
from .ollama import OllamaClient, OllamaError
from .prompts import Prompt, grade_response, load_prompts
from .util import canonical_json, stable_hash, utc_now


class RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunOutcome:
    success: bool
    accepted: bool
    primary_eligible: bool


MAX_PREFLIGHT_SWAP_BYTES = 1024**3


def swap_used_bytes(value: Optional[str]) -> Optional[int]:
    """Parse macOS `sysctl vm.swapusage` output into used bytes."""
    if not value:
        return None
    match = re.search(r"\bused\s*=\s*([0-9.]+)([KMGT])", value, re.IGNORECASE)
    if not match:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[match.group(2).upper()]
    return int(float(match.group(1)) * scale)


def _command_output(command: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def host_snapshot(client: Optional[OllamaClient] = None) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "processor": platform.processor(),
        "environment": {
            key: os.environ.get(key)
            for key in ("OLLAMA_CONTEXT_LENGTH", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS")
        },
    }
    if sys.platform == "darwin":
        snapshot["sw_vers"] = _command_output(["sw_vers"])
        snapshot["hardware_model"] = _command_output(["sysctl", "-n", "hw.model"])
        snapshot["cpu_brand"] = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        snapshot["cpu_cores"] = _command_output(["sysctl", "-n", "hw.ncpu"])
        snapshot["memory_bytes"] = _command_output(["sysctl", "-n", "hw.memsize"])
        snapshot["swapusage"] = _command_output(["sysctl", "vm.swapusage"])
        snapshot["swap_used_bytes"] = swap_used_bytes(snapshot["swapusage"])
        snapshot["power_source"] = _command_output(["pmset", "-g", "batt"])
        snapshot["power_settings"] = _command_output(["pmset", "-g", "custom"])
    if client is not None:
        try:
            snapshot["ollama_version"] = client.version()
            snapshot["ollama_tags"] = client.tags()
        except OllamaError as exc:
            snapshot["ollama_error"] = str(exc)
    return snapshot


def block_snapshot(client: OllamaClient) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"ollama_ps": client.ps()}
    if sys.platform == "darwin":
        snapshot["swapusage"] = _command_output(["sysctl", "vm.swapusage"])
        snapshot["vm_stat"] = _command_output(["vm_stat"])
        snapshot["memory_pressure"] = _command_output(["memory_pressure", "-Q"])
        snapshot["power_source"] = _command_output(["pmset", "-g", "batt"])
    return snapshot


def _installed_names(tags: List[Dict[str, Any]]) -> set:
    names = set()
    for tag in tags:
        for key in ("name", "model"):
            if tag.get(key):
                names.add(str(tag[key]))
    return names


def _manifest(client: OllamaClient, model: ModelSpec, tags: List[Dict[str, Any]]) -> Dict[str, Any]:
    matching = [tag for tag in tags if model.name in {str(tag.get("name")), str(tag.get("model"))}]
    show = client.show(model.name)
    return {"tag": matching[0] if matching else None, "show": show}


def _model_orders(models: List[ModelSpec], repetitions: int, seed: int) -> List[List[ModelSpec]]:
    base = list(models)
    random.Random(seed).shuffle(base)
    orders = []
    for round_index in range(repetitions):
        offset = round_index % len(base)
        rotated = base[offset:] + base[:offset]
        # Reverse alternate rounds to reduce monotone early/late position confounding.
        if round_index % 2:
            rotated = list(reversed(rotated))
        orders.append(rotated)
    return orders


def _prompt_order(prompts: List[Prompt], seed: int, round_index: int, model_name: str) -> List[Prompt]:
    ordered = list(prompts)
    model_seed = int(stable_hash(model_name)[:8], 16)
    random.Random(seed + round_index * 104729 + model_seed).shuffle(ordered)
    return ordered


def _cell_key(
    config_hash: str,
    model_name: str,
    prompt_id: str,
    round_index: int,
    label: str = "measured",
) -> str:
    return stable_hash(f"{config_hash}|{model_name}|{prompt_id}|{round_index}|{label}")


WARMUP_PROMPT = Prompt(
    id="__carbonbench_warmup__",
    category="warmup",
    prompt=(
        "This is an unmeasured warm-up. Reply with exactly one line containing FINAL: READY. "
        "Do not add any other text."
    ),
    source="GreenLoop CarbonBench",
    license="CC0-1.0",
    grader={"type": "exact", "answers": ["READY"]},
    metadata={"excluded_from_primary_analysis": True},
)


def run_experiment(
    config: ExperimentConfig,
    db_path: Path,
    *,
    base_url: str = "http://127.0.0.1:11434",
    energy_override: Optional[str] = None,
    max_runs: Optional[int] = None,
    only_model: Optional[str] = None,
) -> Dict[str, Any]:
    prompts = load_prompts(config.prompt_file)
    if config.prompt_ids is not None:
        indexed = {prompt.id: prompt for prompt in prompts}
        unknown = [prompt_id for prompt_id in config.prompt_ids if prompt_id not in indexed]
        if unknown:
            raise RunError("Configured prompt_ids are missing: " + ", ".join(unknown))
        prompts = [indexed[prompt_id] for prompt_id in config.prompt_ids]
    models = [model for model in config.models if only_model is None or model.name == only_model]
    if not models:
        raise RunError(f"No configured model matches {only_model!r}")
    client = OllamaClient(base_url=base_url, timeout=config.request_timeout_seconds)
    try:
        tags = client.tags()
    except OllamaError as exc:
        raise RunError(f"Ollama is unavailable at {base_url}: {exc}") from exc
    installed = _installed_names(tags)
    missing = [model.name for model in models if model.name not in installed]
    if missing:
        raise RunError("Models are not installed: " + ", ".join(missing) + ". Run the pull command first.")

    energy_mode = energy_override or config.energy.mode
    if energy_mode not in {"none", "powermetrics"}:
        raise RunError("energy mode must be none or powermetrics")
    if energy_override is not None and energy_override != config.energy.mode:
        raise RunError(
            "An energy-mode override would change the protocol without changing its identity. "
            "Copy the config and change energy.mode instead."
        )
    snapshot = host_snapshot(client)
    used_swap = snapshot.get("swap_used_bytes")
    if (
        energy_mode == "powermetrics"
        and isinstance(used_swap, int)
        and used_swap > MAX_PREFLIGHT_SWAP_BYTES
    ):
        raise RunError(
            f"Preflight failed: macOS is using {used_swap / 1024**3:.2f} GiB of swap "
            f"(limit {MAX_PREFLIGHT_SWAP_BYTES / 1024**3:.0f} GiB). Restart the Mac, "
            "close memory-intensive applications, and run `carbonbench doctor` before retrying."
        )
    configured_names = {model.name for model in config.models}
    initially_loaded = client.ps()
    unrelated_loaded = [
        str(item.get("name") or item.get("model"))
        for item in initially_loaded
        if str(item.get("name") or item.get("model")) not in configured_names
    ]
    if unrelated_loaded:
        raise RunError(
            "Unrelated Ollama models are resident: "
            + ", ".join(unrelated_loaded)
            + ". Stop them before benchmarking."
        )
    for item in initially_loaded:
        loaded_name = str(item.get("name") or item.get("model"))
        if loaded_name:
            client.unload(loaded_name)
    session_id = str(uuid.uuid4())
    raw_power_path = db_path.parent / f"{db_path.stem}-{session_id}.powermetrics.pliststream"
    collector = (
        PowerMetricsCollector(config.energy.sample_interval_ms, raw_power_path)
        if energy_mode == "powermetrics"
        else NullCollector()
    )
    store = Store(db_path)
    store.register_experiment(config, snapshot)
    store.register_models(config.config_hash, config.models)
    store.register_prompts(config.config_hash, prompts + [WARMUP_PROMPT])
    for model in models:
        store.update_model_manifest(config.config_hash, model.name, _manifest(client, model, tags))

    measured_completed = 0
    failed = 0
    skipped = 0
    flagged = 0
    retryable_attempts = 0
    schedule_index = 0
    recorded_baselines: List[float] = []
    collector_started = False
    try:
        collector.start()
        collector_started = True
        orders = _model_orders(models, config.repetitions, config.seed)
        stop_requested = False
        for round_index, model_order in enumerate(orders):
            if stop_requested:
                break
            for model in model_order:
                if stop_requested:
                    break
                remaining = [
                    prompt
                    for prompt in _prompt_order(prompts, config.seed, round_index, model.name)
                    if not store.has_accepted(
                        _cell_key(config.config_hash, model.name, prompt.id, round_index)
                    )
                ]
                if not remaining:
                    skipped += len(prompts)
                    continue
                print(
                    f"Round {round_index + 1}/{config.repetitions}: {model.name} "
                    f"({len(remaining)} pending prompts)",
                    flush=True,
                )

                # Warm-ups are intentionally not part of primary analysis. They also load the model.
                for warmup_index in range(config.warmups_per_model):
                    warmup_outcome = _execute_one(
                        store,
                        collector,
                        client,
                        config,
                        session_id,
                        model,
                        WARMUP_PROMPT,
                        round_index,
                        schedule_index,
                        None,
                        energy_mode,
                        is_warmup=True,
                        warmup_index=warmup_index,
                    )
                    if not warmup_outcome.success:
                        raise RunError(f"Warm-up failed for {model.name}; measured block was not started")
                    schedule_index += 1

                loaded_models = client.ps()
                loaded_match = [
                    item
                    for item in loaded_models
                    if model.name in {str(item.get("name")), str(item.get("model"))}
                ]
                if len(loaded_models) != 1 or not loaded_match:
                    names = [str(item.get("name") or item.get("model")) for item in loaded_models]
                    raise RunError(
                        f"Expected only {model.name} after warm-up; resident models: {names}"
                    )
                block_manifest = _manifest(client, model, tags)
                block_manifest["loaded_state"] = loaded_match[0]
                store.update_model_manifest(config.config_hash, model.name, block_manifest)
                store.insert_block_observation(
                    config.config_hash,
                    session_id,
                    round_index,
                    model.name,
                    "pre",
                    block_snapshot(client),
                )

                pre_baseline: Optional[float] = None
                if energy_mode == "powermetrics" and config.energy.baseline_seconds > 0:
                    print(
                        f"  loaded-idle baseline before block ({config.energy.baseline_seconds:g}s)...",
                        flush=True,
                    )
                    baseline_start = time.monotonic()
                    time.sleep(config.energy.baseline_seconds)
                    baseline_end = time.monotonic()
                    pre_baseline = collector.baseline(baseline_start, baseline_end)
                    if pre_baseline is None:
                        raise RunError("No valid powermetrics samples in the pre-block baseline")
                    recorded_baselines.append(pre_baseline)

                for prompt in remaining:
                    if max_runs is not None and measured_completed >= max_runs:
                        stop_requested = True
                        break
                    outcome = _execute_one(
                        store,
                        collector,
                        client,
                        config,
                        session_id,
                        model,
                        prompt,
                        round_index,
                        schedule_index,
                        pre_baseline,
                        energy_mode,
                        is_warmup=False,
                        warmup_index=0,
                    )
                    schedule_index += 1
                    measured_completed += int(outcome.accepted)
                    failed += int(not outcome.success)
                    flagged += int(outcome.success and not outcome.primary_eligible)
                    retryable_attempts += int(not outcome.accepted)

                if energy_mode == "powermetrics" and config.energy.baseline_seconds > 0:
                    print(
                        f"  loaded-idle baseline after block ({config.energy.baseline_seconds:g}s)...",
                        flush=True,
                    )
                    baseline_start = time.monotonic()
                    time.sleep(config.energy.baseline_seconds)
                    baseline_end = time.monotonic()
                    post_baseline = collector.baseline(baseline_start, baseline_end)
                    if post_baseline is None:
                        raise RunError("No valid powermetrics samples in the post-block baseline")
                    recorded_baselines.append(post_baseline)
                    paired_baseline = (
                        (pre_baseline + post_baseline) / 2.0
                        if pre_baseline is not None
                        else post_baseline
                    )
                    store.update_block_baseline(
                        config.config_hash,
                        session_id,
                        model.name,
                        round_index,
                        paired_baseline,
                    )
                store.insert_block_observation(
                    config.config_hash,
                    session_id,
                    round_index,
                    model.name,
                    "post",
                    block_snapshot(client),
                )
                try:
                    client.unload(model.name)
                    if any(
                        model.name in {str(item.get("name")), str(item.get("model"))}
                        for item in client.ps()
                    ):
                        raise OllamaError(f"{model.name} remained resident after unload")
                except OllamaError as exc:
                    raise RunError(
                        f"Could not establish a clean model boundary after {model.name}: {exc}"
                    ) from exc
                if config.cooldown_seconds and not stop_requested:
                    time.sleep(config.cooldown_seconds)
    except EnergyError as exc:
        raise RunError(str(exc)) from exc
    finally:
        if collector_started:
            collector.stop()
        store.store_power_samples(config.config_hash, session_id, collector.samples())
        store.close()
    return {
        "database": str(db_path),
        "session_id": session_id,
        "config_hash": config.config_hash,
        "completed": measured_completed,
        "failed": failed,
        "flagged": flagged,
        "retryable_attempts": retryable_attempts,
        "skipped": skipped,
        "energy_mode": energy_mode,
        "baseline_soc_mw": (
            sum(recorded_baselines) / len(recorded_baselines) if recorded_baselines else None
        ),
        "raw_powermetrics": str(raw_power_path) if energy_mode == "powermetrics" else None,
    }


def _execute_one(
    store: Store,
    collector: Any,
    client: OllamaClient,
    config: ExperimentConfig,
    session_id: str,
    model: ModelSpec,
    prompt: Prompt,
    round_index: int,
    schedule_index: int,
    baseline_soc_mw: Optional[float],
    energy_mode: str,
    *,
    is_warmup: bool,
    warmup_index: int,
) -> RunOutcome:
    label = f"warmup-{warmup_index}" if is_warmup else "measured"
    cell_key = _cell_key(config.config_hash, model.name, prompt.id, round_index, label)
    attempt_index = store.next_attempt(cell_key)
    run_key = stable_hash(f"{cell_key}|{attempt_index}|{session_id}")
    request_prompt = (
        f"[Measurement nonce {run_key[:16]}; ignore this identifier.]\n\n{prompt.prompt}"
        if not is_warmup
        else prompt.prompt
    )
    seed = config.seed + round_index
    requested_num_predict = int(prompt.metadata.get("num_predict_override", config.num_predict))
    started_at = utc_now()
    start = time.monotonic()
    status = "success"
    error: Optional[str] = None
    result = None
    response = ""
    quality_score: Optional[float] = None
    quality_detail: Optional[str] = None
    try:
        result = client.generate(
            model.name,
            request_prompt,
            temperature=config.temperature,
            seed=seed,
            num_predict=requested_num_predict,
            num_ctx=config.num_ctx,
            keep_alive=config.keep_alive,
            reasoning=model.reasoning,
        )
        response = result.response
        quality_score, quality_detail = grade_response(response, prompt.grader)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    end = time.monotonic()
    ended_at = utc_now()
    # Allow the collector pipe to flush the sample whose interval ends with this request.
    if energy_mode == "powermetrics":
        time.sleep(config.energy.sample_interval_ms / 1000.0 * 1.15)
    window = summarize_window(
        collector.samples(), start, end, config.energy.sample_interval_ms, baseline_soc_mw
    )
    flags = []
    if energy_mode == "powermetrics" and window.sample_coverage < config.energy.minimum_sample_coverage:
        flags.append("low_power_sample_coverage")
    if (
        not is_warmup
        and result
        and result.load_duration_ns
        and result.load_duration_ns > 1_000_000_000
    ):
        flags.append("unexpected_model_load")
    if result and result.prompt_eval_cached_count:
        flags.append("prompt_cache_hit")
    if result and result.done_reason == "length":
        flags.append(
            "expected_output_cap" if prompt.category == "controlled_decode" else "output_truncated"
        )
    if window.invalid_sample_count:
        flags.append("invalid_power_sample")
    retryable_flags = {
        "unexpected_model_load",
        "prompt_cache_hit",
        "low_power_sample_coverage",
        "invalid_power_sample",
    }
    accepted = status == "success" and not any(flag in retryable_flags for flag in flags)
    primary_eligible = accepted and "output_truncated" not in flags
    gross_wh = window.gross_soc_wh
    factor = config.energy.grid_intensity_g_per_kwh
    emissions = gross_wh / 1000.0 * factor if gross_wh is not None and factor is not None else None
    response_sha = hashlib.sha256(response.encode("utf-8")).hexdigest() if response else None
    stored_response = None if config.redact_responses else response
    raw_payload = dict(result.payload) if result else None
    if raw_payload:
        # The returned token-id context can dwarf all useful telemetry and may reconstruct prompt text.
        raw_payload.pop("context", None)
    row = {
        "run_key": run_key,
        "cell_key": cell_key,
        "attempt_index": attempt_index,
        "accepted": int(accepted),
        "primary_eligible": int(primary_eligible),
        "config_hash": config.config_hash,
        "session_id": session_id,
        "schedule_index": schedule_index,
        "round_index": round_index,
        "model_name": model.name,
        "prompt_id": prompt.id,
        "is_warmup": int(is_warmup),
        "seed": seed,
        "requested_num_predict": requested_num_predict,
        "status": status,
        "error": error,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "start_monotonic_s": start,
        "end_monotonic_s": end,
        "wall_duration_s": end - start,
        "first_token_duration_s": result.first_token_duration_s if result else None,
        "request_prompt_sha256": hashlib.sha256(request_prompt.encode("utf-8")).hexdigest(),
        "response_text": stored_response,
        "response_sha256": response_sha,
        "thinking_chars": len(result.thinking) if result else None,
        "done_reason": result.done_reason if result else None,
        "prompt_eval_count": result.prompt_eval_count if result else None,
        "prompt_eval_cached_count": result.prompt_eval_cached_count if result else None,
        "eval_count": result.eval_count if result else None,
        "total_duration_ns": result.total_duration_ns if result else None,
        "load_duration_ns": result.load_duration_ns if result else None,
        "prompt_eval_duration_ns": result.prompt_eval_duration_ns if result else None,
        "eval_duration_ns": result.eval_duration_ns if result else None,
        "quality_score": quality_score,
        "quality_detail": quality_detail,
        "energy_method": energy_mode,
        "power_sample_count": window.sample_count if energy_mode == "powermetrics" else None,
        "power_invalid_sample_count": (
            window.invalid_sample_count if energy_mode == "powermetrics" else None
        ),
        "power_covered_seconds": window.covered_seconds if energy_mode == "powermetrics" else None,
        "power_sample_coverage": window.sample_coverage if energy_mode == "powermetrics" else None,
        "mean_cpu_mw": window.mean_cpu_mw,
        "mean_gpu_mw": window.mean_gpu_mw,
        "mean_ane_mw": window.mean_ane_mw,
        "mean_soc_mw": window.mean_soc_mw,
        "baseline_soc_mw": window.baseline_soc_mw,
        "gross_soc_j": window.gross_soc_j,
        "incremental_soc_j": window.incremental_soc_j,
        "gross_soc_wh": window.gross_soc_wh,
        "incremental_soc_wh": window.incremental_soc_wh,
        "grid_intensity_g_per_kwh": factor,
        "emissions_gco2e_proxy": emissions,
        "raw_response_json": canonical_json(raw_payload) if raw_payload else None,
        "flags_json": canonical_json(flags),
    }
    store.insert_run(row)
    if is_warmup:
        prefix = "warm-up"
    else:
        prefix = f"{prompt.id}"
    if status == "success":
        tokens = (result.prompt_eval_count or 0) + (result.eval_count or 0) if result else 0
        joules = f", {window.gross_soc_j:.2f} J proxy" if window.gross_soc_j is not None else ""
        quality_label = f"{quality_score:.3f}" if quality_score is not None else "ungraded"
        print(f"  {prefix}: {tokens} tokens, {end - start:.2f}s{joules}, quality={quality_label}", flush=True)
    else:
        print(f"  {prefix}: FAILED: {error}", file=sys.stderr, flush=True)
    return RunOutcome(
        success=status == "success",
        accepted=accepted,
        primary_eligible=primary_eligible,
    )
