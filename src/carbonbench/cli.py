from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .analysis import AnalysisError, analyze_database, verify_database
from .config import ConfigError, load_config
from .datasets import DatasetError, build_prompt_suite
from .db import StoreError
from .energy import EnergyError, parse_plist_stream
from .ollama import OllamaClient, OllamaError, pull_models
from .prompts import PromptError, load_prompts
from .runner import RunError, host_snapshot, run_experiment
from .util import file_sha256


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def _sudo_cached() -> Optional[bool]:
    if sys.platform != "darwin" or shutil.which("sudo") is None:
        return None
    result = subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False)
    return result.returncode == 0


def command_doctor(args: argparse.Namespace) -> int:
    client = OllamaClient(args.ollama_url, timeout=10)
    snapshot = host_snapshot(client)
    snapshot["ollama_executable"] = shutil.which("ollama")
    snapshot["powermetrics_executable"] = shutil.which("powermetrics")
    snapshot["powermetrics_sudo_cached"] = _sudo_cached()
    disk = shutil.disk_usage(Path.cwd())
    snapshot["workspace_disk_free_bytes"] = disk.free
    warnings = []
    if snapshot.get("ollama_error"):
        warnings.append("Ollama API is unavailable")
    if sys.platform == "darwin" and not snapshot["powermetrics_executable"]:
        warnings.append("powermetrics is unavailable")
    if snapshot["powermetrics_sudo_cached"] is False:
        warnings.append("Run `sudo -v` immediately before a powermetrics experiment")
    env = snapshot["environment"]
    if env.get("OLLAMA_NUM_PARALLEL") not in (None, "1"):
        warnings.append("Set OLLAMA_NUM_PARALLEL=1 before starting Ollama for serial benchmarking")
    power_source = snapshot.get("power_source") or ""
    if re.search(r"\bcharging\b", power_source.lower()):
        warnings.append("Battery is charging; wall-power validation would include charging energy")
    elif "battery power" in power_source.lower():
        warnings.append("The Mac is on battery; use stable AC power for the multi-hour core experiment")
    if disk.free < 35 * 1024**3:
        warnings.append("Less than 35 GiB is free; the six core model files may not fit safely")
    snapshot["warnings"] = warnings
    _print_json(snapshot)
    return 0 if not snapshot.get("ollama_error") else 2


def command_fetch_prompts(args: argparse.Namespace) -> int:
    manifest = build_prompt_suite(
        output=args.output.resolve(),
        raw_directory=args.raw_directory.resolve(),
        seed=args.seed,
        gsm8k_count=args.gsm8k,
        squad_count=args.squad,
        controlled_count=args.controlled,
    )
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json").resolve()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_json({**manifest, "manifest": str(manifest_path)})
    return 0


def command_pull(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pull_models([model.name for model in config.models], dry_run=args.dry_run)
    return 0


def command_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    prompts = load_prompts(config.prompt_file)
    if config.prompt_ids is not None:
        indexed = {prompt.id: prompt for prompt in prompts}
        missing = [prompt_id for prompt_id in config.prompt_ids if prompt_id not in indexed]
        if missing:
            raise PromptError("Configured prompt_ids are missing: " + ", ".join(missing))
        prompts = [indexed[prompt_id] for prompt_id in config.prompt_ids]
    measured = len(config.models) * len(prompts) * config.repetitions
    blocks = len(config.models) * config.repetitions
    plan = {
        "name": config.name,
        "config_hash": config.config_hash,
        "models": [model.name for model in config.models],
        "model_count": len(config.models),
        "prompt_file": str(config.prompt_file),
        "prompt_file_sha256": file_sha256(config.prompt_file),
        "prompt_count": len(prompts),
        "repetitions": config.repetitions,
        "measured_requests": measured,
        "warmup_requests": blocks * config.warmups_per_model,
        "model_blocks": blocks,
        "baseline_time_minutes": (
            blocks * 2 * config.energy.baseline_seconds / 60
            if config.energy.mode == "powermetrics"
            else 0
        ),
        "energy_mode": config.energy.mode,
        "context_tokens": config.num_ctx,
        "maximum_output_tokens": config.num_predict,
        "temperature": config.temperature,
    }
    _print_json(plan)
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = run_experiment(
        config,
        args.database.resolve(),
        base_url=args.ollama_url,
        energy_override=args.energy,
        max_runs=args.max_runs,
        only_model=args.model,
    )
    _print_json(result)
    return 0 if result["failed"] == 0 else 3


def command_verify(args: argparse.Namespace) -> int:
    result = verify_database(args.database.resolve(), args.config_hash)
    _print_json(result)
    return 0 if result["all_required_checks_pass"] else 4


def command_analyze(args: argparse.Namespace) -> int:
    result = analyze_database(args.database.resolve(), args.output.resolve(), args.config_hash)
    _print_json(
        {
            "report": str((args.output / "report.md").resolve()),
            "summary": str((args.output / "summary.json").resolve()),
            "models": len(result["models"]),
            "estimator_fitted": result["estimator"] is not None,
        }
    )
    return 0


def command_inspect_power(args: argparse.Namespace) -> int:
    data = args.path.read_bytes()
    samples = parse_plist_stream(data)
    invalid = sum(sample.invalid for sample in samples)
    _print_json(
        {
            "path": str(args.path.resolve()),
            "sha256": file_sha256(args.path),
            "bytes": len(data),
            "samples": len(samples),
            "invalid_samples": invalid,
            "duration_seconds": sum(sample.elapsed_ns for sample in samples) / 1_000_000_000,
            "first_sample": (samples[0].__dict__ if samples else None),
            "last_sample": (samples[-1].__dict__ if samples else None),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carbonbench",
        description="Run reproducible local Ollama telemetry experiments on Apple Silicon.",
    )
    parser.add_argument(
        "--ollama-url", default="http://127.0.0.1:11434", help="Ollama API base URL"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect the host, Ollama, and energy prerequisites")
    doctor.set_defaults(func=command_doctor)

    fetch = subparsers.add_parser("fetch-prompts", help="Download and freeze the 100-prompt pilot suite")
    fetch.add_argument("--output", type=Path, default=Path("prompts/pilot_100.jsonl"))
    fetch.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    fetch.add_argument("--seed", type=int, default=20260928)
    fetch.add_argument("--gsm8k", type=int, default=35)
    fetch.add_argument("--squad", type=int, default=45)
    fetch.add_argument("--controlled", type=int, default=20)
    fetch.set_defaults(func=command_fetch_prompts)

    pull = subparsers.add_parser("pull", help="Pull every model in an experiment config")
    pull.add_argument("--config", type=Path, required=True)
    pull.add_argument("--dry-run", action="store_true")
    pull.set_defaults(func=command_pull)

    plan = subparsers.add_parser("plan", help="Validate a config and print its exact run count")
    plan.add_argument("--config", type=Path, required=True)
    plan.set_defaults(func=command_plan)

    run = subparsers.add_parser("run", help="Execute or resume a benchmark")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--database", type=Path, required=True)
    run.add_argument("--energy", choices=("none", "powermetrics"), default=None)
    run.add_argument("--max-runs", type=int, default=None, help="Stop after this many successful measured runs")
    run.add_argument("--model", default=None, help="Run only one configured model")
    run.set_defaults(func=command_run)

    verify = subparsers.add_parser("verify", help="Check matrix completeness and telemetry integrity")
    verify.add_argument("--database", type=Path, required=True)
    verify.add_argument("--config-hash", default=None)
    verify.set_defaults(func=command_verify)

    analyze = subparsers.add_parser("analyze", help="Create CSV, JSON, Markdown, and SVG results")
    analyze.add_argument("--database", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--config-hash", default=None)
    analyze.set_defaults(func=command_analyze)

    inspect_power = subparsers.add_parser(
        "inspect-power", help="Validate and summarize a raw powermetrics plist stream"
    )
    inspect_power.add_argument("path", type=Path)
    inspect_power.set_defaults(func=command_inspect_power)
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (
        AnalysisError,
        ConfigError,
        DatasetError,
        EnergyError,
        OllamaError,
        PromptError,
        RunError,
        StoreError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; completed rows remain safely stored in SQLite.", file=sys.stderr)
        return 130
