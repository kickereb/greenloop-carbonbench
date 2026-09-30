from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .util import file_sha256, stable_hash


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    parameters_b: Optional[float]
    active_parameters_b: Optional[float]
    quantization: Optional[str]
    role: str
    reasoning: bool


@dataclass(frozen=True)
class EnergyConfig:
    mode: str
    sample_interval_ms: int
    baseline_seconds: float
    grid_intensity_g_per_kwh: Optional[float]
    minimum_sample_coverage: float


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    raw: Dict[str, Any]
    config_hash: str
    name: str
    prompt_file: Path
    prompt_ids: Optional[List[str]]
    models: List[ModelSpec]
    repetitions: int
    warmups_per_model: int
    seed: int
    temperature: float
    num_predict: int
    num_ctx: int
    keep_alive: str
    request_timeout_seconds: float
    cooldown_seconds: float
    redact_responses: bool
    energy: EnergyConfig


def _number(value: Any, field: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ConfigError(f"{field} must be >= {minimum}")
    return result


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    if value < minimum:
        raise ConfigError(f"{field} must be >= {minimum}")
    return value


def load_config(path: Path) -> ExperimentConfig:
    path = path.resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Config root must be an object")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ConfigError("name is required")

    prompt_value = raw.get("prompt_file")
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        raise ConfigError("prompt_file is required")
    prompt_path = Path(prompt_value)
    if not prompt_path.is_absolute():
        prompt_path = (path.parent / prompt_path).resolve()

    raw_models = raw.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError("models must be a non-empty list")
    models: List[ModelSpec] = []
    names = set()
    for index, item in enumerate(raw_models):
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            raise ConfigError(f"models[{index}] must be a string or object")
        model_name = str(item.get("name", "")).strip()
        if not model_name:
            raise ConfigError(f"models[{index}].name is required")
        if model_name in names:
            raise ConfigError(f"Duplicate model: {model_name}")
        names.add(model_name)
        role = str(item.get("role", "fit"))
        if role not in {"fit", "holdout", "descriptive"}:
            raise ConfigError(f"models[{index}].role must be fit, holdout, or descriptive")

        def optional_float(key: str) -> Optional[float]:
            value = item.get(key)
            return None if value is None else _number(value, f"models[{index}].{key}", 0.0)

        parameters = optional_float("parameters_b")
        active = (
            optional_float("active_parameters_b")
            if "active_parameters_b" in item
            else parameters
        )
        models.append(
            ModelSpec(
                name=model_name,
                family=str(item.get("family", model_name.split(":", 1)[0])),
                parameters_b=parameters,
                active_parameters_b=active,
                quantization=(str(item["quantization"]) if item.get("quantization") else None),
                role=role,
                reasoning=bool(item.get("reasoning", False)),
            )
        )

    energy_raw = raw.get("energy", {})
    if not isinstance(energy_raw, dict):
        raise ConfigError("energy must be an object")
    mode = str(energy_raw.get("mode", "none"))
    if mode not in {"none", "powermetrics"}:
        raise ConfigError("energy.mode must be none or powermetrics")
    grid_factor = energy_raw.get("grid_intensity_g_per_kwh")
    if grid_factor is not None:
        grid_factor = _number(grid_factor, "energy.grid_intensity_g_per_kwh", 0.0)
    energy = EnergyConfig(
        mode=mode,
        sample_interval_ms=_integer(energy_raw.get("sample_interval_ms", 250), "energy.sample_interval_ms", 100),
        baseline_seconds=_number(energy_raw.get("baseline_seconds", 15), "energy.baseline_seconds", 0.0),
        grid_intensity_g_per_kwh=grid_factor,
        minimum_sample_coverage=_number(
            energy_raw.get("minimum_sample_coverage", 0.60),
            "energy.minimum_sample_coverage",
            0.0,
        ),
    )
    if energy.minimum_sample_coverage > 1:
        raise ConfigError("energy.minimum_sample_coverage must be <= 1")

    prompt_ids_raw = raw.get("prompt_ids")
    prompt_ids: Optional[List[str]] = None
    if prompt_ids_raw is not None:
        if not isinstance(prompt_ids_raw, list) or not prompt_ids_raw:
            raise ConfigError("prompt_ids must be a non-empty list when supplied")
        prompt_ids = [str(value) for value in prompt_ids_raw]
        if any(not value.strip() for value in prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
            raise ConfigError("prompt_ids must contain unique, non-empty IDs")

    prompt_hash = file_sha256(prompt_path) if prompt_path.exists() else None
    return ExperimentConfig(
        source_path=path,
        raw=raw,
        config_hash=stable_hash({"config": raw, "prompt_file_sha256": prompt_hash}),
        name=name,
        prompt_file=prompt_path,
        prompt_ids=prompt_ids,
        models=models,
        repetitions=_integer(raw.get("repetitions", 3), "repetitions", 1),
        warmups_per_model=_integer(raw.get("warmups_per_model", 1), "warmups_per_model", 0),
        seed=_integer(raw.get("seed", 20260928), "seed", 0),
        temperature=_number(raw.get("temperature", 0), "temperature", 0.0),
        num_predict=_integer(raw.get("num_predict", 128), "num_predict", 1),
        num_ctx=_integer(raw.get("num_ctx", 4096), "num_ctx", 256),
        keep_alive=str(raw.get("keep_alive", "30m")),
        request_timeout_seconds=_number(
            raw.get("request_timeout_seconds", 900), "request_timeout_seconds", 1.0
        ),
        cooldown_seconds=_number(raw.get("cooldown_seconds", 2), "cooldown_seconds", 0.0),
        redact_responses=bool(raw.get("redact_responses", False)),
        energy=energy,
    )
