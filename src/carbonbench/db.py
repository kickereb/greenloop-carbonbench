from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import ExperimentConfig, ModelSpec
from .energy import PowerSample
from .prompts import Prompt
from .util import canonical_json, file_sha256, utc_now


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS experiments (
    config_hash TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    prompt_file TEXT NOT NULL,
    prompt_file_sha256 TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    host_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    config_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    family TEXT NOT NULL,
    parameters_b REAL,
    active_parameters_b REAL,
    quantization TEXT,
    role TEXT NOT NULL,
    reasoning INTEGER NOT NULL,
    manifest_json TEXT,
    PRIMARY KEY (config_hash, name),
    FOREIGN KEY (config_hash) REFERENCES experiments(config_hash)
);

CREATE TABLE IF NOT EXISTS prompts (
    config_hash TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    category TEXT NOT NULL,
    source TEXT NOT NULL,
    license TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_chars INTEGER NOT NULL,
    prompt_bytes INTEGER NOT NULL,
    grader_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (config_hash, prompt_id),
    FOREIGN KEY (config_hash) REFERENCES experiments(config_hash)
);

CREATE TABLE IF NOT EXISTS runs (
    run_key TEXT PRIMARY KEY,
    cell_key TEXT,
    attempt_index INTEGER,
    accepted INTEGER,
    primary_eligible INTEGER,
    config_hash TEXT NOT NULL,
    session_id TEXT NOT NULL,
    schedule_index INTEGER NOT NULL,
    round_index INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    is_warmup INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    requested_num_predict INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    started_at_utc TEXT NOT NULL,
    ended_at_utc TEXT NOT NULL,
    start_monotonic_s REAL NOT NULL,
    end_monotonic_s REAL NOT NULL,
    wall_duration_s REAL NOT NULL,
    first_token_duration_s REAL,
    request_prompt_sha256 TEXT,
    response_text TEXT,
    response_sha256 TEXT,
    thinking_chars INTEGER,
    done_reason TEXT,
    prompt_eval_count INTEGER,
    prompt_eval_cached_count INTEGER,
    eval_count INTEGER,
    total_duration_ns INTEGER,
    load_duration_ns INTEGER,
    prompt_eval_duration_ns INTEGER,
    eval_duration_ns INTEGER,
    quality_score REAL,
    quality_detail TEXT,
    energy_method TEXT NOT NULL,
    power_sample_count INTEGER,
    power_invalid_sample_count INTEGER,
    power_covered_seconds REAL,
    power_sample_coverage REAL,
    mean_cpu_mw REAL,
    mean_gpu_mw REAL,
    mean_ane_mw REAL,
    mean_soc_mw REAL,
    baseline_soc_mw REAL,
    gross_soc_j REAL,
    incremental_soc_j REAL,
    gross_soc_wh REAL,
    incremental_soc_wh REAL,
    grid_intensity_g_per_kwh REAL,
    emissions_gco2e_proxy REAL,
    raw_response_json TEXT,
    flags_json TEXT NOT NULL,
    FOREIGN KEY (config_hash, model_name) REFERENCES models(config_hash, name),
    FOREIGN KEY (config_hash, prompt_id) REFERENCES prompts(config_hash, prompt_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_config_model ON runs(config_hash, model_name);
CREATE INDEX IF NOT EXISTS idx_runs_config_prompt ON runs(config_hash, prompt_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(config_hash, status, is_warmup);

CREATE TABLE IF NOT EXISTS power_samples (
    config_hash TEXT NOT NULL,
    session_id TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    monotonic_s REAL NOT NULL,
    wall_time_s REAL NOT NULL,
    elapsed_ns INTEGER NOT NULL,
    cpu_mw REAL,
    gpu_mw REAL,
    ane_mw REAL,
    combined_mw REAL,
    invalid INTEGER NOT NULL,
    thermal_pressure TEXT,
    PRIMARY KEY (session_id, sample_index),
    FOREIGN KEY (config_hash) REFERENCES experiments(config_hash)
);

CREATE TABLE IF NOT EXISTS block_observations (
    config_hash TEXT NOT NULL,
    session_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    phase TEXT NOT NULL,
    at_utc TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (session_id, round_index, model_name, phase),
    FOREIGN KEY (config_hash, model_name) REFERENCES models(config_hash, name)
);
"""


class StoreError(RuntimeError):
    pass


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self._ensure_column("runs", "cell_key", "TEXT")
        self._ensure_column("runs", "attempt_index", "INTEGER")
        self._ensure_column("runs", "accepted", "INTEGER")
        self._ensure_column("runs", "primary_eligible", "INTEGER")
        self._ensure_column("runs", "requested_num_predict", "INTEGER")
        self._ensure_column("runs", "request_prompt_sha256", "TEXT")
        self._ensure_column("runs", "power_invalid_sample_count", "INTEGER")
        self._ensure_column("runs", "power_covered_seconds", "REAL")
        self._ensure_column("power_samples", "elapsed_ns", "INTEGER")
        self._ensure_column("power_samples", "invalid", "INTEGER")
        self._ensure_column("power_samples", "thermal_pressure", "TEXT")
        self.connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_cell_attempt ON runs(cell_key, attempt_index)"
        )
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        existing = {row[1] for row in self.connection.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def register_experiment(self, config: ExperimentConfig, host: Dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO experiments
            (config_hash, name, config_json, prompt_file, prompt_file_sha256, created_at_utc, host_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config.config_hash,
                config.name,
                canonical_json(config.raw),
                str(config.prompt_file),
                file_sha256(config.prompt_file),
                utc_now(),
                canonical_json(host),
            ),
        )
        self.connection.commit()

    def register_models(self, config_hash: str, models: Iterable[ModelSpec]) -> None:
        self.connection.executemany(
            """
            INSERT INTO models
            (config_hash, name, family, parameters_b, active_parameters_b, quantization, role, reasoning, manifest_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(config_hash, name) DO UPDATE SET
              family=excluded.family,
              parameters_b=excluded.parameters_b,
              active_parameters_b=excluded.active_parameters_b,
              quantization=excluded.quantization,
              role=excluded.role,
              reasoning=excluded.reasoning
            """,
            [
                (
                    config_hash,
                    model.name,
                    model.family,
                    model.parameters_b,
                    model.active_parameters_b,
                    model.quantization,
                    model.role,
                    int(model.reasoning),
                )
                for model in models
            ],
        )
        self.connection.commit()

    def update_model_manifest(self, config_hash: str, name: str, manifest: Dict[str, Any]) -> None:
        existing = self.connection.execute(
            "SELECT manifest_json FROM models WHERE config_hash=? AND name=?",
            (config_hash, name),
        ).fetchone()
        run_count = self.connection.execute(
            "SELECT COUNT(*) FROM runs WHERE config_hash=? AND model_name=?",
            (config_hash, name),
        ).fetchone()[0]
        if existing is not None and existing[0] and run_count:
            old = json.loads(existing[0])
            old_digest = ((old.get("tag") or {}).get("digest"))
            new_digest = ((manifest.get("tag") or {}).get("digest"))
            if old_digest and new_digest and old_digest != new_digest:
                raise StoreError(
                    f"Model digest changed for {name}: {old_digest} -> {new_digest}. "
                    "Use a new database or pin the original model."
                )
        self.connection.execute(
            "UPDATE models SET manifest_json=? WHERE config_hash=? AND name=?",
            (canonical_json(manifest), config_hash, name),
        )
        self.connection.commit()

    def register_prompts(self, config_hash: str, prompts: Iterable[Prompt]) -> None:
        self.connection.executemany(
            """
            INSERT INTO prompts
            (config_hash, prompt_id, category, source, license, prompt_text, prompt_chars,
             prompt_bytes, grader_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(config_hash, prompt_id) DO UPDATE SET
              category=excluded.category,
              source=excluded.source,
              license=excluded.license,
              prompt_text=excluded.prompt_text,
              prompt_chars=excluded.prompt_chars,
              prompt_bytes=excluded.prompt_bytes,
              grader_json=excluded.grader_json,
              metadata_json=excluded.metadata_json
            """,
            [
                (
                    config_hash,
                    prompt.id,
                    prompt.category,
                    prompt.source,
                    prompt.license,
                    prompt.prompt,
                    len(prompt.prompt),
                    len(prompt.prompt.encode("utf-8")),
                    canonical_json(prompt.grader),
                    canonical_json(prompt.metadata),
                )
                for prompt in prompts
            ],
        )
        self.connection.commit()

    def has_accepted(self, cell_key: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM runs WHERE cell_key=? AND accepted=1 LIMIT 1", (cell_key,)
        ).fetchone()
        return row is not None

    def next_attempt(self, cell_key: str) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(attempt_index), -1) + 1 FROM runs WHERE cell_key=?",
            (cell_key,),
        ).fetchone()
        return int(row[0])

    def insert_run(self, row: Dict[str, Any]) -> None:
        keys = list(row)
        sql = f"INSERT INTO runs ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})"
        self.connection.execute(sql, [row[key] for key in keys])
        self.connection.commit()

    def store_power_samples(
        self, config_hash: str, session_id: str, samples: Iterable[PowerSample]
    ) -> None:
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO power_samples
            (config_hash, session_id, sample_index, monotonic_s, wall_time_s, elapsed_ns,
             cpu_mw, gpu_mw, ane_mw, combined_mw, invalid, thermal_pressure)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    config_hash,
                    session_id,
                    index,
                    sample.monotonic_s,
                    sample.wall_time_s,
                    sample.elapsed_ns,
                    sample.cpu_mw,
                    sample.gpu_mw,
                    sample.ane_mw,
                    sample.combined_mw,
                    int(sample.invalid),
                    sample.thermal_pressure,
                )
                for index, sample in enumerate(samples)
            ],
        )
        self.connection.commit()

    def update_block_baseline(
        self,
        config_hash: str,
        session_id: str,
        model_name: str,
        round_index: int,
        baseline_soc_mw: float,
    ) -> None:
        self.connection.execute(
            """
            UPDATE runs
            SET baseline_soc_mw=?,
                incremental_soc_j=gross_soc_j - (? * power_covered_seconds / 1000.0),
                incremental_soc_wh=(gross_soc_j - (? * power_covered_seconds / 1000.0)) / 3600.0
            WHERE config_hash=? AND session_id=? AND model_name=? AND round_index=? AND is_warmup=0
            """,
            (
                baseline_soc_mw,
                baseline_soc_mw,
                baseline_soc_mw,
                config_hash,
                session_id,
                model_name,
                round_index,
            ),
        )
        self.connection.commit()

    def insert_block_observation(
        self,
        config_hash: str,
        session_id: str,
        round_index: int,
        model_name: str,
        phase: str,
        snapshot: Dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO block_observations
            (config_hash, session_id, round_index, model_name, phase, at_utc, snapshot_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config_hash,
                session_id,
                round_index,
                model_name,
                phase,
                utc_now(),
                canonical_json(snapshot),
            ),
        )
        self.connection.commit()

    def rows(self, query: str, parameters: Iterable[Any] = ()) -> List[sqlite3.Row]:
        return list(self.connection.execute(query, tuple(parameters)))
