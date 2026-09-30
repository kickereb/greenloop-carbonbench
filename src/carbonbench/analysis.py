from __future__ import annotations

import csv
import html
import json
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .util import bootstrap_mean_ci, format_optional, median, percentile, write_json
from .db import Store


class AnalysisError(RuntimeError):
    pass


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _safe_median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(clean) if clean else None


def _coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(clean) < 2:
        return None
    center = statistics.fmean(clean)
    return statistics.stdev(clean) / abs(center) if center else None


def _spearman(actual: Sequence[float], predicted: Sequence[float]) -> Optional[float]:
    if len(actual) < 2 or len(actual) != len(predicted):
        return None

    def ranks(values: Sequence[float]) -> List[float]:
        ordered = sorted(enumerate(values), key=lambda item: item[1])
        result = [0.0] * len(values)
        position = 0
        while position < len(ordered):
            end = position + 1
            while end < len(ordered) and ordered[end][1] == ordered[position][1]:
                end += 1
            rank = (position + 1 + end) / 2.0
            for original_index, _ in ordered[position:end]:
                result[original_index] = rank
            position = end
        return result

    left, right = ranks(actual), ranks(predicted)
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _nnls_coordinate_descent(
    matrix: Sequence[Sequence[float]],
    target: Sequence[float],
    iterations: int = 3000,
) -> List[float]:
    if not matrix or len(matrix) != len(target):
        raise AnalysisError("Estimator matrix and target are empty or misaligned")
    width = len(matrix[0])
    coefficients = [0.0] * width
    weights = [1.0] * len(target)
    for _outer in range(5):
        for _ in range(iterations // 5):
            max_change = 0.0
            for column in range(width):
                numerator = 0.0
                denominator = 0.0
                for row_index, row in enumerate(matrix):
                    other = sum(row[j] * coefficients[j] for j in range(width) if j != column)
                    x = row[column]
                    weight = weights[row_index]
                    numerator += weight * x * (target[row_index] - other)
                    denominator += weight * x * x
                candidate = max(0.0, numerator / denominator) if denominator else 0.0
                max_change = max(max_change, abs(candidate - coefficients[column]))
                coefficients[column] = candidate
            if max_change < 1e-10:
                break
        residuals = [
            y - sum(x * beta for x, beta in zip(row, coefficients))
            for row, y in zip(matrix, target)
        ]
        scale = statistics.median(abs(value) for value in residuals) * 1.4826
        if scale <= 1e-12:
            break
        threshold = 1.345 * scale
        weights = [1.0 if abs(value) <= threshold else threshold / abs(value) for value in residuals]
    return coefficients


def _fit_estimator(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["parameters_b"] is None
            or row["active_parameters_b"] is None
            or row["prompt_eval_count"] is None
            or row["eval_count"] is None
            or row["energy_j"] is None
            or row["energy_j"] <= 0
        ):
            continue
        cells[(row["model_name"], row["prompt_id"])].append(row)

    aggregated: List[Dict[str, Any]] = []
    for group in cells.values():
        exemplar = dict(group[0])
        exemplar["prompt_eval_count"] = _safe_median([row["prompt_eval_count"] for row in group])
        exemplar["eval_count"] = _safe_median([row["eval_count"] for row in group])
        exemplar["energy_j"] = _safe_median([row["energy_j"] for row in group])
        aggregated.append(exemplar)

    training = [
        row
        for row in aggregated
        if row["role"] == "fit" and row["benchmark_split"] != "locked"
    ]
    primary_test = [
        row
        for row in aggregated
        if row["role"] == "holdout" and row["benchmark_split"] == "locked"
    ]
    if len(training) < 10 or len(primary_test) < 3:
        return None

    raw_features = [
        [
            1.0,
            float(row["parameters_b"]) * float(row["prompt_eval_count"]),
            float(row["active_parameters_b"]) * float(row["eval_count"]),
        ]
        for row in training
    ]
    scales = [1.0]
    for column in (1, 2):
        values = [row[column] for row in raw_features if row[column] > 0]
        scales.append(statistics.median(values) if values else 1.0)
    matrix = [[row[index] / scales[index] for index in range(3)] for row in raw_features]
    target = [float(row["energy_j"]) for row in training]
    normalized_beta = _nnls_coordinate_descent(matrix, target)
    beta = [normalized_beta[index] / scales[index] for index in range(3)]

    total_token_rate = statistics.median(
        float(row["energy_j"]) / (float(row["prompt_eval_count"]) + float(row["eval_count"]))
        for row in training
        if float(row["prompt_eval_count"]) + float(row["eval_count"]) > 0
    )

    def prediction(row: Dict[str, Any]) -> float:
        return max(
            0.0,
            beta[0]
            + beta[1] * float(row["parameters_b"]) * float(row["prompt_eval_count"])
            + beta[2] * float(row["active_parameters_b"]) * float(row["eval_count"]),
        )

    def metrics(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        actual = [float(row["energy_j"]) for row in subset]
        predicted = [prediction(row) for row in subset]
        baseline = [
            total_token_rate * (float(row["prompt_eval_count"]) + float(row["eval_count"]))
            for row in subset
        ]
        ape = [abs(p - a) / a for a, p in zip(actual, predicted) if a > 0]
        baseline_ape = [abs(p - a) / a for a, p in zip(actual, baseline) if a > 0]
        within_factor_2 = [0.5 <= p / a <= 2.0 for a, p in zip(actual, predicted) if a > 0]
        return {
            "n": len(actual),
            "mdape": _safe_median(ape),
            "p90_ape": percentile(ape, 0.90),
            "share_within_40_percent": _safe_mean([float(value <= 0.40) for value in ape]),
            "share_within_factor_2": _safe_mean([float(value) for value in within_factor_2]),
            "spearman": _spearman(actual, predicted),
            "naive_token_mdape": _safe_median(baseline_ape),
            "mdape_improvement_over_naive": (
                1.0 - (_safe_median(ape) / _safe_median(baseline_ape))
                if _safe_median(ape) is not None and _safe_median(baseline_ape)
                else None
            ),
        }

    by_family = {}
    for family in sorted({row["family"] for row in primary_test}):
        by_family[family] = metrics([row for row in primary_test if row["family"] == family])
    return {
        "formula": "E_J = a + b_prefill*(total_parameters_B*input_tokens) + b_decode*(active_parameters_B*output_tokens)",
        "energy_field": rows[0]["energy_field"] if rows else None,
        "coefficients": {"a": beta[0], "b_prefill": beta[1], "b_decode": beta[2]},
        "training_cells": len(training),
        "primary_test_definition": "held-out model family AND locked prompt",
        "primary_test": metrics(primary_test),
        "by_holdout_family": by_family,
        "caution": "This estimator transfers only within this Mac/Ollama setup; it does not validate cloud-model emissions.",
    }


def _load_analysis_rows(connection: sqlite3.Connection, config_hash: str) -> List[Dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT r.*, m.family, m.parameters_b, m.active_parameters_b, m.role,
               p.category, p.metadata_json
        FROM runs r
        JOIN models m ON m.config_hash=r.config_hash AND m.name=r.model_name
        JOIN prompts p ON p.config_hash=r.config_hash AND p.prompt_id=r.prompt_id
        WHERE r.config_hash=? AND r.is_warmup=0
        ORDER BY r.schedule_index
        """,
        (config_hash,),
    ).fetchall()
    result = []
    for source in rows:
        row = dict(source)
        metadata = json.loads(row.pop("metadata_json") or "{}")
        row["benchmark_split"] = metadata.get("benchmark_split", "development")
        # Older databases predate explicit attempt acceptance columns.  Preserve
        # backwards compatibility while making the scientific population explicit.
        row["accepted_flag"] = bool(row.get("accepted")) if row.get("accepted") is not None else row["status"] == "success"
        row["primary_flag"] = bool(row.get("primary_eligible")) if row.get("primary_eligible") is not None else row["accepted_flag"]
        result.append(row)
    use_incremental = any(row.get("incremental_soc_j") is not None for row in result)
    energy_field = "incremental_soc_j" if use_incremental else "gross_soc_j"
    for row in result:
        row["energy_j"] = row.get(energy_field)
        row["energy_field"] = energy_field
    return result


def verify_database(db_path: Path, config_hash: Optional[str] = None) -> Dict[str, Any]:
    # Apply additive table migrations before read-only analysis of an older run database.
    migration_store = Store(db_path)
    migration_store.close()
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    hashes = [row[0] for row in connection.execute("SELECT config_hash FROM experiments ORDER BY created_at_utc")]
    if not hashes:
        connection.close()
        raise AnalysisError("The database contains no experiment")
    if config_hash is None:
        if len(hashes) > 1:
            connection.close()
            raise AnalysisError("Database contains multiple configs; specify --config-hash")
        config_hash = hashes[0]
    experiment = connection.execute(
        "SELECT * FROM experiments WHERE config_hash=?", (config_hash,)
    ).fetchone()
    if experiment is None:
        connection.close()
        raise AnalysisError(f"Unknown config hash: {config_hash}")
    config = json.loads(experiment["config_json"])
    prompt_count = connection.execute(
        "SELECT COUNT(*) FROM prompts WHERE config_hash=? AND prompt_id!='__carbonbench_warmup__'",
        (config_hash,),
    ).fetchone()[0]
    model_count = connection.execute(
        "SELECT COUNT(*) FROM models WHERE config_hash=?", (config_hash,)
    ).fetchone()[0]
    expected = prompt_count * model_count * int(config.get("repetitions", 1))
    accepted_pred = "COALESCE(r.accepted, CASE WHEN r.status='success' THEN 1 ELSE 0 END)=1"
    counts = dict(connection.execute(
        f"""
        SELECT COUNT(*) AS recorded,
               SUM(CASE WHEN r.status='success' THEN 1 ELSE 0 END) AS successful_attempts,
               SUM(CASE WHEN r.status!='success' THEN 1 ELSE 0 END) AS failed_attempts,
               COUNT(DISTINCT CASE WHEN {accepted_pred} THEN COALESCE(r.cell_key, r.run_key) END) AS accepted_cells,
               SUM(CASE WHEN {accepted_pred} AND r.flags_json LIKE '%unexpected_model_load%' THEN 1 ELSE 0 END) AS loads,
               SUM(CASE WHEN {accepted_pred} AND r.flags_json LIKE '%prompt_cache_hit%' THEN 1 ELSE 0 END) AS cache_hits,
               SUM(CASE WHEN {accepted_pred} AND r.flags_json LIKE '%low_power_sample_coverage%' THEN 1 ELSE 0 END) AS low_coverage,
               SUM(CASE WHEN {accepted_pred} AND r.flags_json LIKE '%invalid_power_sample%' THEN 1 ELSE 0 END) AS invalid_power,
               SUM(CASE WHEN {accepted_pred} AND r.flags_json LIKE '%output_truncated%' AND p.category!='controlled_decode' THEN 1 ELSE 0 END) AS quality_truncations
        FROM runs r JOIN prompts p ON p.config_hash=r.config_hash AND p.prompt_id=r.prompt_id
        WHERE r.config_hash=? AND r.is_warmup=0
        """, (config_hash,)).fetchone())
    counts["successful"] = counts.get("accepted_cells") or 0
    counts["failed"] = counts.get("failed_attempts") or 0
    duplicate_cells = connection.execute(
        f"""
        SELECT COUNT(*) FROM (
          SELECT COALESCE(cell_key, run_key) cell, COUNT(*) n
          FROM runs r WHERE config_hash=? AND is_warmup=0 AND {accepted_pred}
          GROUP BY cell HAVING n>1
        )
        """,
        (config_hash,),
    ).fetchone()[0]
    coverage_values = [
        row[0]
        for row in connection.execute(
            f"SELECT power_sample_coverage FROM runs r WHERE config_hash=? AND is_warmup=0 AND {accepted_pred} AND power_sample_coverage IS NOT NULL",
            (config_hash,),
        )
    ]
    observations = connection.execute(
        "SELECT session_id, round_index, model_name, phase, snapshot_json FROM block_observations WHERE config_hash=?",
        (config_hash,),
    ).fetchall()
    by_block: Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for observation in observations:
        by_block[(observation[0], observation[1], observation[2])][observation[3]] = json.loads(
            observation[4]
        )
    pageout_blocks = 0
    compared_blocks = 0
    complete_observation_blocks = 0
    for phases in by_block.values():
        if "pre" not in phases or "post" not in phases:
            continue
        complete_observation_blocks += 1
        pre_match = re.search(r"Pageouts:\s+(\d+)", phases["pre"].get("vm_stat") or "")
        post_match = re.search(r"Pageouts:\s+(\d+)", phases["post"].get("vm_stat") or "")
        if pre_match and post_match:
            compared_blocks += 1
            if int(post_match.group(1)) > int(pre_match.group(1)):
                pageout_blocks += 1
    requires_power = str(config.get("energy", {}).get("mode", "none")) == "powermetrics"
    requires_block_observation = requires_power and float(config.get("energy", {}).get("baseline_seconds", 0)) > 0
    energy_methods = [row[0] for row in connection.execute(
        f"SELECT DISTINCT energy_method FROM runs r WHERE config_hash=? AND is_warmup=0 AND {accepted_pred}", (config_hash,)
    )]
    connection.close()
    checks = {
        "complete_matrix": (counts.get("accepted_cells") or 0) == expected,
        "no_duplicate_accepted_cells": duplicate_cells == 0,
        "energy_method_matches_config": energy_methods == [str(config.get("energy", {}).get("mode", "none"))],
        "median_power_coverage_at_least_90_percent": (
            len(coverage_values) == expected and _safe_median(coverage_values) >= 0.90
            if coverage_values
            else (False if requires_power else None)
        ),
        "no_unexpected_model_loads": (counts["loads"] or 0) == 0,
        "no_prompt_cache_hits": (counts["cache_hits"] or 0) == 0,
        "no_low_coverage_requests": (counts["low_coverage"] or 0) == 0,
        "no_invalid_power_requests": (counts["invalid_power"] or 0) == 0,
        "no_truncated_quality_requests": (counts["quality_truncations"] or 0) == 0,
        "no_pageouts_during_model_blocks": (
            pageout_blocks == 0 if (requires_power and compared_blocks) else None
        ),
        "complete_pre_post_block_observations": (
            complete_observation_blocks == model_count * int(config.get("repetitions", 1))
            if requires_block_observation
            else None
        ),
    }
    return {
        "config_hash": config_hash,
        "expected_measured_runs": expected,
        **counts,
        "duplicate_cells": duplicate_cells,
        "median_power_sample_coverage": _safe_median(coverage_values),
        "blocks_with_pageout_counters": compared_blocks,
        "blocks_with_pageouts": pageout_blocks,
        "blocks_with_pre_post_observations": complete_observation_blocks,
        "checks": checks,
        "all_required_checks_pass": all(value is True for value in checks.values() if value is not None),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_pareto_svg(path: Path, models: List[Dict[str, Any]]) -> Optional[Path]:
    points = [row for row in models if row["mean_energy_j"] is not None and row["mean_quality"] is not None]
    if not points:
        return None
    width, height, margin = 760, 480, 70
    xs = [row["mean_energy_j"] for row in points]
    ys = [row["mean_quality"] for row in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(1.0, max(ys))
    if x_max == x_min:
        x_max = x_min + 1

    def sx(value: float) -> float:
        return margin + (value - x_min) / (x_max - x_min) * (width - 2 * margin)

    def sy(value: float) -> float:
        return height - margin - (value - y_min) / (y_max - y_min) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Mean energy per request (J; labelled proxy)</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">Mean deterministic quality score</text>',
    ]
    for row in points:
        x, y = sx(row["mean_energy_j"]), sy(row["mean_quality"])
        label = html.escape(row["model"])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#147d64"><title>{label}</title></circle>')
        parts.append(f'<text x="{x+9:.1f}" y="{y-7:.1f}" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def analyze_database(
    db_path: Path, output_directory: Path, config_hash: Optional[str] = None
) -> Dict[str, Any]:
    verification = verify_database(db_path, config_hash)
    config_hash = verification["config_hash"]
    connection = sqlite3.connect(str(db_path))
    rows = _load_analysis_rows(connection, config_hash)
    connection.close()
    # Retries and flagged attempts stay in runs.csv for auditability, but never
    # contaminate the primary model/energy estimates.
    successful = [
        row for row in rows
        if row["status"] == "success" and row["accepted_flag"] and row["primary_flag"]
    ]
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in successful:
        by_model[row["model_name"]].append(row)
    model_summary: List[Dict[str, Any]] = []
    for model_name, group in sorted(by_model.items()):
        decode_rates = [
            row["eval_count"] * 1_000_000_000 / row["eval_duration_ns"]
            for row in group
            if row["eval_count"] is not None and row["eval_duration_ns"]
        ]
        prompt_energy: Dict[str, List[float]] = defaultdict(list)
        for row in group:
            if row["energy_j"] is not None:
                prompt_energy[row["prompt_id"]].append(float(row["energy_j"]))
        # Bootstrap prompt-level medians so hardware repetitions are not treated as independent prompts.
        energy_values = [statistics.median(values) for values in prompt_energy.values()]
        energy_point, energy_low, energy_high = bootstrap_mean_ci(energy_values) if energy_values else (None, None, None)
        model_summary.append(
            {
                "model": model_name,
                "family": group[0]["family"],
                "role": group[0]["role"],
                "successful_runs": len(group),
                "mean_quality": _safe_mean([row["quality_score"] for row in group]),
                "mean_energy_j": energy_point,
                "energy_mean_ci95_low_j": energy_low,
                "energy_mean_ci95_high_j": energy_high,
                "median_wall_s": _safe_median([row["wall_duration_s"] for row in group]),
                "median_first_token_s": _safe_median([row["first_token_duration_s"] for row in group]),
                "median_decode_tokens_s": _safe_median(decode_rates),
                "mean_input_tokens": _safe_mean([row["prompt_eval_count"] for row in group]),
                "mean_output_tokens": _safe_mean([row["eval_count"] for row in group]),
                "median_power_coverage": _safe_median([row["power_sample_coverage"] for row in group]),
            }
        )

    cell_energy: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in successful:
        if row["energy_j"] is not None:
            cell_energy[(row["model_name"], row["prompt_id"])].append(float(row["energy_j"]))
    cvs = [
        value
        for value in (_coefficient_of_variation(values) for values in cell_energy.values())
        if value is not None
    ]
    repeatability = {
        "cells_with_two_or_more_repeats": len(cvs),
        "median_within_cell_cv": _safe_median(cvs),
        "share_cells_cv_at_most_10_percent": _safe_mean([float(value <= 0.10) for value in cvs]),
        "h1_pass": (
            _safe_median(cvs) <= 0.10
            and _safe_mean([float(value <= 0.10) for value in cvs]) >= 0.90
            if cvs
            else None
        ),
    }
    estimator = _fit_estimator(successful) if verification["all_required_checks_pass"] else None
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "model_summary.csv", model_summary)
    export_fields = [
        "model_name",
        "family",
        "role",
        "prompt_id",
        "category",
        "benchmark_split",
        "round_index",
        "status",
        "prompt_eval_count",
        "prompt_eval_cached_count",
        "eval_count",
        "requested_num_predict",
        "wall_duration_s",
        "first_token_duration_s",
        "quality_score",
        "gross_soc_j",
        "incremental_soc_j",
        "power_sample_coverage",
        "flags_json",
    ]
    _write_csv(
        output_directory / "runs.csv",
        [{field: row.get(field) for field in export_fields} for row in rows],
    )
    svg = _write_pareto_svg(output_directory / "quality_energy.svg", model_summary)
    summary = {
        "database": str(db_path),
        "config_hash": config_hash,
        "verification": verification,
        "repeatability": repeatability,
        "models": model_summary,
        "estimator": estimator,
        "energy_label": (
            "incremental estimated Apple CPU+GPU+ANE SoC energy where positive; otherwise gross estimated SoC energy"
        ),
        "claim_boundary": (
            "These are same-device component-energy proxies, not wall electricity or validated cloud-model emissions."
        ),
    }
    write_json(output_directory / "summary.json", summary)

    lines = [
        "# CarbonBench local report",
        "",
        f"Config hash: `{config_hash}`",
        "",
        "**Claim boundary:** " + summary["claim_boundary"],
        "",
        "## Data quality",
        "",
        f"- Expected measured runs: {verification['expected_measured_runs']}",
        f"- Successful: {verification.get('successful') or 0}",
        f"- Failed: {verification.get('failed') or 0}",
        f"- Median telemetry coverage: {format_optional(verification['median_power_sample_coverage'], 3)}",
        f"- Complete matrix: {verification['checks']['complete_matrix']}",
        "",
        "## Model summary",
        "",
        "| Model | Role | Runs | Quality | Mean energy J | Median latency s | Decode tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary:
        lines.append(
            f"| {row['model']} | {row['role']} | {row['successful_runs']} | "
            f"{format_optional(row['mean_quality'], 3)} | {format_optional(row['mean_energy_j'], 3)} | "
            f"{format_optional(row['median_wall_s'], 3)} | {format_optional(row['median_decode_tokens_s'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Repeatability",
            "",
            f"Median within-cell CV: {format_optional(repeatability['median_within_cell_cv'], 3)}. ",
            f"H1 pass: {repeatability['h1_pass']}.",
            "",
            "## Held-out estimator",
            "",
        ]
    )
    if estimator:
        test = estimator["primary_test"]
        lines.extend(
            [
                f"Formula: `{estimator['formula']}`",
                "",
                f"Double-held-out cells: {test['n']}; MdAPE: {format_optional(test['mdape'], 3)}; "
                f"P90 APE: {format_optional(test['p90_ape'], 3)}; Spearman: {format_optional(test['spearman'], 3)}.",
            ]
        )
    else:
        lines.append("Not enough fit/development and holdout/locked cells to fit the prespecified estimator.")
    if svg:
        lines.extend(["", "## Quality–energy view", "", "![Quality versus energy](quality_energy.svg)"])
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Do not interpret a lower energy value as better unless quality remains inside the predeclared non-inferiority margin. "
            "Do not extrapolate this Mac result to cloud carbon without wall-power calibration, server hardware, PUE, batching, and location data.",
            "",
        ]
    )
    (output_directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
