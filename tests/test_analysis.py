import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from carbonbench.analysis import analyze_database, verify_database
from carbonbench.config import load_config
from carbonbench.db import Store
from carbonbench.prompts import load_prompts
from carbonbench.util import stable_hash, utc_now


class AnalysisTests(unittest.TestCase):
    def test_complete_synthetic_matrix_and_estimator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "prompts.jsonl"
            prompt_rows = []
            for index in range(10):
                prompt_rows.append(
                    {
                        "id": f"p{index}",
                        "category": "test",
                        "prompt": f"Prompt {index}",
                        "source": "test",
                        "license": "CC0",
                        "grader": {"type": "exact", "answers": ["ok"]},
                        "metadata": {"benchmark_split": "locked" if index >= 8 else "development"},
                    }
                )
            prompt_path.write_text(
                "".join(json.dumps(row) + "\n" for row in prompt_rows), encoding="utf-8"
            )
            config_payload = {
                "name": "synthetic",
                "prompt_file": "prompts.jsonl",
                "repetitions": 1,
                "models": [
                    {"name": "fit-a", "family": "fit", "parameters_b": 2, "role": "fit"},
                    {"name": "fit-b", "family": "fit", "parameters_b": 4, "role": "fit"},
                    {"name": "hold-a", "family": "alpha", "parameters_b": 3, "role": "holdout"},
                    {"name": "hold-b", "family": "beta", "parameters_b": 6, "role": "holdout"},
                ],
                "energy": {"mode": "powermetrics", "baseline_seconds": 0},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")
            config = load_config(config_path)
            prompts = load_prompts(prompt_path)
            db_path = root / "runs.sqlite"
            store = Store(db_path)
            store.register_experiment(config, {"host": "test"})
            store.register_models(config.config_hash, config.models)
            store.register_prompts(config.config_hash, prompts)
            index = 0
            for model in config.models:
                for prompt_index, prompt in enumerate(prompts):
                    input_tokens = 100 + prompt_index * 5
                    output_tokens = 20 + prompt_index
                    energy = 0.2 + 0.001 * model.parameters_b * input_tokens + 0.002 * model.active_parameters_b * output_tokens
                    key = stable_hash(f"{model.name}-{prompt.id}")
                    store.insert_run(
                        {
                            "run_key": key,
                            "config_hash": config.config_hash,
                            "session_id": "session",
                            "schedule_index": index,
                            "round_index": 0,
                            "model_name": model.name,
                            "prompt_id": prompt.id,
                            "is_warmup": 0,
                            "seed": 1,
                            "requested_num_predict": 64,
                            "status": "success",
                            "started_at_utc": utc_now(),
                            "ended_at_utc": utc_now(),
                            "start_monotonic_s": float(index),
                            "end_monotonic_s": float(index + 1),
                            "wall_duration_s": 1.0,
                            "prompt_eval_count": input_tokens,
                            "eval_count": output_tokens,
                            "quality_score": 1.0,
                            "energy_method": "powermetrics",
                            "power_sample_count": 2,
                            "power_invalid_sample_count": 0,
                            "power_covered_seconds": 1.0,
                            "power_sample_coverage": 1.0,
                            "gross_soc_j": energy + 0.1,
                            "incremental_soc_j": energy,
                            "gross_soc_wh": (energy + 0.1) / 3600,
                            "incremental_soc_wh": energy / 3600,
                            "flags_json": "[]",
                        }
                    )
                    index += 1
            store.insert_run(
                {
                    "run_key": "rejected-cache-attempt",
                    "cell_key": "rejected-cache-cell",
                    "attempt_index": 0,
                    "accepted": 0,
                    "primary_eligible": 0,
                    "config_hash": config.config_hash,
                    "session_id": "session",
                    "schedule_index": index,
                    "round_index": 0,
                    "model_name": config.models[0].name,
                    "prompt_id": prompts[0].id,
                    "is_warmup": 0,
                    "seed": 1,
                    "requested_num_predict": 64,
                    "status": "success",
                    "started_at_utc": utc_now(),
                    "ended_at_utc": utc_now(),
                    "start_monotonic_s": float(index),
                    "end_monotonic_s": float(index + 1),
                    "wall_duration_s": 1.0,
                    "prompt_eval_count": 100,
                    "prompt_eval_cached_count": 24,
                    "eval_count": 20,
                    "quality_score": 1.0,
                    "energy_method": "powermetrics",
                    "power_sample_count": 2,
                    "power_invalid_sample_count": 0,
                    "power_covered_seconds": 1.0,
                    "power_sample_coverage": 1.0,
                    "gross_soc_j": 1.0,
                    "incremental_soc_j": 0.9,
                    "gross_soc_wh": 1.0 / 3600,
                    "incremental_soc_wh": 0.9 / 3600,
                    "flags_json": '["prompt_cache_hit"]',
                }
            )
            store.close()
            verification = verify_database(db_path)
            self.assertTrue(verification["checks"]["complete_matrix"])
            self.assertEqual(verification["cache_hits"], 1)
            self.assertFalse(verification["checks"]["no_prompt_cache_hits"])
            with sqlite3.connect(str(db_path)) as connection:
                connection.execute("DELETE FROM runs WHERE run_key='rejected-cache-attempt'")
            output = root / "report"
            summary = analyze_database(db_path, output)
            self.assertIsNotNone(summary["estimator"])
            self.assertEqual(summary["estimator"]["primary_test"]["n"], 4)
            self.assertTrue((output / "report.md").exists())
            self.assertTrue((output / "model_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
