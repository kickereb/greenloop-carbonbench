import json
import tempfile
import unittest
from pathlib import Path

from carbonbench.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def _write(self, directory, payload):
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_relative_prompt_path_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                {"name": "test", "prompt_file": "prompts.jsonl", "models": ["model:1b"]},
            )
            config = load_config(path)
            self.assertEqual(config.prompt_file, Path(directory).resolve() / "prompts.jsonl")
            self.assertEqual(config.models[0].family, "model")
            self.assertEqual(config.repetitions, 3)

    def test_duplicate_model_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                {"name": "test", "prompt_file": "x", "models": ["same", "same"]},
            )
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_invalid_energy_mode_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                {
                    "name": "test",
                    "prompt_file": "x",
                    "models": ["m"],
                    "energy": {"mode": "magic"},
                },
            )
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_prompt_content_is_part_of_config_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "p.jsonl"
            prompt.write_text("one\n", encoding="utf-8")
            path = self._write(
                directory,
                {"name": "test", "prompt_file": "p.jsonl", "models": ["m"]},
            )
            first = load_config(path).config_hash
            prompt.write_text("two\n", encoding="utf-8")
            second = load_config(path).config_hash
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
