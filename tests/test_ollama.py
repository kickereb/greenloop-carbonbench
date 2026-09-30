import json
import unittest
from unittest.mock import patch

from carbonbench.ollama import OllamaClient


class FakeStream:
    def __init__(self, rows):
        self.rows = [json.dumps(row).encode("utf-8") + b"\n" for row in rows]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.rows)


class OllamaTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_stream_assembly_and_usage(self, urlopen):
        urlopen.return_value = FakeStream(
            [
                {"response": "forty ", "done": False},
                {"response": "two", "done": False},
                {
                    "response": "",
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "prompt_eval_cached_count": 2,
                    "eval_count": 2,
                    "total_duration": 100,
                    "load_duration": 1,
                    "prompt_eval_duration": 20,
                    "eval_duration": 40,
                },
            ]
        )
        result = OllamaClient().generate(
            "model",
            "prompt",
            temperature=0,
            seed=1,
            num_predict=4,
            num_ctx=4096,
            keep_alive="5m",
        )
        self.assertEqual(result.response, "forty two")
        self.assertEqual(result.prompt_eval_cached_count, 2)
        self.assertEqual(result.eval_count, 2)
        self.assertIsNotNone(result.first_token_duration_s)


if __name__ == "__main__":
    unittest.main()

