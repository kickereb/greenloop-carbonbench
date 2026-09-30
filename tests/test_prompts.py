import json
import tempfile
import unittest
from pathlib import Path

from carbonbench.prompts import PromptError, grade_response, load_prompts, normalize_answer, token_f1


class PromptTests(unittest.TestCase):
    def test_normalize_removes_articles_and_punctuation(self):
        self.assertEqual(normalize_answer("The 42-MWh, result!"), "42mwh result")

    def test_exact_uses_final_line(self):
        score, _ = grade_response("Reasoning here\nFINAL: GL-123", {"type": "exact", "answers": ["GL-123"]})
        self.assertEqual(score, 1.0)

    def test_numeric_uses_final_number(self):
        score, detail = grade_response("4 + 5 = 9\nFINAL: 9", {"type": "numeric", "answer": "9"})
        self.assertEqual(score, 1.0)
        self.assertIn("predicted=9", detail)

    def test_squad_f1_uses_best_alias(self):
        score, _ = grade_response(
            "FINAL: 42 megawatt hours", {"type": "squad_f1", "answers": ["forty two", "42 megawatt hours"]}
        )
        self.assertEqual(score, 1.0)

    def test_contains_all(self):
        score, _ = grade_response("wind and SOLAR", {"type": "contains_all", "answers": ["wind", "solar"]})
        self.assertEqual(score, 1.0)

    def test_duplicate_prompt_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            row = {"id": "x", "prompt": "hello", "grader": {"type": "none"}}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(PromptError):
                load_prompts(path)


if __name__ == "__main__":
    unittest.main()

