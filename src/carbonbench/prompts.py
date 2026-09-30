from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    prompt: str
    source: str
    license: str
    grader: Dict[str, Any]
    metadata: Dict[str, Any]


def load_prompts(path: Path) -> List[Prompt]:
    prompts: List[Prompt] = []
    seen = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise PromptError(f"Prompt file not found: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PromptError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise PromptError(f"Prompt at {path}:{line_number} must be an object")
        prompt_id = str(raw.get("id", "")).strip()
        text = raw.get("prompt")
        if not prompt_id or not isinstance(text, str) or not text.strip():
            raise PromptError(f"Prompt at {path}:{line_number} needs id and prompt")
        if prompt_id in seen:
            raise PromptError(f"Duplicate prompt id: {prompt_id}")
        seen.add(prompt_id)
        grader = raw.get("grader", {"type": "none"})
        if not isinstance(grader, dict) or "type" not in grader:
            raise PromptError(f"Prompt {prompt_id} has an invalid grader")
        prompts.append(
            Prompt(
                id=prompt_id,
                category=str(raw.get("category", "unspecified")),
                prompt=text,
                source=str(raw.get("source", "unspecified")),
                license=str(raw.get("license", "unspecified")),
                grader=grader,
                metadata=raw.get("metadata", {}) if isinstance(raw.get("metadata", {}), dict) else {},
            )
        )
    if not prompts:
        raise PromptError(f"No prompts found in {path}")
    return prompts


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def _last_final(response: str) -> str:
    matches = re.findall(r"(?im)^\s*(?:final(?: answer)?|answer)\s*:\s*(.+?)\s*$", response)
    return matches[-1].strip() if matches else response.strip()


def token_f1(prediction: str, references: List[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    best = 0.0
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        if not prediction_tokens and not reference_tokens:
            best = max(best, 1.0)
            continue
        counts: Dict[str, int] = {}
        for token in prediction_tokens:
            counts[token] = counts.get(token, 0) + 1
        common = 0
        for token in reference_tokens:
            if counts.get(token, 0) > 0:
                common += 1
                counts[token] -= 1
        if common:
            precision = common / len(prediction_tokens)
            recall = common / len(reference_tokens)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def grade_response(response: str, grader: Dict[str, Any]) -> Tuple[Optional[float], str]:
    grader_type = str(grader.get("type", "none"))
    candidate = _last_final(response)
    answers = [str(item) for item in grader.get("answers", [])]
    if grader_type == "none":
        return None, "ungraded"
    if grader_type == "exact":
        score = float(any(normalize_answer(candidate) == normalize_answer(answer) for answer in answers))
        return score, "normalized exact match"
    if grader_type == "squad_f1":
        score = token_f1(candidate, answers)
        return score, "token F1 against answer aliases"
    if grader_type == "contains_all":
        normalized = normalize_answer(response)
        missing = [answer for answer in answers if normalize_answer(answer) not in normalized]
        return (0.0 if missing else 1.0), ("missing: " + ", ".join(missing) if missing else "all terms found")
    if grader_type == "regex":
        pattern = str(grader.get("pattern", ""))
        matched = bool(re.search(pattern, response, flags=re.IGNORECASE | re.MULTILINE))
        return float(matched), f"regex {'matched' if matched else 'did not match'}"
    if grader_type == "numeric":
        expected = str(grader.get("answer", answers[0] if answers else ""))
        expected_clean = expected.replace(",", "").strip()
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", candidate)
        predicted = numbers[-1].replace(",", "") if numbers else ""
        try:
            tolerance = float(grader.get("tolerance", 0.0))
            matched = bool(predicted) and abs(float(predicted) - float(expected_clean)) <= tolerance
        except ValueError:
            matched = predicted == expected_clean
        return float(matched), f"predicted={predicted or 'none'} expected={expected_clean}"
    raise PromptError(f"Unknown grader type: {grader_type}")
