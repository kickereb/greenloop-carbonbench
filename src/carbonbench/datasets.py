from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .util import file_sha256


GSM8K = {
    "url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
    "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "commit": "3101c7d5072418e28b9008a6636bde82a006892c",
    "license": "MIT",
}

SQUAD = {
    "url": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/eee5fdbf62f8613a7812b03419e6b29617b74fd1/dataset/dev-v1.1.json",
    "sha256": "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
    "commit": "eee5fdbf62f8613a7812b03419e6b29617b74fd1",
    "license": "MIT",
}


class DatasetError(RuntimeError):
    pass


def _download(source: Dict[str, str], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and file_sha256(destination) == source["sha256"]:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(source["url"], headers={"User-Agent": "greenloop-carbonbench/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except Exception as exc:
        raise DatasetError(f"Could not download {source['url']}: {exc}") from exc
    observed = file_sha256(temporary)
    if observed != source["sha256"]:
        raise DatasetError(
            f"Checksum mismatch for {source['url']}: expected {source['sha256']}, got {observed}"
        )
    temporary.replace(destination)
    return destination


def _gsm8k_prompts(path: Path) -> List[Dict[str, Any]]:
    results = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        match = re.search(r"####\s*([^\n]+)\s*$", row["answer"])
        if not match:
            continue
        answer = match.group(1).replace(",", "").strip()
        results.append(
            {
                "id": f"gsm8k-test-{index:04d}",
                "category": "mathematical_reasoning",
                "prompt": (
                    "Solve the problem. Show concise reasoning, then write exactly one line in the form "
                    "FINAL: <number>.\n\nProblem:\n" + row["question"]
                ),
                "source": "GSM8K test split",
                "license": GSM8K["license"],
                "grader": {"type": "numeric", "answer": answer, "tolerance": 1e-9},
                "metadata": {"source_index": index, "source_commit": GSM8K["commit"]},
            }
        )
    return results


def _squad_prompts(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = []
    index = 0
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = sorted({item["text"] for item in qa.get("answers", [])})
                if not answers:
                    continue
                results.append(
                    {
                        "id": f"squad-dev-{index:05d}",
                        "category": "grounded_question_answering",
                        "prompt": (
                            "Answer only from the supplied context. If the answer is absent, write "
                            "FINAL: INSUFFICIENT_EVIDENCE. Otherwise end with exactly one line in the form "
                            "FINAL: <answer>.\n\nCONTEXT:\n"
                            + context
                            + "\n\nQUESTION:\n"
                            + qa["question"]
                        ),
                        "source": "SQuAD v1.1 development split",
                        "license": SQUAD["license"],
                        "grader": {"type": "squad_f1", "answers": answers},
                        "metadata": {
                            "source_id": qa["id"],
                            "source_index": index,
                            "source_commit": SQUAD["commit"],
                            "context_chars": len(context),
                        },
                    }
                )
                index += 1
    return results


def _controlled_prompts(count: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    retrieval_count = (count + 1) // 2
    lengths = [128, 512, 1024, 2048]
    for index in range(retrieval_count):
        target_words = lengths[index % len(lengths)]
        sentinel = f"GL-{index:03d}-{(index * 7919) % 100000:05d}"
        filler_sentence = "A storage site logs wind, solar, demand, and battery observations every interval. "
        filler = (filler_sentence * ((target_words // len(filler_sentence.split())) + 2)).split()
        before = " ".join(filler[: max(1, target_words // 2)])
        after = " ".join(filler[max(1, target_words // 2) : target_words])
        context = f"{before}\nThe audit marker is {sentinel}.\n{after}"
        results.append(
            {
                "id": f"controlled-retrieval-{index:03d}",
                "category": "controlled_long_context",
                "prompt": (
                    "Read the context and recover the audit marker. Do not infer or alter it. "
                    "Write exactly one line: FINAL: <audit marker>.\n\nCONTEXT:\n" + context
                ),
                "source": "GreenLoop controlled generator v1",
                "license": "CC0-1.0",
                "grader": {"type": "exact", "answers": [sentinel]},
                "metadata": {"target_words": target_words, "sentinel": sentinel},
            }
        )
    decode_budgets = [32, 64, 128, 256]
    for index in range(count - retrieval_count):
        budget = decode_budgets[index % len(decode_budgets)]
        results.append(
            {
                "id": f"controlled-decode-{index:03d}",
                "category": "controlled_decode",
                "prompt": (
                    "Generate a long sequence of short, distinct observations about an imaginary renewable "
                    "energy system. Keep writing until the system stops you. Do not summarize, conclude, or "
                    "finish early. Start directly with observation 1."
                ),
                "source": "GreenLoop controlled generator v1",
                "license": "CC0-1.0",
                "grader": {"type": "none"},
                "metadata": {
                    "requested_output_cap": budget,
                    "num_predict_override": budget,
                    "measurement_only": True,
                },
            }
        )
    return results


def _stratified_squad_sample(rows: List[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {"short": [], "medium": [], "long": []}
    for row in rows:
        chars = int(row["metadata"]["context_chars"])
        bucket = "short" if chars < 600 else "medium" if chars < 1200 else "long"
        row["metadata"]["context_bucket"] = bucket
        buckets[bucket].append(row)
    sample: List[Dict[str, Any]] = []
    allocation = [count // 3, count // 3, count - 2 * (count // 3)]
    for bucket_name, wanted in zip(("short", "medium", "long"), allocation):
        pool = buckets[bucket_name]
        if wanted > len(pool):
            raise DatasetError(f"Not enough SQuAD rows in {bucket_name} bucket")
        sample.extend(rng.sample(pool, wanted))
    return sample


def build_prompt_suite(
    output: Path,
    raw_directory: Path,
    seed: int = 20260928,
    gsm8k_count: int = 35,
    squad_count: int = 45,
    controlled_count: int = 20,
) -> Dict[str, Any]:
    if min(gsm8k_count, squad_count, controlled_count) < 0:
        raise DatasetError("Prompt counts cannot be negative")
    gsm_path = _download(GSM8K, raw_directory / "gsm8k-test.jsonl")
    squad_path = _download(SQUAD, raw_directory / "squad-dev-v1.1.json")
    rng = random.Random(seed)
    gsm_rows = _gsm8k_prompts(gsm_path)
    squad_rows = _squad_prompts(squad_path)
    if gsm8k_count > len(gsm_rows):
        raise DatasetError("Requested more GSM8K prompts than are available")
    selected = rng.sample(gsm_rows, gsm8k_count)
    selected.extend(_stratified_squad_sample(squad_rows, squad_count, rng))
    selected.extend(_controlled_prompts(controlled_count))
    # Lock approximately 20% inside each category before the final shuffle.
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for row in selected:
        by_category.setdefault(row["category"], []).append(row)
    for rows in by_category.values():
        rng.shuffle(rows)
        locked_count = max(1, round(len(rows) * 0.20)) if len(rows) >= 5 else 0
        for index, row in enumerate(rows):
            row["metadata"]["benchmark_split"] = "locked" if index < locked_count else "development"
    rng.shuffle(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "output_filename": output.name,
        "total": len(selected),
        "seed": seed,
        "counts": {
            "gsm8k": gsm8k_count,
            "squad": squad_count,
            "controlled": controlled_count,
        },
        "raw_sources": {
            "gsm8k": {**GSM8K, "cached_filename": gsm_path.name},
            "squad": {**SQUAD, "cached_filename": squad_path.name},
        },
        "prompt_file_sha256": file_sha256(output),
    }
