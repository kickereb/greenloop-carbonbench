from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerateResult:
    payload: Dict[str, Any]
    response: str
    prompt_eval_count: Optional[int]
    prompt_eval_cached_count: Optional[int]
    eval_count: Optional[int]
    total_duration_ns: Optional[int]
    load_duration_ns: Optional[int]
    prompt_eval_duration_ns: Optional[int]
    eval_duration_ns: Optional[int]
    done_reason: Optional[str]
    first_token_duration_s: Optional[float]
    thinking: str


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 900):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _json(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", "User-Agent": "greenloop-carbonbench/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code} for {path}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama request failed for {path}: {exc}") from exc

    def version(self) -> str:
        return str(self._json("GET", "/api/version").get("version", "unknown"))

    def tags(self) -> List[Dict[str, Any]]:
        return list(self._json("GET", "/api/tags").get("models", []))

    def ps(self) -> List[Dict[str, Any]]:
        return list(self._json("GET", "/api/ps").get("models", []))

    def show(self, model: str) -> Dict[str, Any]:
        return self._json("POST", "/api/show", {"model": model, "verbose": True})

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float,
        seed: int,
        num_predict: int,
        num_ctx: int,
        keep_alive: str,
        reasoning: bool = False,
        raw: bool = False,
    ) -> GenerateResult:
        request_payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }
        if raw:
            request_payload["raw"] = True
        if not reasoning:
            request_payload["think"] = False
        body = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/api/generate",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "greenloop-carbonbench/0.1"},
        )
        started = time.perf_counter()
        first_token_s: Optional[float] = None
        response_parts: List[str] = []
        thinking_parts: List[str] = []
        payload: Dict[str, Any] = {}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as stream:
                for raw_line in stream:
                    if not raw_line.strip():
                        continue
                    chunk = json.loads(raw_line.decode("utf-8"))
                    text = str(chunk.get("response", ""))
                    thinking_text = str(chunk.get("thinking", ""))
                    if first_token_s is None and (text or thinking_text):
                        first_token_s = time.perf_counter() - started
                    response_parts.append(text)
                    thinking_parts.append(thinking_text)
                    if chunk.get("done"):
                        payload = chunk
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code} for /api/generate: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama streaming request failed: {exc}") from exc
        if not payload:
            raise OllamaError("Ollama stream ended without a final telemetry record")

        def integer(name: str) -> Optional[int]:
            value = payload.get(name)
            return int(value) if isinstance(value, (int, float)) else None

        return GenerateResult(
            payload=payload,
            response="".join(response_parts),
            prompt_eval_count=integer("prompt_eval_count"),
            prompt_eval_cached_count=integer("prompt_eval_cached_count"),
            eval_count=integer("eval_count"),
            total_duration_ns=integer("total_duration"),
            load_duration_ns=integer("load_duration"),
            prompt_eval_duration_ns=integer("prompt_eval_duration"),
            eval_duration_ns=integer("eval_duration"),
            done_reason=(str(payload["done_reason"]) if payload.get("done_reason") is not None else None),
            first_token_duration_s=first_token_s,
            thinking="".join(thinking_parts),
        )

    def unload(self, model: str) -> None:
        self._json(
            "POST",
            "/api/generate",
            {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        )


def pull_models(model_names: Iterable[str], dry_run: bool = False) -> None:
    for model in model_names:
        command = ["ollama", "pull", model]
        if dry_run:
            print(" ".join(command))
            continue
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise OllamaError("The ollama executable is not on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise OllamaError(f"Could not pull {model} (exit {exc.returncode})") from exc
