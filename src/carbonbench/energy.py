from __future__ import annotations

import math
import os
import plistlib
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class EnergyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PowerSample:
    """One powermetrics interval; monotonic_s denotes the approximate interval end."""

    monotonic_s: float
    wall_time_s: float
    elapsed_ns: int
    cpu_mw: Optional[float]
    gpu_mw: Optional[float]
    ane_mw: Optional[float]
    combined_mw: Optional[float]
    invalid: bool = False
    thermal_pressure: Optional[str] = None

    @property
    def start_monotonic_s(self) -> float:
        return self.monotonic_s - self.elapsed_ns / 1_000_000_000.0

    @property
    def soc_mw(self) -> Optional[float]:
        if self.combined_mw is not None:
            return self.combined_mw
        values = [value for value in (self.cpu_mw, self.gpu_mw, self.ane_mw) if value is not None]
        return sum(values) if values else None


@dataclass(frozen=True)
class EnergyWindow:
    sample_count: int
    invalid_sample_count: int
    covered_seconds: float
    sample_coverage: float
    mean_cpu_mw: Optional[float]
    mean_gpu_mw: Optional[float]
    mean_ane_mw: Optional[float]
    mean_soc_mw: Optional[float]
    baseline_soc_mw: Optional[float]
    gross_soc_j: Optional[float]
    incremental_soc_j: Optional[float]
    gross_soc_wh: Optional[float]
    incremental_soc_wh: Optional[float]


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _recursive_value(value: Any, wanted: str) -> Any:
    if isinstance(value, dict):
        if wanted in value:
            return value[wanted]
        for child in value.values():
            found = _recursive_value(child, wanted)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _recursive_value(child, wanted)
            if found is not None:
                return found
    return None


def _power_value(payload: Dict[str, Any], key: str) -> Optional[float]:
    processor = payload.get("processor")
    if isinstance(processor, dict):
        direct = _finite_number(processor.get(key))
        if direct is not None:
            return direct
    direct = _finite_number(payload.get(key))
    if direct is not None:
        return direct
    return _finite_number(_recursive_value(payload, key))


def parse_plist_document(
    document: bytes,
    *,
    end_monotonic_s: Optional[float] = None,
    wall_time_s: Optional[float] = None,
) -> PowerSample:
    try:
        payload = plistlib.loads(document)
    except Exception as exc:
        raise EnergyError(f"Invalid powermetrics plist document: {exc}") from exc
    if not isinstance(payload, dict):
        raise EnergyError("powermetrics plist root is not a dictionary")
    elapsed_value = payload.get("elapsed_ns")
    if elapsed_value is None:
        elapsed_value = _recursive_value(payload, "elapsed_ns")
    elapsed_number = _finite_number(elapsed_value)
    if elapsed_number is None or elapsed_number <= 0:
        raise EnergyError("powermetrics sample has no positive elapsed_ns")
    processor = payload.get("processor", {})
    invalid = bool(payload.get("invalid")) or (
        isinstance(processor, dict) and bool(processor.get("invalid"))
    )
    thermal = payload.get("thermal_pressure")
    if thermal is None:
        thermal = _recursive_value(payload, "thermal_pressure")
    received_monotonic = time.monotonic()
    received_wall = time.time()
    plist_timestamp = payload.get("timestamp")
    parsed_wall: Optional[float] = None
    if isinstance(plist_timestamp, datetime):
        if plist_timestamp.tzinfo is None:
            plist_timestamp = plist_timestamp.replace(tzinfo=timezone.utc)
        parsed_wall = plist_timestamp.timestamp()
    elif isinstance(plist_timestamp, str):
        try:
            parsed = datetime.fromisoformat(plist_timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed_wall = parsed.timestamp()
        except ValueError:
            parsed_wall = None
    effective_wall = wall_time_s if wall_time_s is not None else (parsed_wall or received_wall)
    # powermetrics plist timestamps can have only whole-second resolution. Never use
    # them as the live integration clock. The collector supplies a cumulative
    # high-resolution monotonic endpoint; standalone parsing uses receipt time.
    effective_monotonic = end_monotonic_s if end_monotonic_s is not None else received_monotonic
    return PowerSample(
        monotonic_s=effective_monotonic,
        wall_time_s=effective_wall,
        elapsed_ns=int(elapsed_number),
        cpu_mw=_power_value(payload, "cpu_power"),
        gpu_mw=_power_value(payload, "gpu_power"),
        ane_mw=_power_value(payload, "ane_power"),
        combined_mw=_power_value(payload, "combined_power"),
        invalid=invalid,
        thermal_pressure=(str(thermal) if thermal is not None else None),
    )


def parse_plist_stream(data: bytes) -> List[PowerSample]:
    """Parse a captured NUL-delimited plist stream, useful for offline verification."""
    documents = [chunk.strip() for chunk in data.split(b"\x00") if chunk.strip()]
    samples: List[PowerSample] = []
    end_s = 0.0
    for document in documents:
        sample = parse_plist_document(document, end_monotonic_s=end_s)
        end_s += sample.elapsed_ns / 1_000_000_000.0
        samples.append(
            PowerSample(
                monotonic_s=end_s,
                wall_time_s=end_s,
                elapsed_ns=sample.elapsed_ns,
                cpu_mw=sample.cpu_mw,
                gpu_mw=sample.gpu_mw,
                ane_mw=sample.ane_mw,
                combined_mw=sample.combined_mw,
                invalid=sample.invalid,
                thermal_pressure=sample.thermal_pressure,
            )
        )
    return samples


def _overlap_seconds(sample: PowerSample, start_s: float, end_s: float) -> float:
    return max(0.0, min(end_s, sample.monotonic_s) - max(start_s, sample.start_monotonic_s))


def summarize_window(
    samples: Iterable[PowerSample],
    start_s: float,
    end_s: float,
    sample_interval_ms: int,
    baseline_soc_mw: Optional[float],
) -> EnergyWindow:
    del sample_interval_ms  # actual elapsed_ns, not requested cadence, is canonical
    duration = max(0.0, end_s - start_s)
    selected = [sample for sample in samples if _overlap_seconds(sample, start_s, end_s) > 0]
    valid = [sample for sample in selected if not sample.invalid and sample.soc_mw is not None]
    invalid_count = len(selected) - len(valid)

    intervals = sorted(
        (max(start_s, sample.start_monotonic_s), min(end_s, sample.monotonic_s)) for sample in valid
    )
    covered = 0.0
    if intervals:
        left, right = intervals[0]
        for next_left, next_right in intervals[1:]:
            if next_left <= right:
                right = max(right, next_right)
            else:
                covered += right - left
                left, right = next_left, next_right
        covered += right - left
    coverage = min(1.0, covered / duration) if duration > 0 else 0.0

    def integrated(attribute: str) -> tuple:
        energy_j = 0.0
        seconds = 0.0
        for sample in valid:
            value = sample.soc_mw if attribute == "soc_mw" else getattr(sample, attribute)
            if value is None:
                continue
            overlap = _overlap_seconds(sample, start_s, end_s)
            energy_j += float(value) * overlap / 1000.0
            seconds += overlap
        return (energy_j if seconds else None), seconds

    cpu_j, cpu_seconds = integrated("cpu_mw")
    gpu_j, gpu_seconds = integrated("gpu_mw")
    ane_j, ane_seconds = integrated("ane_mw")
    gross_j, soc_seconds = integrated("soc_mw")

    def average(energy_j: Optional[float], seconds: float) -> Optional[float]:
        return energy_j * 1000.0 / seconds if energy_j is not None and seconds > 0 else None

    incremental_j = None
    if gross_j is not None and baseline_soc_mw is not None:
        incremental_j = gross_j - baseline_soc_mw * soc_seconds / 1000.0
    return EnergyWindow(
        sample_count=len(valid),
        invalid_sample_count=invalid_count,
        covered_seconds=covered,
        sample_coverage=coverage,
        mean_cpu_mw=average(cpu_j, cpu_seconds),
        mean_gpu_mw=average(gpu_j, gpu_seconds),
        mean_ane_mw=average(ane_j, ane_seconds),
        mean_soc_mw=average(gross_j, soc_seconds),
        baseline_soc_mw=baseline_soc_mw,
        gross_soc_j=gross_j,
        incremental_soc_j=incremental_j,
        gross_soc_wh=(gross_j / 3600.0 if gross_j is not None else None),
        incremental_soc_wh=(incremental_j / 3600.0 if incremental_j is not None else None),
    )


class NullCollector:
    raw_path: Optional[Path] = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def samples(self) -> List[PowerSample]:
        return []

    def baseline(self, start_s: float, end_s: float) -> Optional[float]:
        return None


class PowerMetricsCollector:
    """Continuously collect and preserve Apple powermetrics plist samples."""

    def __init__(self, sample_interval_ms: int = 500, raw_path: Optional[Path] = None):
        if sys.platform != "darwin":
            raise EnergyError("powermetrics is only available on macOS")
        executable = shutil.which("powermetrics") or "/usr/bin/powermetrics"
        if not os.path.exists(executable):
            raise EnergyError("powermetrics was not found")
        self.sample_interval_ms = sample_interval_ms
        self.raw_path = raw_path
        prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
        self.command = prefix + [
            executable,
            "--samplers",
            "cpu_power,gpu_power,ane_power,thermal",
            "--format",
            "plist",
            "--buffer-size",
            "0",
            "--sample-rate",
            str(sample_interval_ms),
            "--poweravg",
            "0",
            "--handle-invalid-values",
        ]
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._samples: List[PowerSample] = []
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._error: Optional[str] = None
        self._last_sample_end_monotonic: Optional[float] = None

    def start(self) -> None:
        if self._process is not None:
            return
        if self.raw_path is not None:
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        # On macOS, sudo tickets may be scoped to the parent process rather than
        # only the terminal.  A ticket created by the user's shell can therefore
        # be invisible to the sudo process launched below (especially through
        # caffeinate).  Validate from this process so both sudo calls share the
        # same parent.  The password prompt is handled by sudo on the controlling
        # terminal; CarbonBench never reads or stores the password.
        if os.geteuid() != 0:
            try:
                authorization = subprocess.run(["sudo", "-v"], check=False)
            except FileNotFoundError as exc:
                raise EnergyError("sudo was not found; powermetrics requires administrator access") from exc
            if authorization.returncode != 0:
                raise EnergyError("sudo authorization failed; powermetrics was not started")
        try:
            self._process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                # Keep the controlling terminal so macOS sudo can reuse the
                # credential established above. A separate process group still
                # lets stop() interrupt sudo and powermetrics together.
                preexec_fn=os.setpgrp,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise EnergyError(f"Could not start powermetrics: {exc}") from exc
        self._thread = threading.Thread(target=self._read_loop, name="powermetrics-reader", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=max(6.0, self.sample_interval_ms / 1000.0 * 5)):
            code = self._process.poll()
            self.stop()
            detail = self._error or f"no samples arrived (exit={code})"
            raise EnergyError(
                "powermetrics did not start: "
                + detail
                + ". Check that the password prompt succeeded and retry."
            )

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        buffer = b""
        raw_handle = self.raw_path.open("wb") if self.raw_path is not None else None
        try:
            while True:
                chunk = self._process.stdout.read(65536)
                if not chunk:
                    break
                if raw_handle is not None:
                    raw_handle.write(chunk)
                    raw_handle.flush()
                buffer += chunk
                while b"\x00" in buffer:
                    document, buffer = buffer.split(b"\x00", 1)
                    if document.strip():
                        self._accept_document(document)
            if buffer.strip().endswith(b"</plist>"):
                self._accept_document(buffer)
        finally:
            if raw_handle is not None:
                raw_handle.close()
            if self._process.stderr is not None:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace").strip()
                if stderr:
                    self._error = stderr[-1000:]

    def _accept_document(self, document: bytes) -> None:
        try:
            parsed = parse_plist_document(document)
        except EnergyError as exc:
            self._error = str(exc)
            return
        received = time.monotonic()
        with self._lock:
            endpoint = (
                received
                if self._last_sample_end_monotonic is None
                else self._last_sample_end_monotonic + parsed.elapsed_ns / 1_000_000_000.0
            )
            self._last_sample_end_monotonic = endpoint
            sample = PowerSample(
                monotonic_s=endpoint,
                wall_time_s=parsed.wall_time_s,
                elapsed_ns=parsed.elapsed_ns,
                cpu_mw=parsed.cpu_mw,
                gpu_mw=parsed.gpu_mw,
                ane_mw=parsed.ane_mw,
                combined_mw=parsed.combined_mw,
                invalid=parsed.invalid,
                thermal_pressure=parsed.thermal_pressure,
            )
            self._samples.append(sample)
        self._ready.set()

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._process = None

    def samples(self) -> List[PowerSample]:
        with self._lock:
            return list(self._samples)

    def baseline(self, start_s: float, end_s: float) -> Optional[float]:
        values = [
            sample.soc_mw
            for sample in self.samples()
            if _overlap_seconds(sample, start_s, end_s) > 0
            and not sample.invalid
            and sample.soc_mw is not None
        ]
        return statistics.median(values) if values else None
