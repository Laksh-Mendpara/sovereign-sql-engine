"""Thread-safe in-process metrics collector.

Tracks:
  - Request counters (total, failed, guard-blocked, out-of-topic, by-label)
  - Per-stage latency histograms with approximate percentiles
  - Global uptime

Exposed at GET /metrics as a JSON snapshot.
"""

from __future__ import annotations

import threading
import time
import os
import subprocess
import resource
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from .logging_config import clear_span_context, get_logger, new_span_id, set_span_context

logger = get_logger(__name__)

# Fixed bucket upper-bounds in milliseconds
_BUCKETS_MS = (10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, float("inf"))


# ── Latency histogram ──────────────────────────────────────────────────────────

@dataclass
class LatencyHistogram:
    name: str
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _count: int = field(default=0, init=False)
    _sum_ms: float = field(default=0.0, init=False)
    _buckets: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self._buckets = [0] * len(_BUCKETS_MS)

    def observe(self, latency_ms: float) -> None:
        with self._lock:
            self._count += 1
            self._sum_ms += latency_ms
            for i, bound in enumerate(_BUCKETS_MS):
                if latency_ms <= bound:
                    self._buckets[i] += 1
                    break

    def snapshot(self) -> dict:
        with self._lock:
            count = self._count
            sum_ms = self._sum_ms
            buckets = list(self._buckets)

        avg = (sum_ms / count) if count else 0.0
        return {
            "count": count,
            "sum_ms": round(sum_ms, 2),
            "avg_ms": round(avg, 2),
            "p50_ms": self._approx_percentile(0.50, count, buckets),
            "p95_ms": self._approx_percentile(0.95, count, buckets),
            "p99_ms": self._approx_percentile(0.99, count, buckets),
            "buckets": {
                (f"le_{b}ms" if b != float("inf") else "le_inf"): buckets[i]
                for i, b in enumerate(_BUCKETS_MS)
            },
        }

    @staticmethod
    def _approx_percentile(p: float, count: int, buckets: list[int]) -> float:
        if count == 0:
            return 0.0
        target = max(1, int(p * count))
        cumulative = 0
        prev = 0.0
        for i, bound in enumerate(_BUCKETS_MS):
            cumulative += buckets[i]
            if cumulative >= target:
                hi = prev * 2 if bound == float("inf") else float(bound)
                n = buckets[i]
                frac = (target - (cumulative - n)) / n if n else 0.0
                return round(prev + frac * (hi - prev), 2)
            prev = float(bound) if bound != float("inf") else prev
        return round(prev, 2)


# ── Simple counter ─────────────────────────────────────────────────────────────

@dataclass
class Counter:
    name: str
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _val: int = field(default=0, init=False)

    def inc(self, by: int = 1) -> None:
        with self._lock:
            self._val += by

    @property
    def value(self) -> int:
        with self._lock:
            return self._val


# ── Per-request stage timer ────────────────────────────────────────────────────

@dataclass
class StageTimer:
    """Accumulates wall-clock timings for each pipeline stage in a single request."""

    _timings: dict[str, float] = field(default_factory=dict, init=False)
    _t0: float = field(default_factory=time.perf_counter, init=False)

    def record(self, stage: str, elapsed_ms: float) -> None:
        self._timings[stage] = round(elapsed_ms, 2)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1_000, 2)

    def as_dict(self) -> dict[str, float]:
        return {**self._timings, "total_ms": self.total_ms}


# ── Global singleton ───────────────────────────────────────────────────────────

class MetricsCollector:
    """Process-lifetime singleton — one instance per worker."""

    def __init__(self) -> None:
        self._start = time.time()

        # Counters
        self.requests_total = Counter("requests_total")
        self.requests_failed = Counter("requests_failed")
        self.requests_guard_blocked = Counter("requests_guard_blocked")
        self.requests_out_of_topic = Counter("requests_out_of_topic")

        # Per-classification-label counter
        self._label_lock = threading.Lock()
        self._label_counts: dict[str, int] = {"easy": 0, "difficult": 0, "out_of_topic": 0}

        # Per-stage histograms
        self.guard_latency = LatencyHistogram("guard")
        self.classifier_latency = LatencyHistogram("classifier")
        self.pinecone_latency = LatencyHistogram("pinecone")
        self.neo4j_latency = LatencyHistogram("neo4j")
        self.metadata_latency = LatencyHistogram("metadata")
        self.runpod_latency = LatencyHistogram("runpod")
        self.total_latency = LatencyHistogram("total")

    def inc_label(self, label: str) -> None:
        with self._label_lock:
            self._label_counts[label] = self._label_counts.get(label, 0) + 1

    def snapshot(self) -> dict:
        return _collector_snapshot(self)

def _system_snapshot() -> dict[str, float | int | list[dict[str, float | int]]]:
    rss_bytes = 0
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_bytes = int(usage.ru_maxrss * 1024)
    except Exception:
        rss_bytes = 0

    cpu_seconds = round(time.process_time(), 2)
    thread_count = threading.active_count()
    open_fds = None
    try:
        open_fds = len(os.listdir("/proc/self/fd"))
    except Exception:
        open_fds = None

    load1 = load5 = load15 = None
    try:
        load1, load5, load15 = os.getloadavg()
    except Exception:
        pass

    gpu_metrics: list[dict[str, float | int]] = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6:
                continue
            gpu_metrics.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_gpu": float(parts[2]),
                    "utilization_memory": float(parts[3]),
                    "memory_used_mb": float(parts[4]),
                    "memory_total_mb": float(parts[5]),
                }
            )
    except Exception:
        gpu_metrics = []

    return {
        "cpu_seconds": cpu_seconds,
        "rss_bytes": rss_bytes,
        "thread_count": thread_count,
        "open_fds": open_fds if open_fds is not None else 0,
        "load1": round(load1, 2) if load1 is not None else 0.0,
        "load5": round(load5, 2) if load5 is not None else 0.0,
        "load15": round(load15, 2) if load15 is not None else 0.0,
        "gpu": gpu_metrics,
    }


_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector


def _collector_snapshot(collector: MetricsCollector) -> dict:
    with collector._label_lock:
        labels = dict(collector._label_counts)
    return {
        "uptime_seconds": round(time.time() - collector._start, 1),
        "system": _system_snapshot(),
        "requests": {
            "total": collector.requests_total.value,
            "failed": collector.requests_failed.value,
            "guard_blocked": collector.requests_guard_blocked.value,
            "out_of_topic": collector.requests_out_of_topic.value,
            "by_classification": labels,
        },
        "latency_histograms": {
            "guard_ms":      collector.guard_latency.snapshot(),
            "classifier_ms": collector.classifier_latency.snapshot(),
            "pinecone_ms":   collector.pinecone_latency.snapshot(),
            "neo4j_ms":      collector.neo4j_latency.snapshot(),
            "metadata_ms":   collector.metadata_latency.snapshot(),
            "runpod_ms":     collector.runpod_latency.snapshot(),
            "total_ms":      collector.total_latency.snapshot(),
        },
    }


def _prom_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def render_prometheus_metrics() -> str:
    """Render the in-process metrics snapshot in Prometheus exposition format."""
    snapshot = _collector_snapshot(get_metrics())
    lines: list[str] = []

    def add_help(name: str, help_text: str) -> None:
        lines.append(f"# HELP {name} {help_text}")

    def add_type(name: str, metric_type: str) -> None:
        lines.append(f"# TYPE {name} {metric_type}")

    add_help("sovereign_sql_requests_total", "Total number of pipeline requests.")
    add_type("sovereign_sql_requests_total", "counter")
    lines.append(f"sovereign_sql_requests_total {snapshot['requests']['total']}")

    add_help("sovereign_sql_requests_failed_total", "Total number of failed pipeline requests.")
    add_type("sovereign_sql_requests_failed_total", "counter")
    lines.append(f"sovereign_sql_requests_failed_total {snapshot['requests']['failed']}")

    add_help("sovereign_sql_requests_guard_blocked_total", "Total number of requests blocked by guardrail.")
    add_type("sovereign_sql_requests_guard_blocked_total", "counter")
    lines.append(f"sovereign_sql_requests_guard_blocked_total {snapshot['requests']['guard_blocked']}")

    add_help("sovereign_sql_requests_out_of_topic_total", "Total number of requests classified out of topic.")
    add_type("sovereign_sql_requests_out_of_topic_total", "counter")
    lines.append(f"sovereign_sql_requests_out_of_topic_total {snapshot['requests']['out_of_topic']}")

    add_help("sovereign_sql_uptime_seconds", "Process uptime in seconds.")
    add_type("sovereign_sql_uptime_seconds", "gauge")
    lines.append(f"sovereign_sql_uptime_seconds {snapshot['uptime_seconds']}")

    system = snapshot.get("system", {})
    add_help("sovereign_sql_process_cpu_seconds_total", "Total CPU seconds used by the process.")
    add_type("sovereign_sql_process_cpu_seconds_total", "counter")
    lines.append(f"sovereign_sql_process_cpu_seconds_total {system.get('cpu_seconds', 0)}")

    add_help("sovereign_sql_process_resident_memory_bytes", "Resident memory size in bytes.")
    add_type("sovereign_sql_process_resident_memory_bytes", "gauge")
    lines.append(f"sovereign_sql_process_resident_memory_bytes {system.get('rss_bytes', 0)}")

    add_help("sovereign_sql_process_threads", "Number of active threads.")
    add_type("sovereign_sql_process_threads", "gauge")
    lines.append(f"sovereign_sql_process_threads {system.get('thread_count', 0)}")

    add_help("sovereign_sql_process_open_fds", "Number of open file descriptors.")
    add_type("sovereign_sql_process_open_fds", "gauge")
    lines.append(f"sovereign_sql_process_open_fds {system.get('open_fds', 0)}")

    add_help("sovereign_sql_system_load1", "1-minute load average.")
    add_type("sovereign_sql_system_load1", "gauge")
    lines.append(f"sovereign_sql_system_load1 {system.get('load1', 0.0)}")

    add_help("sovereign_sql_system_load5", "5-minute load average.")
    add_type("sovereign_sql_system_load5", "gauge")
    lines.append(f"sovereign_sql_system_load5 {system.get('load5', 0.0)}")

    add_help("sovereign_sql_system_load15", "15-minute load average.")
    add_type("sovereign_sql_system_load15", "gauge")
    lines.append(f"sovereign_sql_system_load15 {system.get('load15', 0.0)}")

    gpu_metrics = system.get("gpu", []) if isinstance(system, dict) else []
    if gpu_metrics:
        add_help("sovereign_sql_gpu_utilization_percent", "GPU utilization percent by device.")
        add_type("sovereign_sql_gpu_utilization_percent", "gauge")
        add_help("sovereign_sql_gpu_memory_utilization_percent", "GPU memory utilization percent by device.")
        add_type("sovereign_sql_gpu_memory_utilization_percent", "gauge")
        add_help("sovereign_sql_gpu_memory_used_bytes", "GPU memory used in bytes by device.")
        add_type("sovereign_sql_gpu_memory_used_bytes", "gauge")
        add_help("sovereign_sql_gpu_memory_total_bytes", "GPU memory total in bytes by device.")
        add_type("sovereign_sql_gpu_memory_total_bytes", "gauge")
        for gpu in gpu_metrics:
            index = gpu.get("index", 0)
            name = _prom_escape(str(gpu.get("name", "unknown")))
            util_gpu = gpu.get("utilization_gpu", 0.0)
            util_mem = gpu.get("utilization_memory", 0.0)
            mem_used_bytes = float(gpu.get("memory_used_mb", 0.0)) * 1024 * 1024
            mem_total_bytes = float(gpu.get("memory_total_mb", 0.0)) * 1024 * 1024
            lines.append(
                f'sovereign_sql_gpu_utilization_percent{{index="{index}",device="{name}"}} {util_gpu}'
            )
            lines.append(
                f'sovereign_sql_gpu_memory_utilization_percent{{index="{index}",device="{name}"}} {util_mem}'
            )
            lines.append(
                f'sovereign_sql_gpu_memory_used_bytes{{index="{index}",device="{name}"}} {mem_used_bytes}'
            )
            lines.append(
                f'sovereign_sql_gpu_memory_total_bytes{{index="{index}",device="{name}"}} {mem_total_bytes}'
            )

    add_help("sovereign_sql_classification_total", "Requests grouped by classification label.")
    add_type("sovereign_sql_classification_total", "counter")
    for label, value in snapshot["requests"]["by_classification"].items():
        lines.append(
            f'sovereign_sql_classification_total{{label="{_prom_escape(str(label))}"}} {value}'
        )

    stage_histograms = snapshot["latency_histograms"]
    bucket_bounds = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    for stage_name, data in stage_histograms.items():
        metric_name = f"sovereign_sql_{stage_name}"
        add_help(metric_name, f"Latency histogram for {stage_name}.")
        add_type(metric_name, "histogram")
        cumulative = 0
        buckets = data["buckets"]
        for bound in bucket_bounds:
            key = f"le_{bound}ms"
            cumulative += int(buckets.get(key, 0))
            lines.append(
                f'{metric_name}_bucket{{stage="{_prom_escape(stage_name)}",le="{bound}"}} {cumulative}'
            )
        lines.append(f'{metric_name}_bucket{{stage="{_prom_escape(stage_name)}",le="+Inf"}} {int(data["count"])}')
        lines.append(f'{metric_name}_sum{{stage="{_prom_escape(stage_name)}"}} {data["sum_ms"]}')
        lines.append(f'{metric_name}_count{{stage="{_prom_escape(stage_name)}"}} {int(data["count"])}')

    return "\n".join(lines) + "\n"


# ── timed_span context manager ─────────────────────────────────────────────────

@contextmanager
def timed_span(
    name: str,
    stage_timer: StageTimer,
    histogram: LatencyHistogram | None = None,
) -> Generator[str, None, None]:
    """Time a block, set span context for logs, record metrics.

    Usage::

        with timed_span("guard_ms", timer, metrics.guard_latency) as span_id:
            result = do_work()
    """
    span_id = new_span_id()
    set_span_context(span_id, name)
    t0 = time.perf_counter()
    logger.debug("span.start", extra={"span_name": name, "span_id": span_id})
    try:
        yield span_id
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        stage_timer.record(name, elapsed_ms)
        if histogram is not None:
            histogram.observe(elapsed_ms)
        logger.debug(
            "span.end",
            extra={"span_name": name, "span_id": span_id, "elapsed_ms": round(elapsed_ms, 2)},
        )
        clear_span_context()
