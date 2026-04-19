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
        with self._label_lock:
            labels = dict(self._label_counts)
        return {
            "uptime_seconds": round(time.time() - self._start, 1),
            "requests": {
                "total": self.requests_total.value,
                "failed": self.requests_failed.value,
                "guard_blocked": self.requests_guard_blocked.value,
                "out_of_topic": self.requests_out_of_topic.value,
                "by_classification": labels,
            },
            "latency_histograms": {
                "guard_ms":      self.guard_latency.snapshot(),
                "classifier_ms": self.classifier_latency.snapshot(),
                "pinecone_ms":   self.pinecone_latency.snapshot(),
                "neo4j_ms":      self.neo4j_latency.snapshot(),
                "metadata_ms":   self.metadata_latency.snapshot(),
                "runpod_ms":     self.runpod_latency.snapshot(),
                "total_ms":      self.total_latency.snapshot(),
            },
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
