import sys
import os
import logging
import multiprocessing
import traceback
import time
import threading
import runpod
from runpod import RunPodLogger

log = RunPodLogger()

vllm_engine = None
openai_engine = None

# ---------------------------------------------------------------------------
# OpenTelemetry Setup (Grafana Cloud OTLP) — always enabled
# ---------------------------------------------------------------------------
from opentelemetry import metrics as _otel_metrics, _logs
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry._logs import SeverityNumber

resource = Resource.create({
    "service.name": os.getenv("OTEL_SERVICE_NAME", "vllm-worker"),
})

# ── Logs ──────────────────────────────────────────────────────────────────
_logger_provider = LoggerProvider(resource=resource)
_logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
_logs.set_logger_provider(_logger_provider)

# Bridge: every logging.info / .warning / .error from vLLM, transformers,
# etc. automatically flows to Grafana Loki via OTLP.
logging.getLogger().addHandler(
    LoggingHandler(level=logging.DEBUG, logger_provider=_logger_provider)
)

# Direct OTLP logger for our own structured events
otel_logger = _logs.get_logger("vllm.handler")

# ── Metrics ───────────────────────────────────────────────────────────────
_metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(), export_interval_millis=15_000
)
_meter_provider = MeterProvider(resource=resource, metric_readers=[_metric_reader])
_otel_metrics.set_meter_provider(_meter_provider)
meter = _otel_metrics.get_meter("vllm.engine")

logging.info(
    "OTEL observability enabled → %s (service=%s)",
    os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    os.getenv("OTEL_SERVICE_NAME", "vllm-worker"),
)

# ---------------------------------------------------------------------------
# Metric instruments
# ---------------------------------------------------------------------------
m_request_latency = meter.create_histogram(
    "vllm.request_latency", unit="s",
    description="End-to-end request latency",
)
m_ttft = meter.create_histogram(
    "vllm.time_to_first_token", unit="s",
    description="Time to first token",
)
m_tpot = meter.create_histogram(
    "vllm.time_per_output_token", unit="s",
    description="Time per output token (decode phase / token count)",
)
m_prompt_tokens = meter.create_histogram(
    "vllm.prompt_tokens", unit="count",
    description="Number of prompt tokens",
)
m_output_tokens = meter.create_histogram(
    "vllm.output_tokens", unit="count",
    description="Number of generation tokens",
)
m_queue_time = meter.create_histogram(
    "vllm.time_in_queue", unit="s",
    description="Time spent waiting for inference",
)
m_running_reqs = meter.create_gauge(
    "vllm.requests.running", unit="count",
    description="Currently executing requests",
)
m_gpu_util = meter.create_gauge(
    "vllm.vram.utilization", unit="percent",
    description="KV-cache VRAM utilization",
)
m_cpu_util = meter.create_gauge(
    "vllm.cpu.utilization", unit="percent",
    description="CPU KV-cache block utilization",
)
m_cache_hits = meter.create_counter(
    "vllm.cache.hits", unit="count",
    description="Number of KV prefix-cache hits",
)


# ---------------------------------------------------------------------------
# Background thread: relay vLLM engine stats → OTEL gauges
# ---------------------------------------------------------------------------
def _vllm_metrics_relay():
    """Poll vLLM engine internals every 15 s and push to OTEL gauges."""
    while True:
        try:
            time.sleep(15)
            if vllm_engine is None or not hasattr(vllm_engine, "engine"):
                continue

            eng = vllm_engine.engine
            if hasattr(eng, "get_num_unfinished_requests"):
                m_running_reqs.set(eng.get_num_unfinished_requests())

            stat_logger = getattr(eng, "stat_logger", None)
            if stat_logger and hasattr(stat_logger, "metrics"):
                mets = stat_logger.metrics
                gpu = mets.get("gpu_cache_usage")
                cpu = mets.get("cpu_cache_usage")
                if gpu is not None:
                    m_gpu_util.set(gpu * 100.0)
                if cpu is not None:
                    m_cpu_util.set(cpu * 100.0)
        except Exception:
            pass  # never crash the background thread


# ---------------------------------------------------------------------------
# Helper to emit structured OTEL logs safely
# ---------------------------------------------------------------------------
def _emit_log(body: str, attributes: dict | None = None, error: bool = False):
    """Emit a structured log record to Grafana Loki via OTLP."""
    try:
        otel_logger.emit(
            body=body,
            severity_number=SeverityNumber.ERROR if error else SeverityNumber.INFO,
            severity_text="ERROR" if error else "INFO",
            attributes=attributes,
        )
    except Exception:
        pass  # never break the handler over telemetry


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
async def handler(job):
    req_start_time = time.monotonic()

    try:
        from utils import JobInput

        job_input = JobInput(job["input"])
        engine = openai_engine if job_input.openai_route else vllm_engine

        _emit_log(
            f"Started inference for job {job['id']}",
            {"job_id": job["id"], "openai_route": str(job_input.openai_route)},
        )

        results_generator = engine.generate(job_input)

        first_token_time = None
        token_count = 0

        async for batch in results_generator:
            if first_token_time is None:
                first_token_time = time.monotonic()
                m_ttft.record(
                    first_token_time - req_start_time, {"job_id": job["id"]}
                )
            else:
                token_count += 1

            yield batch

        req_end_time = time.monotonic()
        total_latency = req_end_time - req_start_time
        m_request_latency.record(total_latency, {"job_id": job["id"]})

        if token_count > 0 and first_token_time is not None:
            tpot = (req_end_time - first_token_time) / token_count
            m_tpot.record(tpot, {"job_id": job["id"]})
            m_output_tokens.record(token_count, {"job_id": job["id"]})

        _emit_log(
            f"Completed inference for job {job['id']}",
            {"job_id": job["id"], "latency_s": str(round(total_latency, 4))},
        )

    except Exception as e:
        error_str = str(e)
        full_traceback = traceback.format_exc()

        log.error(f"Error during inference: {error_str}")
        log.error(f"Full traceback:\n{full_traceback}")

        _emit_log(
            f"Inference error: {error_str}",
            {"job_id": job["id"], "traceback": full_traceback},
            error=True,
        )

        if "CUDA" in error_str or "cuda" in error_str:
            log.error("Terminating worker due to CUDA/GPU error")
            sys.exit(1)

        yield {"error": error_str}


# ---------------------------------------------------------------------------
# Main-process initialisation
# ---------------------------------------------------------------------------
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    try:
        from engine import vLLMEngine, OpenAIvLLMEngine

        vllm_engine = vLLMEngine()
        openai_engine = OpenAIvLLMEngine(vllm_engine)
        log.info("vLLM engines initialized successfully")

        # Start background metrics relay
        threading.Thread(target=_vllm_metrics_relay, daemon=True).start()
    except Exception as e:
        log.error(f"Worker startup failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda x: vllm_engine.max_concurrency if vllm_engine else 1,
            "return_aggregate_stream": True,
        }
    )
