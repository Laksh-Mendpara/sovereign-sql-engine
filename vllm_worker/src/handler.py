import sys
import os
import multiprocessing
import traceback
import time
import threading
import runpod
from runpod import RunPodLogger
import requests

from opentelemetry import trace, metrics, _logs
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

log = RunPodLogger()

vllm_engine = None
openai_engine = None

# OTEL Setup
resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "vllm-worker")})

# 1. Logs Exporter
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
_logs.set_logger_provider(logger_provider)
otel_logger = _logs.get_logger(__name__)

# 2. Metrics Exporter
# We use a 15-second export interval to maintain reasonable Grafana Cloud usage
metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=15000)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("vllm.engine")

# Define industry-standard SLIs
m_request_latency = meter.create_histogram("vllm.request_latency", unit="s", description="End-to-end request latency")
m_ttft = meter.create_histogram("vllm.time_to_first_token", unit="s", description="Time to first token")
m_tpot = meter.create_histogram("vllm.time_per_output_token", unit="s", description="Time per output token")
m_prompt_tokens = meter.create_histogram("vllm.prompt_tokens", unit="count", description="Number of prompt tokens")
m_output_tokens = meter.create_histogram("vllm.output_tokens", unit="count", description="Number of generation tokens")
m_queue_time = meter.create_histogram("vllm.time_in_queue", unit="s", description="Time spent waiting for inference")

# Gauges tracked in background thread
m_running_reqs = meter.create_gauge("vllm.requests.running", unit="count", description="Currently executing requests")
m_gpu_util = meter.create_gauge("vllm.vram.utilization", unit="percent", description="KV cache VRM utilization")
m_cpu_util = meter.create_gauge("vllm.cpu.utilization", unit="percent", description="CPU block utilization")
m_cache_hits = meter.create_counter("vllm.cache.hits", unit="count", description="Number of KV prefix cache hits")


def vllm_metrics_relay():
    """Background thread to poll vLLM engine stats and update gauges."""
    while True:
        try:
            time.sleep(15)
            if vllm_engine is not None and hasattr(vllm_engine, "engine"):
                # Safe access to AsyncLLMEngine internal stats if they are exposed
                # vLLM metrics hook (StatLogger) generally stores this, but we can 
                # extract simple stats easily from the engine stats or use HTTP if enabled.
                if hasattr(vllm_engine.engine, "get_num_unfinished_requests"):
                    m_running_reqs.set(vllm_engine.engine.get_num_unfinished_requests())
                
                # Fetch GPU cache usage from internal stat logger if available
                stat_logger = getattr(vllm_engine.engine, "stat_logger", None)
                if stat_logger and hasattr(stat_logger, "metrics"):
                    mets = stat_logger.metrics
                    if mets.get('gpu_cache_usage'):
                        m_gpu_util.set(mets['gpu_cache_usage'] * 100.0)
                    if mets.get('cpu_cache_usage'):
                        m_cpu_util.set(mets['cpu_cache_usage'] * 100.0)
        except Exception:
            pass # Suppress background thread errors


async def handler(job):
    req_start_time = time.monotonic()
    
    try:
        from utils import JobInput
        job_input = JobInput(job["input"])
        engine = openai_engine if job_input.openai_route else vllm_engine
        
        otel_logger.emit(
            body=f"Started inference for job {job['id']}",
            attributes={"job_id": job["id"], "openai_route": job_input.openai_route}
        )

        results_generator = engine.generate(job_input)
        
        first_token_time = None
        last_output = None
        token_count = 0
        
        async for batch in results_generator:
            if not first_token_time:
                first_token_time = time.monotonic()
                m_ttft.record(first_token_time - req_start_time, {"job_id": job["id"]})
            else:
                token_count += 1
            
            last_output = batch
            yield batch
            
        req_end_time = time.monotonic()
        m_request_latency.record(req_end_time - req_start_time, {"job_id": job["id"]})
        
        if token_count > 0 and first_token_time:
            tpot = (req_end_time - first_token_time) / token_count
            m_tpot.record(tpot, {"job_id": job["id"]})
            m_output_tokens.record(token_count, {"job_id": job["id"]})
        
        # When bypassing API server directly to vLLMEngine, the batch usually contains stats 
        # based on vLLM's `RequestOutput` metrics.
        # Fallback to simple dictionary parsing if using openai wrapper.
        otel_logger.emit(
            body=f"Completed inference for job {job['id']}",
            attributes={"job_id": job["id"], "latency_s": req_end_time - req_start_time}
        )
        
    except Exception as e:
        error_str = str(e)
        full_traceback = traceback.format_exc()

        log.error(f"Error during inference: {error_str}")
        log.error(f"Full traceback:\n{full_traceback}")
        
        otel_logger.emit(
            body=f"Inference error: {error_str}",
            attributes={"job_id": job["id"], "traceback": full_traceback, "error": True}
        )

        # CUDA errors = worker is broken, exit to let RunPod spin up a healthy one
        if "CUDA" in error_str or "cuda" in error_str:
            log.error("Terminating worker due to CUDA/GPU error")
            sys.exit(1)

        yield {"error": error_str}


# Only run in main process to prevent re-initialization when vLLM spawns worker subprocesses
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    try:
        from engine import vLLMEngine, OpenAIvLLMEngine

        vllm_engine = vLLMEngine()
        openai_engine = OpenAIvLLMEngine(vllm_engine)
        log.info("vLLM engines initialized successfully")
        
        # Start background metrics relay
        threading.Thread(target=vllm_metrics_relay, daemon=True).start()
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
