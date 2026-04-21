# Observability & Monitoring

The Sovereign SQL Engine is built with high-fidelity monitoring at its core, enabling real-time performance tracking and rapid troubleshooting using a standard Grafana Cloud stack.

## Metrics (Prometheus)

All internal pipeline timings and success rates are exposed via a Prometheus-compatible `/metrics/prometheus` endpoint on the backend.

### Key Metric Groups

#### 1. Performance Histograms (Latency)
Measured in milliseconds:
- `sovereign_sql_guard_ms_bucket`: Time spent inside Llama Guard safety check.
- `sovereign_sql_classifier_ms_bucket`: Time spent classifying query difficulty.
- `sovereign_sql_pinecone_ms_bucket`: Vector retrieval latency.
- `sovereign_sql_runpod_ms_bucket`: LLM SQL generation time.
- `sovereign_sql_total_ms_bucket`: End-to-end user perceived latency.

#### 2. Pipeline Success & Logic Gauges
- `sovereign_sql_requests_total`: Total queries received.
- `sovereign_sql_requests_failed_total`: Number of queries that resulted in a `pipeline.error`.
- `sovereign_sql_requests_guard_blocked_total`: Safely rejected inputs.
- `sovereign_sql_classification_total{label="easy|difficult|out_of_topic"}`: Categorization distribution.

#### 3. Infrastructure Health (External)
The backend proxies health status from external inference partners:
- `sovereign_sql_runpod_jobs{status="completed|failed|inProgress"}`: Real-time Runpod serverless queue depth.
- `sovereign_sql_runpod_workers{state="idle|ready|running"}`: Active GPU capacity tracking.

## Logging (Loki)

Logs are pushed to Grafana Loki through two channels:

1.  **Backend Logs**: Traditional FastAPI logging with structured JSON output, including `request_id` and `trace_id` for every line.
2.  **vLLM Inference Logs**: A custom **OTLP Relay** on the Modal nodes captures shell output from the vLLM engine and pushes it to Loki with rich metadata:
    - `service`: `phi4` or `llama-guard`.
    - `component`: `vllm-engine`.
    - `deployment`: `modal`.

## Audit & Persistence (SQLite Cloud)

Every request is durably recorded in the `request_records` table on SQLite Cloud for offline analysis:

- **Record Schema**: `request_id`, `query`, `guard_result`, `generated_sql`, `execution_status`, `error_message`, `raw_payload`.
- **Purpose**: Powering local diagnostics dashboards and providing a dataset for future model fine-tuning.

## Real-time Tracing

By utilizing distributed tracing headers, you can visualize a single user query from the moment it hits the React Frontend, through the FastAPI Backend, and into the Modal GPU workers.

- **Trace propagated via**: `X-Trace-Id` and `X-Request-Id`.
- **Visualization Tool**: Grafana Tempo (recommended) or direct Loki/Prometheus label filtering.
