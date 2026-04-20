# Observability Stack

This repo uses:

- `Prometheus` for metrics scraping and remote_write to Grafana Cloud.
- `Promtail` for container log shipping to Grafana Cloud Loki.
- The backend's existing structured JSON logs and per-stage metrics as the source of truth.

## Directory structure

```text
.
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   └── ...
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── ...
└── infra/
    └── monitoring/
        ├── README.md
        ├── grafana-cloud.env
        ├── prometheus.yml
        ├── prometheus-entrypoint.sh
        ├── promtail.yml
        └── promtail-entrypoint.sh
```

## Metrics plan

Prometheus scrapes `GET /metrics/prometheus` from the backend.

### Canonical counters

- `sovereign_sql_requests_total`
- `sovereign_sql_requests_failed_total`
- `sovereign_sql_requests_guard_blocked_total`
- `sovereign_sql_requests_out_of_topic_total`
- `sovereign_sql_classification_total{label=...}`

### Latency histograms

- `sovereign_sql_guard_ms`
- `sovereign_sql_classifier_ms`
- `sovereign_sql_pinecone_ms`
- `sovereign_sql_neo4j_ms`
- `sovereign_sql_metadata_ms`
- `sovereign_sql_runpod_ms`
- `sovereign_sql_total_ms`

### Standard process / runtime metrics

- `sovereign_sql_process_cpu_seconds_total`
- `sovereign_sql_process_resident_memory_bytes`
- `sovereign_sql_process_threads`
- `sovereign_sql_process_open_fds`
- `sovereign_sql_system_load1`
- `sovereign_sql_system_load5`
- `sovereign_sql_system_load15`
- `sovereign_sql_gpu_utilization_percent{index,device}`
- `sovereign_sql_gpu_memory_utilization_percent{index,device}`
- `sovereign_sql_gpu_memory_used_bytes{index,device}`
- `sovereign_sql_gpu_memory_total_bytes{index,device}`

Each histogram exports:

- `_bucket`
- `_sum`
- `_count`

### Useful future metrics

These are worth adding next if you want deeper operational visibility:

- `sovereign_sql_execution_rows_returned`
- `sovereign_sql_feedback_total`
- `sovereign_sql_service_failures_total{service=...}`
- `sovereign_sql_sse_events_total{event=...}`
- `sovereign_sql_db_write_failures_total{table=...}`

## Logging plan

The backend already emits structured JSON logs to stdout. Promtail tails container logs and ships them to Loki.

### Required log fields

- `timestamp`
- `level`
- `logger`
- `message`
- `service_name`
- `environment`
- `request_id`
- `trace_id`
- `span_id`
- `span_name`
- `filename`
- `lineno`

### Recommended event names

- `pipeline.request`
- `pipeline.start`
- `pipeline.done`
- `pipeline.timeout`
- `pipeline.error`
- `sse.pipeline.request`
- `sse.pipeline.done`
- `sse.client_disconnected`
- `guard.start`
- `guard.done`
- `classifier.start`
- `classifier.done`
- `pinecone.start`
- `pinecone.done`
- `neo4j.start`
- `neo4j.done`
- `metadata.start`
- `metadata.done`
- `runpod.start`
- `runpod.done`
- `sql_execute.start`
- `sql_execute.done`
- `request.validation_error`
- `sse.guard_blocked`
- `sse.out_of_topic`
- `sse.pinecone.timeout`
- `sse.retrieval_chain.error`

### Labels to keep in Loki

Low-cardinality labels only:

- `service`
- `container`
- `level`
- `logger`
- `service_name`
- `span_name`
- `stream`

Do not label `request_id` or `trace_id`; keep those as fields for searching.

## Grafana Cloud wiring

Set the values in `infra/monitoring/grafana-cloud.env` to the exact ingest URLs from the Grafana Cloud portal:

- `GRAFANA_CLOUD_PROM_REMOTE_WRITE_URL` should be the Prometheus metrics endpoint, typically ending in `/api/prom/push`
- `GRAFANA_CLOUD_PROM_USERNAME`
- `GRAFANA_CLOUD_PROM_API_KEY`
- `GRAFANA_CLOUD_LOKI_URL` should be the Loki push endpoint, typically ending in `/loki/api/v1/push`
- `GRAFANA_CLOUD_LOKI_USERNAME`
- `GRAFANA_CLOUD_LOKI_API_KEY`

If those values are empty, the containers still start locally, but they will not ship data to Grafana Cloud.

## Run

```bash
docker compose up --build
```

Backend:

- `http://localhost:8000`

Frontend:

- `http://localhost:3000`

Prometheus UI:

- `http://localhost:9090`
