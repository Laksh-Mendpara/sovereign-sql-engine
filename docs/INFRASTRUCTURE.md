# Infrastructure & Hosting

The Sovereign SQL Engine employs a modern, serverless AI infrastructure designed for maximum performance with minimum operational cost.

## Compute & Inference

The system splits inference between two cost-optimized platforms:

### 1. Modal (Primary LLM Logic)
Used for internal pipeline logic (Guardrails and Classification).

- **Inference Engine**: vLLM with `AWQ 4-bit` quantization for extreme speed and low memory footprint.
- **Hardware Specs**:
    - **Phi-4-mini Node**: NVIDIA L4 GPU (24GB VRAM). High throughput for classification.
    - **Llama Guard 3 Node**: NVIDIA T4 GPU (16GB VRAM). Reliable, low-cost safety auditing.
- **Cold-Start Optimization**:
    - **Memory Snapshots**: Entire container states (including GPU VRAM caches) are snapshotted. This reduces cold-start latency from >30s to <2s.
    - **Sleep/Wake Cycles**: When idle, nodes offload weights to CPU RAM using vLLM's sleep hooks, incurring zero GPU costs while remaining ready for instant wake-up.

### 2. RunPod (SQL Heavy Lifting)
Used for deep Reasoning/Generation tasks (Arctic / Qwen3).

- **Hardware Specs**: Typically runs on NVIDIA A100 or H100 clusters depending on availability.
- **Serverless Integration**: Queries are submitted via the RunPod Serverless SDK, allowing the backend to scale to zero when not in use.

## GPU Cost Optimization Strategies

To maintain an industry-leading cost-to-performance ratio, we implement:

1.  **Strict Scaledown Windows**: `SCALEDOWN_WINDOW` is set to 10 minutes. If no requests are received, Modal automatically terminates or hibernates instances.
2.  **Concurrency Multiplexing**: Each GPU node handles up to 32 concurrent requests (`MAX_CONCURRENT`) by leveraging vLLM's continuous batching capabilities.
3.  **Tiered Model Sizes**:
    - Small, quantized models (Phi-4-mini) are used for high-frequency routing.
    - Larger, more expensive models (Qwen3) are only invoked "on-demand" for difficult queries or error correction.

## Microservices Orchestration

The core application is containerized and orchestrated via **Docker Compose** (compatible with Kubernetes/SaaS deployment):

| Service | Technology | Role |
| :--- | :--- | :--- |
| `backend` | Python 3.12 / FastAPI | Business logic and SSE orchestration. |
| `frontend` | Node.js / Nginx | User interface and static asset serving. |
| `prometheus` | Prometheus / Grafana | Metrics collection and alerting. |
| `promtail` | Promtail / Loki | Log shipping and central correlation. |

## Network & Connectivity

- **Protocols**: REST API for management, SSE for real-time query streams.
- **Security**: OTLP headers are injected for secure transmission of telemetry to Grafana Cloud.
- **DB Connection**: SQLite Cloud provides an encrypted, serverless gateway to our metadata storage.
