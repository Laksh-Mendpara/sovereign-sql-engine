# SSE Event Stream Specification

The Sovereign SQL Engine utilizes a high-frequency **Server-Sent Events (SSE)** stream to maintain a live connection between the backend and the frontend user during query execution.

## Connection Details

- **Endpoint**: `POST /v1/pipeline/stream`
- **Request Body**: `{ "query": "Your question here" }`
- **Content-Type**: `text/event-stream`
- **Keep-Alive**: Managed by the backend with a 50s heartbeat timeout.

## Event Sequence

Events are emitted chronologically as stages complete. Some stages (like Guard and Classifier) may emit in parallel.

| Event Name | Description | Payload Key |
| :--- | :--- | :--- |
| `pipeline.start` | Confirmation that query has entered the system. | `request_id`, `trace_id` |
| `guard` | Safety check result. | `allowed`, `reason` |
| `classification` | Query difficulty categorization. | `label`, `reason` |
| `pinecone` | Vector retrieval matches. | `columns`, `tables` |
| `neo4j` | Expanded schema path. | `schema_tables` |
| `schema` | Full SQL schema context generated. | `schema_sql` |
| `runpod` | Initial or improved SQL generation. | `generated_sql`, `response` |
| `execution.remark` | Analysis of the generated plan (Firewall status). | `remark`, `execution_sql` |
| `execution.error` | Error message if SQL execution fails initially. | `error`, `stage` |
| `execution.data` | Final result set from the database. | `execution_data` (rows) |
| `pipeline.complete` | Final status and per-stage latency metrics. | `metrics`, `skipped` |
| `pipeline.error` | Emitted on critical failure before connection close. | `error`, `detail` |

## Interaction Logic

### 1. Sequential Processing
The frontend UI incrementally builds the "Chain of Thought" by listening for these events. The stream stays open until either `pipeline.complete` or `pipeline.error` is received.

### 2. Event Replay (Caching)
If the backend detects a **Cache Hit**, it re-emits a stored sequence of all successful events (from `pipeline.start` to `pipeline.complete`) in a single rapid burst, providing sub-100ms "instant" responses for known queries.

### 3. Error Recovery
If an `execution.error` is emitted, the frontend displays the error state but **does not close the connection**. It waits for a subsequent `runpod` event (the self-corrected version) to arrive.
