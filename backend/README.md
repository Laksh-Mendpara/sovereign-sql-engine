# Sovereign SQL Backend

Production-grade FastAPI backend for the Text2SQL pipeline.

## Features

- **Async Orchestration**: High optimization with parallel service calls.
- **Structured Logging**: JSON logs with `request_id`, `trace_id`, and `span_id`.
- **Observability**: Built-in metrics (latency histograms, request counters) and health checks.
- **Modular Design**: Clean separation of concerns (routes, middleware, metrics, executor).

## Setup

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Environment Variables**:
   It reads configuration from `../pipeline_test/.env`. Make sure that file is populated.

3. **Run Server**:
   ```bash
   # From the backend directory
   uv sync
   uv run start
   ```

## API Endpoints & Examples

### Health & Observability

**Liveness Probe**
```bash
curl -i http://localhost:8000/health
```

**Readiness Probe** (Checks DB and Neo4j connectivity)
```bash
curl -i http://localhost:8000/ready
```

**Metrics Snapshot**
```bash
curl -s http://localhost:8000/metrics | python3 -m json.tool
```

### Pipeline Execution

**Run SQL Generation Pipeline** (full JSON response)
```bash
curl -X POST http://localhost:8000/v1/pipeline/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which players scored more than 50 goals for Real Madrid?",
    "trace_id": "test-trace-123"
  }' | python3 -m json.tool
```

### SSE Streaming Pipeline (Staged Results)

Stream each stage result to the client *immediately on completion* using **Server-Sent Events**.

**curl (use `-N` for no buffering)**
```bash
curl -N -X POST http://localhost:8000/v1/pipeline/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Which drivers won the most Formula 1 races?"}'
```

**Example output — events arrive progressively:**
```text
event: pipeline.start
id: 1
data: {"request_id":"01KPJXABC...", "trace_id":"...", "query":"Which drivers..."}

event: classification
id: 2
data: {"label":"easy", "reason":"Simple aggregation question", "latency_ms": 890}

event: guard
id: 3
data: {"allowed":true, "reason":"", "latency_ms": 1200}

event: pinecone
id: 4
data: {"columns":[...], "tables":[...], "latency_ms": 2100}

event: neo4j
id: 5
data: {"schema_tables":["races","drivers","results"], "latency_ms": 340}

event: schema
id: 6
data: {"schema_sql":"CREATE TABLE races (...)", "latency_ms": 180}

event: runpod
id: 7
data: {"generated_sql":"SELECT d.forename, d.surname, COUNT(*) AS wins...", "latency_ms": 14200}

event: pipeline.complete
id: 8
data: {"skipped":false, "metrics":{"guard_ms":1200, "classifier_ms":890, ...}}
```

**JavaScript / TypeScript client (browser or Node)**
```javascript
async function streamPipeline(query) {
  const response = await fetch('/v1/pipeline/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Parse SSE frames from buffer
    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';   // keep incomplete last frame

    for (const frame of frames) {
      const lines = frame.trim().split('\n');
      const event = {};
      for (const line of lines) {
        if (line.startsWith('event: ')) event.type = line.slice(7);
        if (line.startsWith('data: '))  event.data = JSON.parse(line.slice(6));
        if (line.startsWith('id: '))    event.id = line.slice(4);
      }
      if (event.type && event.data) {
        console.log(`[${event.id}] ${event.type}:`, event.data);
        // dispatch to your UI here, e.g. updateStageCard(event.type, event.data)
      }
    }
  }
}

streamPipeline('Which drivers won the most Formula 1 races?');
```

## Interactive API Docs

Full OpenAPI documentation with "Try it out" buttons:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
