# Sovereign SQL Frontend

React + Vite frontend for the Text2SQL SSE pipeline.

## What it shows

- The pipeline as it streams, event by event
- Guardrail results
- Classification results, including out-of-logic / out-of-topic
- Retrieval, schema, and RunPod SQL generation
- Execution remark, firewall blocks, and final execution SQL
- Fetched data rows from the execution query
- Per-stage timings when the pipeline completes

## Run

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the frontend:
   ```bash
   npm run dev
   ```

3. Make sure the backend is running on `http://localhost:8000`.

## Configuration

Set `VITE_API_BASE_URL` if your backend is not on `http://localhost:8000`.

Example:
```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```
