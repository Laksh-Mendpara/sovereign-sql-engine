# Text2SQL Pipeline Check

Standalone pipeline checker for:

- Llama Guard validation
- Phi-4 query classification
- Pinecone retrieval and reranking
- Neo4j table-join expansion
- SQLite metadata schema assembly
- Runpod Arctic prompt execution

Run from the repo root:

```bash
python -m pipeline_test.main "your natural language query"
```
