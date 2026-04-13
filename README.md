# Concurrent Lightweight Multi-Agent NL2Data Engine

A lightweight multi-agent system for translating natural language queries into executable SQL and MongoDB queries, optimized for limited hardware resources (single CPU and ≤16GB RAM).

**Authors:** Laksh Mendpara (B23CS1037), Sahil Preet Singh (B23CS1061)  
**Date:** March 2025

## Abstract

Natural language interfaces for databases (NL2SQL or NL2MongoDB) allow users to query structured data without writing database queries. Although large language models (LLMs) can generate queries from natural language, their performance degrades when database schemas contain many tables and relationships.

This project proposes a lightweight multi-agent system that uses a graph-based representation of database schemas to identify relevant tables and reduce prompt size before query generation. Quantized language models running with llama.cpp are used for efficient inference, while a validation step checks generated queries before execution.

## Project Objectives

The system focuses on three main goals:

- **Schema Filtering**: Use a graph representation of the database schema to identify the most relevant tables for a given query.
- **Concurrent Query Handling**: Support multiple user requests using asynchronous request handling.
- **Query Validation**: Verify generated queries in a controlled environment before executing them on the target database.

## System Architecture

```
User Query
    ↓
   FastAPI Gateway
    ↓
┌───────────────────────────────────┐
│ Intent Agent                      │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Table Selection Agent             │
│ (Neo4j Schema Graph)              │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Query Generator                   │
│ (Quantized LLM)                   │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Validation Agent                  │
└───────────────────────────────────┘
    ↓
  Database
```

## Key Methodology

### Graph-Based Schema Representation

The database metadata (tables, columns, and relationships) is represented as a property graph in Neo4j.

- **Entity Identification**: Keywords extracted from the user query are matched with nodes in the schema graph.
- **Relationship Traversal**: If multiple entities are referenced in a query, the system identifies a join path between them using graph traversal.
- **Context Reduction**: Only the relevant subset of tables is provided to the language model, reducing prompt size and improving efficiency.

### Concurrent Inference

Inference is performed using quantized models running with llama-cpp-python. The server supports multiple inference slots that allow several requests to remain active simultaneously.

**Throughput Equation:**
$$\text{Throughput} = \frac{\text{Number of Slots}}{\text{Average Token Generation Time}}$$

Although token generation is sequential on the CPU, asynchronous request handling ensures that multiple users can interact with the system without blocking.

### Lightweight Language Models

Small language models (1.5B–7B parameters) are used with 4-bit quantization to reduce memory usage. These models can generate accurate SQL queries when provided with well-structured schema context.

## Project Structure

```
sovereign-sql-engine/
├── vllm_worker/              # vLLM-based inference worker
│   ├── src/
│   │   ├── engine.py         # Core query generation engine
│   │   ├── handler.py        # Request handling
│   │   ├── engine_args.py    # Configuration
│   │   ├── tokenizer.py      # Tokenization utilities
│   │   ├── constants.py      # Constants and configs
│   │   └── utils.py          # Utility functions
│   ├── Dockerfile            # Container configuration
│   ├── docker-bake.hcl       # Docker build configuration
│   └── docs/                 # Documentation
├── arctic-quantization/      # Model quantization utilities
│   ├── quantize_bnb8.py      # 8-bit quantization
│   ├── quantize_gptq8.py     # GPTQ quantization
│   └── test_model.py         # Model testing
├── metadata_creation/        # Schema graph creation
│   ├── main.py               # Metadata extraction logic
│   └── conn_test.py          # Database connection testing
├── modal_deployment/         # Modal.com deployment
│   ├── app.py                # Modal application
│   └── test.py               # Tests
└── README.md                 # This file
```

## Libraries and Frameworks

- **Hugging Face** – model repository for SQL generation and instruction models
- **llama-cpp-python** – CPU-based inference engine for quantized GGUF models
- **Neo4j** – graph database for representing database schema relationships
- **FastAPI** – backend framework for handling HTTP requests and managing asynchronous workflows
- **LangGraph** – framework for coordinating interactions between system agents
- **Docker** – containerization and deployment
- **vLLM** – optimized inference engine for language models

## Datasets and Models

### Datasets
- **Spider 1.0**: Large-scale domain-specific semantic parsing dataset
- **BIRD Benchmark**: A large-scale text-to-SQL benchmark with diverse databases

### Models
- **SQLCoder-7B** (GGUF format) – for SQL query generation
- **Qwen-2.5-1.5B-Instruct** – for intent detection and routing

## Implementation Stages

1. **Schema Graph Construction**: Extract metadata from the database and store it in Neo4j
2. **Model Setup**: Deploy quantized language models using llama.cpp or vLLM
3. **Agent Pipeline Development**: Implement the sequence of agents for intent detection, table selection, query generation, and validation
4. **API Integration**: Build FastAPI endpoints to expose the query translation service
5. **Frontend Interface**: Develop a web interface to submit queries and display generated SQL and execution results
6. **Deployment**: Containerize and deploy using Docker

## Getting Started

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)
- Neo4j (for schema graph storage)
- 16GB RAM (minimum recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd sovereign-sql-engine

# Install dependencies for vLLM worker
cd vllm_worker
pip install -r builder/requirements.txt

# Install quantization utilities
cd ../arctic-quantization
pip install -e .

# Install metadata creation tools
cd ../metadata_creation
pip install -e .
```

### Running the System

Each component can be run independently:

```bash
# Start the vLLM worker
cd vllm_worker
python -m src.engine

# Setup schema graph (requires Neo4j running)
cd metadata_creation
python main.py --database-url <your-db-connection>

# Test the inference pipeline
cd arctic-quantization
python test_model.py
```

## Architecture Components

### VLLMWorker (`vllm_worker/`)
The core inference component handling query generation using optimized inference engines.

### Quantization Module (`arctic-quantization/`)
Utilities for quantizing models to 4-bit or 8-bit precision for efficient inference on CPU.

### Metadata Creation (`metadata_creation/`)
Extracts database schema information and constructs the Neo4j graph representation.

### Deployment (`modal_deployment/`)
Cloud-ready deployment configuration using Modal.com for serverless inference.

## Performance Characteristics

- **Memory Usage**: ≤16GB RAM for complete system
- **Inference Speed**: Token generation on single CPU with multiple concurrent slots
- **Model Support**: 1.5B–7B parameter models with 4-bit quantization
- **Supported Databases**: SQL (MySQL, PostgreSQL, etc.) and MongoDB

## Contributing

Contributions are welcome! Please ensure:
- Code follows the project structure
- Models are in GGUF format
- Tests pass before submitting PRs

## License

See [LICENSE](LICENSE) file for details.

## References

- Spider 1.0 Dataset: https://spider.ws/
- BIRD Benchmark: https://bird-bench.github.io/
- llama-cpp-python: https://github.com/abetlen/llama-cpp-python