"""
Phi-4 and Llama Guard on Modal with vLLM

Serves Microsoft Phi-4-mini-instruct and Llama-Guard-3-1B as
serverless OpenAI-compatible APIs on Modal, backed by vLLM.

Usage:
    modal deploy app.py     # deploy as persistent endpoints
    modal run app.py        # quick smoke test (ephemeral)
"""

import json
import os
import socket
import subprocess
import time
from typing import Any

import aiohttp
import modal

# -- Timing helper --
# We express timeouts in minutes for readability, but Modal expects seconds.
MINUTES = 60  # seconds per minute

# Backoff settings for _wait_ready().
WAIT_READY_INITIAL_BACKOFF = 0.5   # seconds; doubles each iteration
WAIT_READY_MAX_BACKOFF = 5.0       # seconds; upper bound on per-iteration sleep

# Port the vLLM HTTP server listens on inside the container.
VLLM_PORT = 8000

# HuggingFace model identifiers
PHI4_MODEL_NAME = "ByteMaster01/phi-4-mini-instruct-awq4"
LLAMA_GUARD_MODEL_NAME = "ByteMaster01/llama-guard-3-1b-awq4"

# ---- Runtime configuration ------------------------------------------------
# All of these can be overridden by setting environment variables *before*
# running `modal deploy`.  Defaults work fine for most use-cases.
N_GPU = int(os.environ.get("N_GPU", "1"))
MAX_MODEL_LEN_PHI4 = int(os.environ.get("MAX_MODEL_LEN_PHI4", "4096"))
MAX_MODEL_LEN_LLAMA_GUARD = int(os.environ.get("MAX_MODEL_LEN_LLAMA_GUARD", "2084"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
VLLM_QUANTIZATION = os.environ.get("VLLM_QUANTIZATION", "auto").strip().lower()
VLLM_VERSION = os.environ.get("VLLM_VERSION", "0.19.0").strip()
FLASHINFER_VERSION = os.environ.get("FLASHINFER_VERSION", "").strip()

# Fast cold-start snapshotting optimization
ENABLE_SNAPSHOTS = os.environ.get("ENABLE_SNAPSHOTS", "1") == "1"

SCALEDOWN_WINDOW = int(os.environ.get("SCALEDOWN_WINDOW", "1"))  # minutes
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "32"))

# ---- Modal app -------------------------------------------------------------
app = modal.App("phi4-and-llama-guard-inference")

# ---- Container image -------------------------------------------------------
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        *(
            [
                f"vllm=={VLLM_VERSION}",
                "huggingface-hub==0.36.0",
            ]
            + ([f"flashinfer-python=={FLASHINFER_VERSION}"] if FLASHINFER_VERSION else [])
        )
    )
    .env(
        {
            # Enable high-performance Xet backend for faster weight downloads.
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

# When GPU memory snapshots are enabled we need vLLM's dev/sleep mode and a
# single Torch Inductor compile thread (avoids snapshot compatibility issues).
if ENABLE_SNAPSHOTS:
    vllm_image = vllm_image.env(
        {
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )

# ---- Persistent volumes ----------------------------------------------------
# Model weights and vLLM compilation artefacts survive container restarts.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ---- In-container helpers ---------------------------------------------------
with vllm_image.imports():
    import requests as _requests


def _sleep(level: int = 1) -> None:
    """Put the vLLM server into sleep mode, offloading weights to CPU RAM."""
    _requests.post(
        f"http://localhost:{VLLM_PORT}/sleep?level={level}",
        timeout=60,
    ).raise_for_status()


def _wake_up() -> None:
    """Bring the vLLM server back from sleep mode."""
    _requests.post(f"http://localhost:{VLLM_PORT}/wake_up", timeout=60).raise_for_status()


def _wait_ready(proc: subprocess.Popen, startup_timeout: int = 10 * MINUTES) -> None:
    """Block until the vLLM server is accepting TCP connections."""
    deadline = time.monotonic() + startup_timeout
    delay = WAIT_READY_INITIAL_BACKOFF
    while True:
        try:
            socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
            return
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited unexpectedly with code {proc.returncode}"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"vLLM server did not become ready within {startup_timeout}s"
                )
            time.sleep(delay)
            delay = min(delay * 2, WAIT_READY_MAX_BACKOFF)


def _warmup() -> None:
    """Fire a handful of throwaway requests so CUDA graphs get compiled."""
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        _requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=300,
        ).raise_for_status()


# ---- vLLM command builder --------------------------------------------------

def _resolve_quantization() -> str | None:
    """Use an explicit override, otherwise let vLLM read the model config."""
    if VLLM_QUANTIZATION in {"", "auto"}:
        return None
    return VLLM_QUANTIZATION


def _build_vllm_cmd(model_name: str, max_model_len: int) -> list[str]:
    """Assemble the `vllm serve` CLI invocation."""
    cmd = [
        "vllm",
        "serve",
        model_name,
        # Register both the full HF name and a short "llm" alias so callers
        # can use either in the `model` field of API requests.
        "--served-model-name", model_name, "llm",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--tensor-parallel-size", str(N_GPU),
        "--trust-remote-code",
        "--max-model-len", str(max_model_len),
    ]

    quantization = _resolve_quantization()
    if quantization:
        cmd += ["--quantization", quantization]

    # Snapshot mode restricts concurrency and seq length so the GPU memory
    # layout is deterministic and can be snapshotted reliably.
    if ENABLE_SNAPSHOTS:
        cmd += [
            "--enable-sleep-mode",
            "--max-num-seqs", "2",
            "--max-num-batched-tokens", str(max_model_len),
        ]

    return cmd


# ---- Server classes ---------------------------------------------------------

@app.cls(
    image=vllm_image,
    gpu=f"L4:{N_GPU}",
    scaledown_window=10, #SCALEDOWN_WINDOW * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=ENABLE_SNAPSHOTS,
    **({"experimental_options": {"enable_gpu_snapshot": True}} if ENABLE_SNAPSHOTS else {}),
)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
class Phi4Server:
    """Wraps a vLLM subprocess to serve Phi-4-mini-instruct."""

    @modal.enter(snap=ENABLE_SNAPSHOTS)
    def start(self):
        cmd = _build_vllm_cmd(PHI4_MODEL_NAME, MAX_MODEL_LEN_PHI4)
        print("Starting vLLM (Phi-4):", " ".join(cmd))
        self.vllm_proc = subprocess.Popen(cmd)
        _wait_ready(self.vllm_proc)
        _warmup()

        if ENABLE_SNAPSHOTS:
            _sleep()

    @modal.enter(snap=False)
    def wake(self):
        if ENABLE_SNAPSHOTS:
            _wake_up()
            _wait_ready(self.vllm_proc)

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        proc = getattr(self, "vllm_proc", None)
        if not proc:
            return

        try:
            if proc.poll() is not None:
                return

            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as e:
            print(f"Error while terminating vLLM subprocess: {e}")


@app.cls(
    image=vllm_image,
    gpu=f"T4:{N_GPU}",
    scaledown_window=10, #SCALEDOWN_WINDOW * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=ENABLE_SNAPSHOTS,
    **({"experimental_options": {"enable_gpu_snapshot": True}} if ENABLE_SNAPSHOTS else {}),
)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
class LlamaGuardServer:
    """Wraps a vLLM subprocess to serve Llama-Guard-3-1B."""

    @modal.enter(snap=ENABLE_SNAPSHOTS)
    def start(self):
        cmd = _build_vllm_cmd(LLAMA_GUARD_MODEL_NAME, MAX_MODEL_LEN_LLAMA_GUARD)
        print("Starting vLLM (Llama Guard):", " ".join(cmd))
        self.vllm_proc = subprocess.Popen(cmd)
        _wait_ready(self.vllm_proc)
        _warmup()

        if ENABLE_SNAPSHOTS:
            _sleep()

    @modal.enter(snap=False)
    def wake(self):
        if ENABLE_SNAPSHOTS:
            _wake_up()
            _wait_ready(self.vllm_proc)

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        proc = getattr(self, "vllm_proc", None)
        if not proc:
            return

        try:
            if proc.poll() is not None:
                return

            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as e:
            print(f"Error while terminating vLLM subprocess: {e}")


# ---- Local entrypoint -------------------------------------------------------

@app.local_entrypoint()
async def test(test_timeout: int = 10 * MINUTES):
    """Run a basic health check and a single streamed inference request directly to both deployments."""
    print("Fetching server URLs...")
    phi4_url = await Phi4Server().serve.get_web_url.aio()
    llama_guard_url = await LlamaGuardServer().serve.get_web_url.aio()
    print(f"Phi-4 Server URL: {phi4_url}")
    print(f"Llama Guard Server URL: {llama_guard_url}")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain what a neural network is in 2 sentences."},
    ]

    async with aiohttp.ClientSession() as session:
        # -- Test Phi-4 --
        print("\n=== Testing Phi-4 ===")
        print("Running health check ...")
        async with session.get(
            f"{phi4_url}/health", timeout=aiohttp.ClientTimeout(total=test_timeout - MINUTES)
        ) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed.")

        print("Chat completion (streaming):")
        await _stream_chat(session, phi4_url, "llm", messages, timeout=2 * MINUTES)

        # -- Test Llama Guard --
        print("\n=== Testing Llama Guard ===")
        print("Running health check ...")
        async with session.get(
            f"{llama_guard_url}/health", timeout=aiohttp.ClientTimeout(total=test_timeout - MINUTES)
        ) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed.")

        print("Chat completion (streaming):")
        await _stream_chat(session, llama_guard_url, "llm", messages, timeout=2 * MINUTES)


async def _stream_chat(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: int,
) -> None:
    """Send a streaming /v1/chat/completions request and print tokens as they arrive."""
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 256,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with session.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]
            try:
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
    print()  # newline after the streamed output
