"""
llm_client.py — SentinelAI RAG Copilot

Multi-backend LLM client with fallback chain.

Priority:
  1. Groq    — free API, fast, supports mixtral-8x7b-32768
  2. HF      — HuggingFace Inference Providers (correct provider routing)
  3. Ollama  — local inference
  4. Mock    — offline / CI testing

Config via .env:
  LLM_BACKEND=groq               # groq | hf | ollama | mock
  GROQ_API_KEY=gsk_...
  GROQ_MODEL=mixtral-8x7b-32768  # optional override
  HF_TOKEN=hf_...
  HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
  HF_PROVIDER=together           # together | fireworks-ai | hf-inference
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_MODEL=mistral
  LLM_MAX_TOKENS=512
  LLM_TEMPERATURE=0.2
"""

import os
import time
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# ── Enums ─────────────────────────────────────────────────────────────────────

class Backend(str, Enum):
    GROQ   = "groq"
    HF     = "hf"
    OLLAMA = "ollama"
    MOCK   = "mock"
# =========================================================
# SECRETS / ENV
# =========================================================

try:

    import streamlit as st

    GROQ_API_KEY = st.secrets.get(
        "GROQ_API_KEY",
        os.getenv("GROQ_API_KEY", "")
    )

    HF_TOKEN = st.secrets.get(
        "HF_TOKEN",
        os.getenv("HF_TOKEN", "")
    )

except Exception:

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY",
        ""
    )

    HF_TOKEN = os.getenv(
        "HF_TOKEN",
        ""
    )
# ── Config ────────────────────────────────────────────────────────────────────

BACKEND      = Backend(os.getenv("LLM_BACKEND", "groq").lower())

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

# HF  — provider= is the key fix; "together" routes to Together AI backend
#       which correctly handles Mistral chat format
HF_TOKEN     = os.getenv("HF_TOKEN", "")
HF_MODEL     = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_PROVIDER  = os.getenv("HF_PROVIDER", "together")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "mistral")

# Generation
MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "512"))
TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ── Response type ─────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text:       str
    backend:    str
    latency_ms: float
    error:      Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        return (
            f"LLMResponse(backend={self.backend!r}, "
            f"latency={self.latency_ms:.0f}ms, "
            f"chars={len(self.text)}, "
            f"ok={self.success})"
        )

# ── Backend: Groq ─────────────────────────────────────────────────────────────

def _call_groq(messages: list[dict]) -> LLMResponse:
    """
    Groq serverless API.
    Free tier: 14,400 req/day.
    Get key: https://console.groq.com
    """
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("Run: pip install groq")

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env")

    t0 = time.perf_counter()
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    latency = (time.perf_counter() - t0) * 1000
    text = resp.choices[0].message.content.strip()
    return LLMResponse(text=text, backend=f"groq/{GROQ_MODEL}", latency_ms=latency)


# ── Backend: HuggingFace ──────────────────────────────────────────────────────

def _call_hf(messages: list[dict]) -> LLMResponse:
    """
    HF Inference Providers (huggingface_hub >= 0.23).

    Root cause of previous errors:
      - api-inference.huggingface.co → deprecated endpoint, DNS fails
      - text_generation() → old pipeline API, routing mismatch on instruct models
      - chat_completion() without provider= → HF serverless mis-routes Mistral

    Fix: InferenceClient(provider="together") routes to Together AI
    which correctly handles Mistral-7B-Instruct chat format via
    OpenAI-compatible /chat/completions endpoint.

    Requires: HF token with billing OR Together free credits.
    Fallback: provider="hf-inference" for rate-limited free tier.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise RuntimeError("Run: pip install huggingface_hub>=0.23")

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set in .env")

    t0 = time.perf_counter()
    client = InferenceClient(
        provider=HF_PROVIDER,
        api_key=HF_TOKEN,
    )
    resp = client.chat.completions.create(
        model=HF_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    latency = (time.perf_counter() - t0) * 1000
    text = resp.choices[0].message.content.strip()
    return LLMResponse(
        text=text,
        backend=f"hf/{HF_PROVIDER}/{HF_MODEL}",
        latency_ms=latency,
    )


# ── Backend: Ollama ───────────────────────────────────────────────────────────

def _call_ollama(messages: list[dict]) -> LLMResponse:
    """
    Ollama local inference.
    Setup: https://ollama.ai → ollama pull mistral
    Runs at localhost:11434 by default.
    """
    import requests as req

    t0 = time.perf_counter()
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }
    response = req.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    latency = (time.perf_counter() - t0) * 1000
    text = response.json()["message"]["content"].strip()
    return LLMResponse(
        text=text,
        backend=f"ollama/{OLLAMA_MODEL}",
        latency_ms=latency,
    )


# ── Backend: Mock ─────────────────────────────────────────────────────────────

def _call_mock(messages: list[dict]) -> LLMResponse:
    """
    Offline mock for testing without API keys.
    Returns a deterministic response shaped like a real RAG answer.
    """
    t0 = time.perf_counter()

    # Extract system and user content for a plausible mock
    system_content = next(
        (m["content"] for m in messages if m["role"] == "system"), ""
    )
    user_content = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    # Detect scenario keywords for a shaped response
    keywords = {
        "bearing": "Elevated vibration signature on the flagged sensor indicates early-stage bearing wear. Recommended action: schedule inspection within 48 hours, apply lubrication, monitor RMS vibration trend.",
        "thermal": "Temperature exceedance pattern detected. Likely cause: coolant flow restriction or increased friction load. Recommended action: verify coolant level and flow rate, inspect heat exchanger.",
        "pressure": "Gradual pressure drop consistent with a minor hydraulic leak or seal degradation. Recommended action: inspect hydraulic lines and fittings in the flagged subsystem before next shift.",
        "rul": "Remaining Useful Life estimate indicates critical window. Recommended action: escalate to maintenance crew, prepare replacement components, plan downtime within the predicted window.",
    }

    combined = (system_content + user_content).lower()
    response_text = next(
        (resp for kw, resp in keywords.items() if kw in combined),
        (
            "Based on the current sensor profile and retrieved documentation, "
            "the flagged anomaly requires attention within the next maintenance window. "
            "Monitor the trend for 2 hours. If deviation persists, escalate to Level 2 inspection. "
            "Source: Equipment Manual Section 4.3 — Fault Isolation Procedure."
        ),
    )

    latency = (time.perf_counter() - t0) * 1000
    return LLMResponse(text=response_text, backend="mock", latency_ms=latency)


# ── Dispatch table ────────────────────────────────────────────────────────────

_DISPATCH = {
    Backend.GROQ:   _call_groq,
    Backend.HF:     _call_hf,
    Backend.OLLAMA: _call_ollama,
    Backend.MOCK:   _call_mock,
}

# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    messages: list[dict],
    backend: Optional[str] = None,
) -> LLMResponse:
    """
    Generate a response from the configured LLM backend.

    Args:
        messages:  OpenAI-format [{"role": "system"|"user"|"assistant", "content": str}]
        backend:   Override backend for this call. One of: groq, hf, ollama, mock.
                   Defaults to LLM_BACKEND env var (default: groq).

    Returns:
        LLMResponse — always returns, never raises.
        On error: .success = False, .error = reason, .text = ""
    """
    target = Backend(backend.lower()) if backend else BACKEND
    fn = _DISPATCH[target]

    try:
        return fn(messages)
    except Exception as exc:
        return LLMResponse(
            text="",
            backend=str(target.value),
            latency_ms=0.0,
            error=str(exc),
        )


def health_check(backend: Optional[str] = None) -> dict:
    """
    Smoke test the configured backend.
    Returns: {"status": "ok"|"error", "backend": str, "latency_ms": float, "error"?: str}
    """
    resp = generate(
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        backend=backend,
    )
    if resp.success:
        return {
            "status": "ok",
            "backend": resp.backend,
            "latency_ms": round(resp.latency_ms, 1),
        }
    return {
        "status": "error",
        "backend": resp.backend,
        "error": resp.error,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SentinelAI LLM client test")
    parser.add_argument(
        "--backend", default=None,
        choices=["groq", "hf", "ollama", "mock"],
        help="Override backend (default: LLM_BACKEND env var)",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Custom prompt to test (default: bearing anomaly query)",
    )
    args = parser.parse_args()

    print(f"[config] backend = {args.backend or BACKEND.value}")
    print(f"[config] max_tokens = {MAX_TOKENS}, temperature = {TEMPERATURE}")
    print()

    # Health check
    print("── Health check ──────────────────────────────")
    result = health_check(backend=args.backend)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()

    if result["status"] != "ok":
        print("Health check failed. Check .env configuration.")
        raise SystemExit(1)

    # Full inference test
    print("── Inference test ────────────────────────────")
    prompt = args.prompt or (
        "Machine: Conveyor Belt CB-02. "
        "Alert: LSTM AE reconstruction error = 0.47 (threshold 0.31). "
        "Flagged sensors: vibration_mm_s (+0.6 above baseline), acoustic_level_db (+18 dB). "
        "RUL: 31 cycles (critical). "
        "What is the most likely failure mode and recommended maintenance action?"
    )

    resp = generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an industrial maintenance expert for a manufacturing plant. "
                    "Answer concisely based on sensor data and equipment knowledge. "
                    "Always cite which sensor triggered your conclusion."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        backend=args.backend,
    )

    print(f"Backend  : {resp.backend}")
    print(f"Latency  : {resp.latency_ms:.0f}ms")
    print(f"Response :\n{resp.text}")