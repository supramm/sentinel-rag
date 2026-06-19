"""
rag_chain.py — SentinelAI RAG Pipeline Orchestrator

Orchestration order:
    context_injector → retriever → hallucination_gate → prompt_builder → llm_client

Returns RAGResult. Never raises — all errors surface in result fields.

── Interface contract (update aliases below if your function names differ) ─────

    context_injector.py  →  get_context(scenario_id: str) -> dict
        Returns: {scenario_id, machine_type, anomaly, health_status, rul,
                  flagged_sensors, retrieval_query, formatted_context}

    retriever.py         →  retrieve(query: str, machine_type: str, top_k: int) -> list[dict]
        Returns: [{text, source, page, machine_type, section_type,
                   cosine_score, jaccard_score, final_score}, ...]

    hallucination_gate.py →  check(chunks: list[dict]) -> dict
        Returns: {passed: bool, reason: str}

    prompt_builder.py    →  build_prompt(context: dict, chunks: list[dict], question: str) -> list[dict]
        Returns: OpenAI message list [{role, content}, ...]

──────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Component imports (update aliases here if function names differ) ──────────

from context_injector import get_scenario_context as get_context                  # noqa: E402
from retriever         import retriever                                  # noqa: E402
from hallucination_gate import evaluate_retrieval as gate_check       # noqa: E402
from prompt_builder    import build_prompt                  # noqa: E402
from llm_client        import generate, LLMResponse         # noqa: E402

# retriever.py exposes a `retriever` singleton with a `.retrieve()` method;
# provide a callable `retrieve` wrapper expected by this module.
retrieve = retriever.retrieve

# ── Logger ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s [rag_chain] %(message)s",
)
log = logging.getLogger("rag_chain")

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """
    Single output type for all pipeline runs.

    Fields:
        answer       — LLM-generated answer. None if refused or error.
        sources      — Display-safe chunk list. Empty if refused or error.
        refused      — True when hallucination gate blocks the query.
        reason       — Gate reason (if refused) or None.
        scenario_id  — Echo of input scenario_id.
        machine_type — Resolved from context.
        query_used   — Actual retrieval query sent to FAISS.
        backend      — LLM backend string (e.g. "groq/mixtral-8x7b-32768").
        latency      — Per-stage timing dict (ms). Keys: context, retrieval,
                       gate, prompt, llm, total.
        error        — Pipeline error string (non-refusal failures). None on success.
    """
    answer:       Optional[str]
    sources:      list[dict]
    refused:      bool
    reason:       Optional[str]
    scenario_id:  str
    machine_type: str
    query_used:   str
    backend:      str
    latency:      dict = field(default_factory=dict)
    error:        Optional[str] = None

    @property
    def success(self) -> bool:
        """True only when an answer was produced with no errors or refusal."""
        return (
            self.answer is not None
            and not self.refused
            and self.error is None
        )

    def to_dict(self) -> dict:
        return {
            "answer":       self.answer,
            "sources":      self.sources,
            "refused":      self.refused,
            "reason":       self.reason,
            "scenario_id":  self.scenario_id,
            "machine_type": self.machine_type,
            "query_used":   self.query_used,
            "backend":      self.backend,
            "latency":      self.latency,
            "error":        self.error,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _timed(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, round((time.perf_counter() - t0) * 1000, 1)


def _format_sources(chunks: list[dict]) -> list[dict]:
    """
    Strip chunks to display-safe fields.
    Truncates text to 400 chars — enough for a UI citation card.
    """
    return [
        {
            "text":          c.get("text", "")[:400],
            "source":        c.get("source", ""),
            "page":          c.get("page", ""),
            "section_type":  c.get("section_type", ""),
            "machine_type":  c.get("machine_type", ""),
            "final_score":   round(c.get("final_score",   0.0), 4),
            "cosine_score":  round(c.get("cosine_score",  0.0), 4),
            "jaccard_score": round(c.get("jaccard_score", 0.0), 4),
        }
        for c in chunks
    ]


# ── Refused query log ─────────────────────────────────────────────────────────

_REFUSED_LOG = Path(__file__).parent / "kb" / "refused_queries.jsonl"


def _log_refused(
    scenario_id: str,
    question:    str,
    query:       str,
    reason:      str,
) -> None:
    """
    Append one line to kb/refused_queries.jsonl.
    Each entry is a KB gap signal — review periodically to extend the knowledge base.
    """
    import datetime
    entry = {
        "ts":          datetime.datetime.utcnow().isoformat() + "Z",
        "scenario_id": scenario_id,
        "question":    question,
        "query":       query,
        "reason":      reason,
    }
    try:
        _REFUSED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _REFUSED_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Could not write refused_queries.jsonl: %s", exc)


# ── Stage runners (each returns RAGResult on failure, None on success) ────────

def _stage_context(scenario_id: str, latency: dict) -> tuple:
    """Returns (context_dict, None) or (None, RAGResult)."""
    try:
        ctx, ms = _timed(get_context, scenario_id)
        latency["context_ms"] = ms
        log.info(
            "context  | scenario=%s machine=%s anomaly=%s rul=%s  (%.0fms)",
            scenario_id,
            ctx.get("machine_type", "?"),
            ctx.get("anomaly_type", "?"),
            ctx.get("rul_days", "?"),
            ms,
        )
        return ctx, None
    except Exception as exc:
        log.error("context_injector failed: %s", exc)
        return None, RAGResult(
            answer=None, sources=[], refused=False, reason=None,
            scenario_id=scenario_id, machine_type="unknown",
            query_used="", backend="none", latency=latency,
            error=f"context_injector: {exc}",
        )


def _stage_retrieval(
    query:        str,
    machine_type: str,
    top_k:        int,
    latency:      dict,
    scenario_id:  str,
) -> tuple:
    """Returns (chunks, None) or (None, RAGResult)."""
    try:
        chunks, ms = _timed(retrieve, query, machine_type=machine_type, top_k=top_k)
        latency["retrieval_ms"] = ms
        log.info(
            "retrieval| %d chunks  query=%r  (%.0fms)",
            len(chunks), query[:60], ms,
        )
        return chunks, None
    except Exception as exc:
        log.error("retriever failed: %s", exc)
        return None, RAGResult(
            answer=None, sources=[], refused=False, reason=None,
            scenario_id=scenario_id, machine_type=machine_type,
            query_used=query, backend="none", latency=latency,
            error=f"retriever: {exc}",
        )


def _stage_gate(
    chunks:      list[dict],
    question:    str,
    query:       str,
    latency:     dict,
    scenario_id: str,
    machine_type: str,
) -> tuple:
    """Returns (None, None) on pass, or (None, RAGResult) on refuse/error."""
    try:
        gate_result, ms = _timed(gate_check, query, chunks)
        latency["gate_ms"] = ms
        passed = gate_result.get("passed", False)
        reason = gate_result.get("reason", "")
        log.info(
            "gate     | passed=%s reason=%r  (%.0fms)",
            passed, reason, ms,
        )
        if not passed:
            _log_refused(scenario_id, question, query, reason)
            return None, RAGResult(
                answer=None, sources=[], refused=True,
                reason=reason or "Retrieval scores below threshold.",
                scenario_id=scenario_id, machine_type=machine_type,
                query_used=query, backend="none", latency=latency,
            )
        return None, None
    except Exception as exc:
        log.error("hallucination_gate failed: %s", exc)
        return None, RAGResult(
            answer=None, sources=[], refused=False, reason=None,
            scenario_id=scenario_id, machine_type=machine_type,
            query_used=query, backend="none", latency=latency,
            error=f"hallucination_gate: {exc}",
        )


def _stage_prompt(
    context:      dict,
    chunks:       list[dict],
    question:     str,
    latency:      dict,
    scenario_id:  str,
    machine_type: str,
    query:        str,
) -> tuple:
    """Returns (messages, None) or (None, RAGResult)."""
    try:
        prompt_text, ms = _timed(build_prompt, context, chunks, question)
        latency["prompt_ms"] = ms
        
        # Wrap string prompt into OpenAI message format
        messages = [
            {
                "role": "user",
                "content": prompt_text
            }
        ]
        
        log.info(
            "prompt   | %d messages  (%.0fms)",
            len(messages), ms,
        )
        return messages, None
    except Exception as exc:
        log.error("prompt_builder failed: %s", exc)
        return None, RAGResult(
            answer=None, sources=[], refused=False, reason=None,
            scenario_id=scenario_id, machine_type=machine_type,
            query_used=query, backend="none", latency=latency,
            error=f"prompt_builder: {exc}",
        )


# ── Public API ────────────────────────────────────────────────────────────────

def run(
    scenario_id:  str,
    question:     str,
    top_k:        int = 5,
    llm_backend:  Optional[str] = None,
) -> RAGResult:
    """
    Full pipeline: loads context from scenario_id, then runs retrieval → gate
    → prompt → LLM.

    Args:
        scenario_id:  Must exist in scenarios.json / results.json.
        question:     User natural-language query.
        top_k:        Chunks to retrieve (default 5).
        llm_backend:  Override LLM backend. None = use LLM_BACKEND env var.

    Returns:
        RAGResult — always. Never raises.
    """
    latency: dict = {}
    t_start = time.perf_counter()

    # 1 ── Context
    context, err = _stage_context(scenario_id, latency)
    if err:
        return err

    machine_type = context.get("machine_type", "")
    query        = context.get("retrieval_query") or question

    # 2 ── Retrieval
    chunks, err = _stage_retrieval(query, machine_type, top_k, latency, scenario_id)
    if err:
        return err

    # 3 ── Hallucination gate
    _, err = _stage_gate(chunks, question, query, latency, scenario_id, machine_type)
    if err:
        return err

    # 4 ── Prompt
    messages, err = _stage_prompt(context, chunks, question, latency, scenario_id, machine_type, query)
    if err:
        return err

    # 5 ── LLM
    llm_resp, llm_ms = _timed(generate, messages, llm_backend)
    latency["llm_ms"]   = llm_ms
    latency["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

    if not llm_resp.success:
        log.error("llm failed | backend=%s error=%s", llm_resp.backend, llm_resp.error)
        return RAGResult(
            answer=None, sources=[], refused=False, reason=None,
            scenario_id=scenario_id, machine_type=machine_type,
            query_used=query, backend=llm_resp.backend, latency=latency,
            error=f"llm_client: {llm_resp.error}",
        )

    log.info(
        "done     | backend=%s total=%.0fms  (ctx=%.0f ret=%.0f gate=%.0f llm=%.0f)",
        llm_resp.backend,
        latency["total_ms"],
        latency.get("context_ms",   0),
        latency.get("retrieval_ms", 0),
        latency.get("gate_ms",      0),
        latency.get("llm_ms",       0),
    )

    return RAGResult(
        answer=llm_resp.text,
        sources=_format_sources(chunks),
        refused=False,
        reason=None,
        scenario_id=scenario_id,
        machine_type=machine_type,
        query_used=query,
        backend=llm_resp.backend,
        latency=latency,
    )


def run_from_context(
    context:     dict,
    question:    str,
    top_k:       int = 5,
    llm_backend: Optional[str] = None,
) -> RAGResult:
    """
    Same pipeline but accepts a pre-built context dict.

    Use this in app.py when the context is already loaded for the current
    session — avoids a redundant scenarios.json read on every chat turn.

    Args:
        context:     Dict as returned by context_injector.get_context().
        question:    User natural-language query.
        top_k:       Chunks to retrieve (default 5).
        llm_backend: Override LLM backend.

    Returns:
        RAGResult — always. Never raises.
    """
    scenario_id  = context.get("scenario_id",     "unknown")
    machine_type = context.get("machine_type",    "")
    query        = context.get("retrieval_query") or question
    latency: dict = {}
    t_start = time.perf_counter()

    # 1 ── Retrieval  (no context stage — already have it)
    chunks, err = _stage_retrieval(query, machine_type, top_k, latency, scenario_id)
    if err:
        return err

    # 2 ── Gate
    _, err = _stage_gate(chunks, question, query, latency, scenario_id, machine_type)
    if err:
        return err

    # 3 ── Prompt
    messages, err = _stage_prompt(context, chunks, question, latency, scenario_id, machine_type, query)
    if err:
        return err

    # 4 ── LLM
    llm_resp, llm_ms = _timed(generate, messages, llm_backend)
    latency["llm_ms"]   = llm_ms
    latency["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

    if not llm_resp.success:
        return RAGResult(
            answer=None, sources=[], refused=False, reason=None,
            scenario_id=scenario_id, machine_type=machine_type,
            query_used=query, backend=llm_resp.backend, latency=latency,
            error=f"llm_client: {llm_resp.error}",
        )

    return RAGResult(
        answer=llm_resp.text,
        sources=_format_sources(chunks),
        refused=False,
        reason=None,
        scenario_id=scenario_id,
        machine_type=machine_type,
        query_used=query,
        backend=llm_resp.backend,
        latency=latency,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SentinelAI RAG chain — end-to-end test")
    parser.add_argument("--scenario",  required=True,
                        help="Scenario ID from scenarios.json (e.g. S_001)")
    parser.add_argument("--question",
                        default="What is the most likely failure mode and recommended maintenance action?")
    parser.add_argument("--top-k",    type=int, default=5,
                        help="Number of chunks to retrieve (default 5)")
    parser.add_argument("--backend",  default=None,
                        choices=["groq", "hf", "ollama", "mock"],
                        help="Override LLM backend (default: LLM_BACKEND env var)")
    parser.add_argument("--json",     action="store_true",
                        help="Dump full JSON result instead of human-readable output")
    args = parser.parse_args()

    result = run(
        scenario_id=args.scenario,
        question=args.question,
        top_k=args.top_k,
        llm_backend=args.backend,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"Scenario   : {result.scenario_id}")
        print(f"Machine    : {result.machine_type}")
        print(f"Query      : {result.query_used}")
        print(f"Success    : {result.success}")

        if result.refused:
            print(f"REFUSED    : {result.reason}")

        elif result.error:
            print(f"ERROR      : {result.error}")

        else:
            print(f"Backend    : {result.backend}")
            print(
                f"Latency    : {result.latency.get('total_ms', 0):.0f}ms total  "
                f"(ctx={result.latency.get('context_ms', 0):.0f}  "
                f"ret={result.latency.get('retrieval_ms', 0):.0f}  "
                f"gate={result.latency.get('gate_ms', 0):.0f}  "
                f"llm={result.latency.get('llm_ms', 0):.0f})"
            )
            print(f"\nAnswer:\n{result.answer}")
            print(f"\nSources ({len(result.sources)}):")
            for i, s in enumerate(result.sources, 1):
                print(
                    f"  [{i}] {s['source']}  p.{s['page']}  "
                    f"§{s['section_type']}  score={s['final_score']}"
                )
                print(f"       {s['text'][:120]}…")

        print(f"{sep}\n")
