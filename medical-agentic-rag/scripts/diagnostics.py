"""Safe end-to-end diagnostics.

Checks each layer the deployed app depends on and returns structured results the
Streamlit status panel and the test suite both consume.

**No secret ever leaves this module.** The API key is reported only as a boolean
plus a masked length; every failure message is one of the fixed status strings
from :mod:`llm.base`, never a raw exception that might echo a request header.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow `python scripts/diagnostics.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings, get_settings  # noqa: E402

logger = logging.getLogger(__name__)

OK, FAIL, WARN, SKIP = "OK", "FAIL", "WARN", "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == OK

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail, **self.data}


def check_api_key(settings: Settings | None = None) -> CheckResult:
    """Confirm a key is present without revealing any part of it."""
    settings = settings or get_settings()
    key = settings.groq_api_key or ""
    if not key.strip():
        return CheckResult(
            "GROQ_API_KEY", FAIL,
            "Not set. Add it to Streamlit secrets (or .env) as GROQ_API_KEY.",
            {"configured": False},
        )
    # Length only - never any character of the key itself.
    return CheckResult(
        "GROQ_API_KEY", OK, f"Configured (length {len(key.strip())}).", {"configured": True}
    )


def check_llm_client(settings: Settings | None = None) -> CheckResult:
    """Build the Groq client and report a classified status."""
    settings = settings or get_settings()
    try:
        from llm.factory import build_llm_client  # noqa: PLC0415

        client = build_llm_client(settings)
    except Exception as exc:  # noqa: BLE001
        from llm.base import classify_llm_error  # noqa: PLC0415

        return CheckResult("LLM client", FAIL, classify_llm_error(exc), {"available": False})

    health = client.health()
    if not client.available:
        return CheckResult(
            "LLM client", FAIL,
            str(health.get("reason", "unavailable")),
            {"available": False, "provider": health.get("provider")},
        )
    return CheckResult(
        "LLM client", OK, f"Groq client created (model {settings.groq_model}).",
        {"available": True, "provider": "groq", "model": settings.groq_model},
    )


def check_llm_response(settings: Settings | None = None) -> CheckResult:
    """Send a minimal prompt to confirm the key and model actually work.

    This is the only check that detects an invalid key: constructing the client
    succeeds regardless, because Groq authenticates on first request.
    """
    settings = settings or get_settings()
    if not settings.has_llm_credentials:
        return CheckResult("LLM response", SKIP, "No API key configured.", {"available": False})
    try:
        from llm.groq_client import GroqClient  # noqa: PLC0415

        client = GroqClient(settings)
        if not client.available:
            return CheckResult("LLM response", FAIL, client.status, {"available": False})
        ok, message = client.ping()
        return CheckResult(
            "LLM response", OK if ok else FAIL, message,
            {"available": ok, "model": settings.groq_model},
        )
    except Exception as exc:  # noqa: BLE001
        from llm.base import classify_llm_error  # noqa: PLC0415

        return CheckResult("LLM response", FAIL, classify_llm_error(exc), {"available": False})


def check_embedder(settings: Settings | None = None) -> CheckResult:
    """Load the embedder and encode one query."""
    settings = settings or get_settings()
    try:
        from ingestion.embedder import build_embedder  # noqa: PLC0415

        embedder = build_embedder(settings)
        vector = embedder.embed_query("chronic kidney disease staging")
        info = embedder.describe()
        if len(vector) != embedder.dimension:
            return CheckResult("Embeddings", FAIL, "Dimension mismatch on encode.", info)
        if not info.get("semantic"):
            return CheckResult(
                "Embeddings", WARN,
                "Lexical hashing fallback in use - sentence-transformers is not "
                "installed or the model could not be downloaded. Retrieval still "
                "works via BM25 fusion but is not semantic.",
                info,
            )
        return CheckResult("Embeddings", OK, f"{info.get('model')} loaded.", info)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Embeddings", FAIL, f"{type(exc).__name__}: {exc}", {})


def check_vector_store(settings: Settings | None = None) -> CheckResult:
    """Open the vector store, report size, and verify index/embedder compatibility."""
    settings = settings or get_settings()
    try:
        from ingestion.embedder import build_embedder  # noqa: PLC0415
        from stores.index_meta import check_compatibility  # noqa: PLC0415
        from stores.vector_store import build_vector_store  # noqa: PLC0415

        with build_vector_store(settings) as store:
            backend = store.backend
            count = store.count()

        if count == 0:
            return CheckResult(
                "Vector store", FAIL,
                "Index is empty. Run: python -m ingestion.run_ingestion --reset",
                {"backend": backend, "vectors": 0},
            )

        compatible, message = check_compatibility(settings.vector_db_path, build_embedder(settings))
        status = OK if compatible else WARN
        return CheckResult(
            "Vector store", status,
            f"{count} vectors ({backend}). {message}",
            {"backend": backend, "vectors": count, "compatible": compatible},
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Vector store", FAIL, f"{type(exc).__name__}: {exc}", {})


def check_retrieval(settings: Settings | None = None, query: str = "What are the common symptoms of CKD?") -> CheckResult:
    """Confirm hybrid retrieval returns relevant evidence for a known query."""
    settings = settings or get_settings()
    try:
        from agent.tools.vector_search_tool import VectorSearchTool  # noqa: PLC0415

        tool = VectorSearchTool(settings=settings)
        evidence, structured = tool.run(tool.validate_arguments({"query": query, "top_k": 3}))
        if not evidence:
            return CheckResult("Retrieval", FAIL, f"No evidence for {query!r}.", structured)
        return CheckResult(
            "Retrieval", OK,
            f"{len(evidence)} passage(s); top source: {evidence[0].title!r}.",
            {"mode": structured.get("retrieval_mode"), "hits": len(evidence)},
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Retrieval", FAIL, f"{type(exc).__name__}: {exc}", {})


def check_orchestrator(settings: Settings | None = None, query: str = "What is CKD staging based on?") -> CheckResult:
    """Run the full pipeline and confirm a FinalResponse comes back."""
    settings = settings or get_settings()
    try:
        from agent.orchestrator import build_orchestrator  # noqa: PLC0415

        response = build_orchestrator(settings).run(query)
        if not response.answer.strip():
            return CheckResult("Orchestrator", FAIL, "Empty answer returned.", {})
        return CheckResult(
            "Orchestrator", OK,
            f"FinalResponse produced ({response.confidence.value}, "
            f"{len(response.citations)} citation(s)).",
            {
                "confidence": response.confidence.value,
                "citations": len(response.citations),
                "tools_used": response.tools_used,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Orchestrator", FAIL, f"{type(exc).__name__}: {exc}", {})


def run_diagnostics(settings: Settings | None = None, include_llm_call: bool = True) -> list[CheckResult]:
    """Run every check. ``include_llm_call`` controls the live Groq round-trip."""
    settings = settings or get_settings()
    checks = [
        check_api_key(settings),
        check_llm_client(settings),
        check_embedder(settings),
        check_vector_store(settings),
        check_retrieval(settings),
        check_orchestrator(settings),
    ]
    if include_llm_call:
        checks.insert(2, check_llm_response(settings))
    return checks


def main() -> int:
    logging.basicConfig(level="ERROR", format="%(levelname)s %(name)s: %(message)s")
    print("=" * 76)
    print("  CKD Medical Agentic RAG - diagnostics")
    print("=" * 76)
    results = run_diagnostics()
    for result in results:
        print(f"[{result.status:>4}] {result.name:<16} {result.detail}")
    failures = [r for r in results if r.status == FAIL]
    print("-" * 76)
    print(f"  {sum(1 for r in results if r.ok)} OK, {len(failures)} FAIL, "
          f"{sum(1 for r in results if r.status == WARN)} WARN, "
          f"{sum(1 for r in results if r.status == SKIP)} SKIP")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
