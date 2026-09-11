"""Gradio frontend for the CKD Agentic RAG demo.

Hugging Face Spaces entry point. This module is a **view layer only**: it calls
the existing :class:`~agent.orchestrator.Orchestrator` and renders what comes
back. Planning, retrieval, synthesis, verification, confidence scoring,
guardrails and citation handling all stay where they are - nothing in this file
recomputes or second-guesses them.

Run locally with::

    python app.py
"""

from __future__ import annotations

import logging
import os
from typing import Any

import gradio as gr

from agent.orchestrator import Orchestrator, build_orchestrator
from agent.state import FinalResponse
from config.settings import Settings, get_settings
from stores.vector_store import build_vector_store

logger = logging.getLogger(__name__)

BANNER = (
    "### 🩺 Chronic Kidney Disease Demo Assistant\n"
    "**Chronic Kidney Disease demo assistant — NOT medical advice, demo data only, "
    "always consult a clinician.**\n\n"
    "Evidence-grounded answers are drawn only from a small synthetic CKD demo corpus "
    "and a synthetic structured drug store. This assistant is specialised for CKD; "
    "questions about other conditions are declined rather than answered from memory."
)

EXAMPLES = [
    "What is CKD staging based on?",
    "Does lisinopril interact with ibuprofen?",
    "What are ACE inhibitors used for in CKD?",
    "How are SGLT2 inhibitors relevant to CKD?",
    "Why are NSAIDs a concern in CKD?",
    "What should I know about metformin and CKD?",
]

FRIENDLY_ERROR = (
    "Sorry, something went wrong while processing your question. Please try again."
)

EMPTY_METADATA = (
    "*Ask a question to see confidence, risk category and sources.*"
)

_orchestrator: Orchestrator | None = None


def ensure_index(settings: Settings) -> int:
    """Build the vector index on first run if it is empty.

    ``data/processed/`` is gitignored, so a freshly cloned checkout - or a fresh
    Hugging Face Space - starts with no index. Rather than silently answering
    every question with "no evidence retrieved", ingest the bundled CKD demo
    corpus once at startup. Returns the number of indexed vectors.
    """
    try:
        with build_vector_store(settings) as store:
            count = store.count()
    except Exception:  # noqa: BLE001 - fall through to ingestion below
        logger.warning("Could not read the vector store; attempting ingestion.", exc_info=True)
        count = 0

    if count:
        logger.info("Vector index already populated (%d vectors).", count)
        return count

    logger.info("Vector index is empty; running ingestion over the bundled CKD corpus.")
    try:
        from ingestion.run_ingestion import run_ingestion  # noqa: PLC0415 - startup only

        report = run_ingestion(settings=settings, reset=False)
        logger.info(
            "Ingestion complete: %d document(s), %d chunk(s) indexed.",
            report.documents,
            report.chunks_indexed,
        )
        return report.vectors_in_store
    except Exception:  # noqa: BLE001 - the app must still start
        logger.exception("Startup ingestion failed; retrieval will return no evidence.")
        return 0


def get_app_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Return the process-wide orchestrator, building it on first use.

    Construction is lazy so importing this module never loads an embedding model,
    opens a database or requires ``GROQ_API_KEY`` - all of which matter on
    Hugging Face Spaces, where import happens before secrets are necessarily used.
    """
    global _orchestrator  # noqa: PLW0603 - one shared instance by design
    if _orchestrator is None:
        settings = settings or get_settings()
        settings.ensure_directories()
        ensure_index(settings)
        _orchestrator = build_orchestrator(settings)
    return _orchestrator


def format_metadata(response: FinalResponse) -> str:
    """Render orchestrator metadata as Markdown.

    Every value shown here comes straight from the orchestrator. The frontend
    computes no confidence of its own and invents no source metadata.
    """
    lines = [
        "#### Response details",
        f"**Confidence:** {response.confidence.value}  ",
        f"**Score:** {response.confidence_score:.2f}  ",
        f"**Risk category:** {response.risk_category.value}  ",
        f"**Tools used:** {', '.join(response.tools_used) if response.tools_used else 'none'}",
        "",
        "*Evidence confidence reflects retrieval quality and claim grounding only. "
        "It is not a probability that a medical statement is correct.*",
        "",
        "#### Sources",
    ]

    if not response.citations:
        lines.append("No retrieved sources.")
    else:
        for citation in response.citations:
            label = citation.title or citation.source or "Untitled source"
            entry = f"- **[{citation.evidence_id}]** {label} — `data_status: {citation.data_status}`"
            if citation.url:
                entry += f"  \n  <{citation.url}>"
            lines.append(entry)

    if response.warnings:
        lines.append("")
        lines.append("#### Warnings")
        lines.extend(f"- {warning}" for warning in response.warnings)

    return "\n".join(lines)


def answer_query(message: str, history: list[Any] | None = None) -> tuple[str, str]:
    """Run one query through the existing pipeline.

    Returns ``(answer_markdown, metadata_markdown)``. Exceptions are logged in
    full and replaced with a generic message, so no traceback, secret or internal
    path reaches the UI.
    """
    query = (message or "").strip()
    if not query:
        return "Please enter a question about chronic kidney disease.", EMPTY_METADATA

    try:
        response = get_app_orchestrator().run(query)
    except Exception:  # noqa: BLE001 - the UI must never surface internals
        logger.exception("Orchestrator failed while handling a UI query")
        return FRIENDLY_ERROR, "*No response metadata available.*"

    return response.answer, format_metadata(response)


def build_interface() -> gr.Blocks:
    """Construct the Gradio UI.

    ``gr.Blocks`` rather than ``gr.ChatInterface`` because the metadata panel has
    to update in step with each answer, which ``ChatInterface`` does not cleanly
    support alongside its own managed chat state.
    """
    with gr.Blocks(title="CKD Agentic RAG Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(BANNER)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=430,
                    type="messages",
                    show_copy_button=True,
                )
                question = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Why are NSAIDs a concern in CKD?",
                    lines=2,
                    max_lines=4,
                )
                with gr.Row():
                    submit = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear")

            with gr.Column(scale=2):
                metadata = gr.Markdown(EMPTY_METADATA, label="Response details")

        gr.Examples(
            examples=EXAMPLES,
            inputs=question,
            label="Example questions (answerable from the demo corpus)",
        )

        gr.Markdown(
            "---\n"
            "Research prototype. The bundled corpus and drug records are **synthetic "
            "demo data**, contain no numeric doses, and are not validated medical "
            "sources. This system does not diagnose, does not recommend doses, and is "
            "not a substitute for a qualified clinician."
        )

        def respond(message: str, chat_history: list[dict[str, str]]):
            chat_history = list(chat_history or [])
            query = (message or "").strip()
            if not query:
                return chat_history, gr.update(), ""
            answer, details = answer_query(query)
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            return chat_history, details, ""

        submit.click(respond, [question, chatbot], [chatbot, metadata, question])
        question.submit(respond, [question, chatbot], [chatbot, metadata, question])
        clear.click(lambda: ([], EMPTY_METADATA, ""), None, [chatbot, metadata, question])

    return demo


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if not settings.has_llm_credentials:
        logger.warning(
            "GROQ_API_KEY is not set. Running in evidence-only mode: retrieval works "
            "and answers are extractive rather than synthesised."
        )
    build_interface().queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=False,  # never surface tracebacks in the browser
    )


demo = build_interface()

if __name__ == "__main__":
    main()
