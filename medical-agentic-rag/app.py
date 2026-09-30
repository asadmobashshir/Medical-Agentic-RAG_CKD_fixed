"""Streamlit frontend for the CKD Agentic RAG demo.

The Agentic RAG backend is unchanged.
This file only provides the Streamlit UI.
"""

from __future__ import annotations

import logging

import streamlit as st

from agent.orchestrator import Orchestrator, build_orchestrator
from agent.state import FinalResponse
from config.settings import Settings, get_settings
from stores.vector_store import build_vector_store


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="CKD Agentic RAG Demo",
    page_icon="🩺",
    layout="wide",
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# UI content
# ---------------------------------------------------------

BANNER = """
# 🩺 Chronic Kidney Disease Demo Assistant

**Chronic Kidney Disease demo assistant — NOT medical advice, demo data only,
always consult a clinician.**

Evidence-grounded answers are drawn only from a small synthetic CKD demo
corpus and a synthetic structured drug store.
"""


EXAMPLES = [
    "What is CKD staging based on?",
    "Does lisinopril interact with ibuprofen?",
    "What are ACE inhibitors used for in CKD?",
    "How are SGLT2 inhibitors relevant to CKD?",
    "Why are NSAIDs a concern in CKD?",
    "What should I know about metformin and CKD?",
]


EMPTY_METADATA = (
    "*Ask a question to see confidence, risk category and sources.*"
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "metadata" not in st.session_state:
    st.session_state.metadata = EMPTY_METADATA


# ---------------------------------------------------------
# Vector index
# ---------------------------------------------------------

def ensure_index(settings: Settings) -> int:
    """Build the vector index if it is empty."""

    try:
        with build_vector_store(settings) as store:
            count = store.count()

        if count:
            logger.info(
                "Vector index already populated: %d vectors.",
                count,
            )
            return count

    except Exception:
        logger.warning(
            "Could not read vector store; attempting ingestion.",
            exc_info=True,
        )

    logger.info(
        "Vector index is empty; running CKD corpus ingestion."
    )

    try:
        from ingestion.run_ingestion import run_ingestion

        report = run_ingestion(
            settings=settings,
            reset=False,
        )

        logger.info(
            "Ingestion complete: %d documents, %d chunks.",
            report.documents,
            report.chunks_indexed,
        )

        return report.vectors_in_store

    except Exception:
        logger.exception("Startup ingestion failed.")
        return 0


# ---------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------

@st.cache_resource
def get_app_orchestrator() -> Orchestrator:
    """Create one shared orchestrator for the Streamlit app."""

    settings = get_settings()

    settings.ensure_directories()

    ensure_index(settings)

    return build_orchestrator(settings)


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

def format_metadata(response: FinalResponse) -> str:

    lines = [
        "### Response Details",
        "",
        f"**Confidence:** {response.confidence.value}",
        "",
        f"**Score:** {response.confidence_score:.2f}",
        "",
        f"**Risk category:** {response.risk_category.value}",
        "",
    ]

    if response.tools_used:
        lines.extend(
            [
                "**Tools used:**",
                "",
                ", ".join(response.tools_used),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "**Tools used:** none",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "*Evidence confidence reflects retrieval quality and claim "
            "grounding only. It is not a probability that a medical "
            "statement is correct.*",
            "",
            "### Sources",
            "",
        ]
    )

    if not response.citations:
        lines.append("No retrieved sources.")

    else:
        for citation in response.citations:

            label = (
                citation.title
                or citation.source
                or "Untitled source"
            )

            lines.append(
                f"- **[{citation.evidence_id}]** {label}"
            )

            lines.append(
                f"  - `data_status: {citation.data_status}`"
            )

            if citation.url:
                lines.append(
                    f"  - {citation.url}"
                )

    if response.warnings:

        lines.extend(
            [
                "",
                "### Warnings",
                "",
            ]
        )

        for warning in response.warnings:
            lines.append(
                f"- {warning}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------
# Query function
# ---------------------------------------------------------

def answer_query(query: str) -> tuple[str, str]:

    query = (query or "").strip()

    if not query:
        return (
            "Please enter a question about chronic kidney disease.",
            EMPTY_METADATA,
        )

    try:

        orchestrator = get_app_orchestrator()

        logger.info(
            "Starting orchestrator for query: %s",
            query,
        )

        response = orchestrator.run(query)

        logger.info(
            "Orchestrator completed successfully."
        )

        return (
            response.answer,
            format_metadata(response),
        )

    except Exception as exc:

        logger.exception(
            "Orchestrator failed."
        )

        return (
            f"⚠️ The CKD pipeline could not complete the request.\n\n"
            f"**Error:** `{type(exc).__name__}: {exc}`",
            "The request failed before response metadata was generated.",
        )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(BANNER)

st.divider()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("🩺 CKD Agentic RAG")

    st.markdown(
        """
        **Pipeline**

        1. Query analysis
        2. Agent planning
        3. Evidence retrieval
        4. Evidence synthesis
        5. Verification
        6. Safety checks
        7. Final answer
        """
    )

    st.divider()

    st.subheader("Example Questions")

    for example in EXAMPLES:

        if st.button(
            example,
            use_container_width=True,
        ):
            st.session_state.selected_question = example

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.metadata = EMPTY_METADATA

        st.rerun()


# ---------------------------------------------------------
# Main layout
# ---------------------------------------------------------

chat_column, details_column = st.columns(
    [3, 2]
)


# ---------------------------------------------------------
# Chat
# ---------------------------------------------------------

with chat_column:

    st.subheader("Conversation")

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    selected_question = st.session_state.pop(
        "selected_question",
        None,
    )

    user_question = st.chat_input(
        "Ask a question about chronic kidney disease..."
    )

    query = (
        selected_question
        or user_question
    )

    if query:

        # Show user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message("user"):

            st.markdown(query)

        # Run existing Agentic RAG backend
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching CKD evidence and generating answer..."
            ):

                answer, metadata = answer_query(
                    query
                )

            st.markdown(answer)

        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.metadata = metadata

        st.rerun()


# ---------------------------------------------------------
# Response details
# ---------------------------------------------------------

with details_column:

    st.subheader("Response Details")

    st.markdown(
        st.session_state.metadata
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Research prototype. The bundled corpus and drug records are "
    "synthetic demo data and are not validated medical sources. "
    "This system does not diagnose, does not recommend doses, and "
    "is not a substitute for a qualified clinician."
)
