"""Retrieval and document grading agent with corrective RAG loop."""

from __future__ import annotations

import re
import threading
import time
from datetime import UTC, datetime

from config.settings import settings
from core.agents.router import call_llm_async
from core.state import DocumentGrade, GraphState  # noqa: TC001
from ingestion.metadata import UserContext
from utils.logging import get_logger
from utils.observability import trace_retrieval

logger = get_logger(__name__)

# Module-level lazy singletons.
# RLock (reentrant) is required because _get_hybrid_searcher holds the lock
# while calling _get_bm25_index, which acquires the same lock recursively.
# A plain Lock would deadlock the calling thread on first invocation.
_hybrid_searcher = None
_reranker = None
_bm25_index = None
_init_lock = threading.RLock()


def _get_bm25_index():
    """Lazily initialize and return the shared BM25Index instance.

    Returns:
        A BM25Index that gets populated during document ingestion.
    """
    global _bm25_index
    if _bm25_index is None:
        with _init_lock:
            if _bm25_index is None:
                from retrieval.hybrid_search import BM25Index

                _bm25_index = BM25Index()
    return _bm25_index


def _get_hybrid_searcher():
    """Lazily initialize and return the HybridSearcher instance.

    Thread-safe via double-checked locking pattern.

    Returns:
        A configured HybridSearcher with QdrantManager, EmbeddingService,
        and shared BM25Index.
    """
    global _hybrid_searcher
    if _hybrid_searcher is None:
        with _init_lock:
            if _hybrid_searcher is None:  # Double-check pattern
                from retrieval.embeddings import EmbeddingService
                from retrieval.hybrid_search import HybridSearcher
                from retrieval.qdrant_client import QdrantManager

                qdrant_manager = QdrantManager()
                embedding_service = EmbeddingService()
                bm25_index = _get_bm25_index()
                _hybrid_searcher = HybridSearcher(
                    qdrant_manager=qdrant_manager,
                    embedding_service=embedding_service,
                    bm25_index=bm25_index,
                )
    return _hybrid_searcher


def _get_reranker():
    """Lazily initialize and return the Reranker instance.

    Thread-safe via double-checked locking pattern.

    Returns:
        A configured Reranker instance.
    """
    global _reranker
    if _reranker is None:
        with _init_lock:
            if _reranker is None:  # Double-check pattern
                from retrieval.reranker import Reranker

                _reranker = Reranker()
    return _reranker


def _get_grading_prompt(query: str, document_text: str) -> str:
    """Build the grading prompt for a single document (fallback mode).

    Args:
        query: The user's query.
        document_text: The text of the document to evaluate.

    Returns:
        Formatted prompt string for the LLM.
    """
    return (
        "You are a document relevance grader. Given a user query and a document, "
        "determine if the document is relevant to answering the query.\n\n"
        f"Query: {query}\n\n"
        f"Document: {document_text[:500]}\n\n"
        "Is this document relevant to the query? "
        "Respond with ONLY 'yes' or 'no', nothing else."
    )


def _get_batch_grading_prompt(query: str, documents: list[DocumentGrade]) -> str:
    """Build a batch grading prompt for all documents at once.

    This is significantly more efficient than grading each document
    individually, as it requires only a single LLM call.

    Args:
        query: The user's query.
        documents: List of documents to grade.

    Returns:
        Formatted prompt string for batch grading.
    """
    doc_lines: list[str] = []
    for i, doc in enumerate(documents, start=1):
        text_preview = doc["text"][:400].replace("\n", " ")
        doc_lines.append(f"DOC {i}: {text_preview}")

    docs_str = "\n\n".join(doc_lines)

    return (
        "You are a document relevance grader. For each document below, "
        "determine if it is relevant to answering the query.\n\n"
        f"Query: {query}\n\n"
        f"Documents:\n{docs_str}\n\n"
        "For EACH document, respond on a separate line with:\n"
        "DOC N: yes   (if relevant)\n"
        "DOC N: no    (if not relevant)\n\n"
        "Respond with ONLY the DOC lines, nothing else."
    )


def _parse_batch_grading(response: str, num_docs: int) -> list[bool] | None:
    """Parse batch grading response into per-document relevance flags.

    Args:
        response: LLM response with DOC N: yes/no lines.
        num_docs: Expected number of documents.

    Returns:
        List of boolean relevance flags, or None if parsing failed.
    """
    lines = [line.strip() for line in response.split("\n") if line.strip()]

    # Parse each DOC line
    parsed: dict[int, bool] = {}
    for line in lines:
        match = re.match(r"DOC\s+(\d+)\s*:\s*(yes|no)", line, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1  # 0-based
            is_relevant = match.group(2).lower() == "yes"
            parsed[idx] = is_relevant

    # Check if we got enough valid results
    if len(parsed) < num_docs * 0.5:
        return None  # Signal fallback to individual grading

    # Build results list, defaulting to True if parsing failed for a doc
    results: list[bool] = []
    for i in range(num_docs):
        results.append(parsed.get(i, True))  # Default to relevant on parse failure

    return results


def _rrf_fuse_results(rankings: list[list], k: int = 60) -> list:
    """Reciprocal-Rank-Fuse multiple lists of SearchResult.

    Each list is treated as an independent retrieval ranking. The same
    doc may appear in multiple lists at different ranks; we sum the RRF
    contributions and re-sort. Deduplication is by `id`.

    Args:
        rankings: List of ranked SearchResult lists.
        k: RRF constant (60 is the canonical default).

    Returns:
        Single deduplicated, fused list ordered by descending RRF score.
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            doc_id = result.id
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = result
    sorted_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)
    fused: list = []
    for doc_id in sorted_ids:
        result = doc_map[doc_id]
        fused_result = result.model_copy(update={"score": fused_scores[doc_id]})
        fused.append(fused_result)
    return fused


async def _generate_fusion_queries(original: str, n: int, prefer_cloud: bool = False) -> list[str]:
    """Ask the LLM for N-1 reformulations of the original query (RAG Fusion).

    The original query is always included as one of the N. Reformulations
    are designed to surface chunks that the original might miss because of
    vocabulary mismatch or under-specification.

    Args:
        original: User's original query.
        n: Total queries desired (N-1 will be generated).
        prefer_cloud: Whether to route the reformulation LLM call to the
            configured cloud provider (still subject to the sensitivity gate
            — fusion sees only the query string, never doc content, so it
            is safe to route to cloud at LOW sensitivity).

    Returns:
        List of query strings (length up to N, original always first).
    """
    if n <= 1:
        return [original]
    prompt = (
        f"Generate {n - 1} alternative phrasings of the user's question. Each "
        "rewrite should preserve the original meaning but vary the vocabulary, "
        "specificity, or angle so that it would retrieve different but still "
        "relevant document chunks. Do NOT answer the question.\n\n"
        "STRICT FORMAT: one rewritten query per line, no numbering, no bullets, "
        "no preamble, no explanation. No `<think>` blocks.\n\n"
        f"Original question: {original}\n\n"
        "Rewrites:"
    )
    try:
        response = await call_llm_async(
            prompt,
            system_prompt="You are a search query rewriter.",
            sensitivity_level="low",  # Reformulation never sees doc content.
            prefer_cloud=prefer_cloud,
        )
        # Strip <think>...</think> blocks if the LLM ran in reasoning mode.
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
        lines = [
            line.strip().lstrip("-*0123456789. ").strip()
            for line in cleaned.splitlines()
            if line.strip()
        ]
        rewrites = [line for line in lines if line and line.lower() != original.lower()]
        rewrites = rewrites[: n - 1]
        return [original, *rewrites] if rewrites else [original]
    except Exception as exc:
        logger.warning("fusion_query_generation_failed", error=str(exc))
        return [original]


async def retrieve_documents(state: GraphState) -> dict:
    """Retrieve documents using hybrid search with RBAC filtering.

    When ``settings.rag_fusion_enabled`` is True, generates
    ``settings.rag_fusion_n_queries`` query reformulations, retrieves each
    in parallel, and Reciprocal-Rank-Fuses the results. This boosts recall
    on vocabulary-mismatched or under-specified queries at the cost of one
    extra LLM call + (N-1) extra Qdrant searches.

    Optionally reranks the final fused list for precision.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with documents list and audit_trail entry.
    """
    query = state.get("rewritten_query") or state["query"]
    user_context_dict = state["user_context"]

    logger.info("retrieving_documents", query_len=len(query))

    user_context = UserContext(**user_context_dict)

    # HyDE (opt-in): embed a hypothetical answer alongside the query so the
    # dense vector lands in document-space. Skipped for ``out_of_scope`` and
    # ``simple`` queries where the cheap regex query would already match.
    search_query = query
    if settings.hyde_enabled and state.get("query_type") in ("complex", ""):
        from retrieval.hyde import generate_hyde_passage

        search_query = await generate_hyde_passage(
            query,
            sensitivity_level=state.get("query_sensitivity", "low"),
            prefer_cloud=state.get("prefer_cloud", False),
        )

    searcher = _get_hybrid_searcher()

    # Self-query (opt-in): extract structured metadata filters from the query
    # and merge them with the RBAC filter for pre-filtered retrieval.
    extra_filter = None
    if settings.self_query_enabled:
        from retrieval.self_query import build_qdrant_filter_conditions, extract_self_query_filters

        sq_filters = await extract_self_query_filters(
            query,
            sensitivity_level=state.get("query_sensitivity", "low"),
            prefer_cloud=state.get("prefer_cloud", False),
        )
        if sq_filters:
            conditions = build_qdrant_filter_conditions(sq_filters)
            extra_filter = searcher._qdrant.build_combined_filter(user_context, conditions)
            logger.info("self_query_applied", filters=list(sq_filters.keys()))

    start = time.perf_counter()
    try:
        # RAG Fusion: parallel search across multiple query reformulations.
        if settings.rag_fusion_enabled and settings.rag_fusion_n_queries > 1:
            queries = await _generate_fusion_queries(
                search_query,
                settings.rag_fusion_n_queries,
                prefer_cloud=state.get("prefer_cloud", False),
            )
            logger.info("rag_fusion_queries", count=len(queries), queries=queries)
            import asyncio as _asyncio

            ranking_lists = await _asyncio.gather(
                *(
                    searcher.search(
                        query=q,
                        user_context=user_context,
                        top_k=settings.top_k,
                        extra_filter=extra_filter,
                    )
                    for q in queries
                ),
                return_exceptions=False,
            )
            search_results = _rrf_fuse_results(ranking_lists)[: settings.top_k]
        else:
            search_results = await searcher.search(
                query=search_query,
                user_context=user_context,
                top_k=settings.top_k,
                extra_filter=extra_filter,
            )

        # Optionally rerank. Gated behind settings.enable_reranker because
        # the first call downloads a ~600MB cross-encoder from HuggingFace
        # with no progress feedback — easily mistaken for a hang.
        if settings.enable_reranker and search_results:
            reranker = _get_reranker()
            if reranker.is_available():
                search_results = reranker.rerank(
                    query=query,
                    documents=search_results,
                    top_k=settings.rerank_top_k,
                )

        # Convert SearchResults to DocumentGrade objects
        documents: list[DocumentGrade] = []
        for result in search_results:
            doc_grade: DocumentGrade = {
                "doc_id": result.id,
                "text": result.text,
                "score": result.score,
                "relevant": False,  # Will be set by grader
                "metadata": result.metadata,
            }
            documents.append(doc_grade)

        logger.info("documents_retrieved", count=len(documents))

    except Exception as exc:
        logger.error("retrieve_documents_failed", error=str(exc))
        documents = []
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        trace_retrieval(
            query=query,
            num_results=len(documents),
            latency_ms=elapsed_ms,
            method="hybrid",
        )

    return {
        "documents": documents,
        "audit_trail": [
            {
                "node": "retriever",
                "action": "retrieve_documents",
                "query": query,
                "documents_count": len(documents),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }


async def _grade_single_document(
    query: str, doc: DocumentGrade, prefer_cloud: bool = False
) -> DocumentGrade:
    """Grade a single document for relevance (fallback for batch failures).

    Args:
        query: The user's query.
        doc: Document to grade.
        prefer_cloud: Whether to route the grading LLM call to cloud
            (subject to sensitivity gate via the inference router).

    Returns:
        DocumentGrade with 'relevant' field populated.
    """
    prompt = _get_grading_prompt(query, doc["text"])
    response = await call_llm_async(
        prompt,
        system_prompt="You are a document relevance grader.",
        prefer_cloud=prefer_cloud,
    )
    is_relevant = response.strip().lower().startswith("yes")
    graded_doc: DocumentGrade = {
        **doc,
        "relevant": is_relevant,
    }
    return graded_doc


async def _grade_documents_batch(
    query: str, documents: list[DocumentGrade], prefer_cloud: bool = False
) -> list[DocumentGrade]:
    """Grade all documents in a single LLM call for efficiency.

    Falls back to individual grading if batch parsing fails.

    Args:
        query: The user's query.
        documents: Documents to grade.
        prefer_cloud: Whether to route the grading LLM call to cloud.

    Returns:
        List of DocumentGrade with 'relevant' field populated.
    """
    import asyncio

    if not documents:
        return []

    if len(documents) == 1:
        # Single document — use simple prompt
        return [await _grade_single_document(query, documents[0], prefer_cloud=prefer_cloud)]

    # Batch grading for multiple documents
    prompt = _get_batch_grading_prompt(query, documents)
    response = await call_llm_async(
        prompt,
        system_prompt="You are a document relevance grader.",
        prefer_cloud=prefer_cloud,
    )

    relevance_flags = _parse_batch_grading(response, len(documents))

    # Validate: if batch parsing failed, fall back to individual grading
    if relevance_flags is None:
        logger.warning(
            "batch_grading_parse_failed",
            expected=len(documents),
            falling_back="individual_grading",
        )
        return await asyncio.gather(
            *[_grade_single_document(query, doc, prefer_cloud=prefer_cloud) for doc in documents]
        )

    graded: list[DocumentGrade] = []
    for doc, is_relevant in zip(documents, relevance_flags, strict=False):
        graded_doc: DocumentGrade = {
            **doc,
            "relevant": is_relevant,
        }
        graded.append(graded_doc)

    return graded


async def grade_documents(state: GraphState) -> dict:
    """Grade each retrieved document for relevance using the LLM.

    Uses batch grading (single LLM call for all documents) for efficiency,
    falling back to individual grading if batch parsing fails.

    Args:
        state: Current graph state with documents list.

    Returns:
        Partial state update with relevant_documents, relevance_ratio,
        updated documents, and audit_trail entry.
    """
    query = state.get("rewritten_query") or state["query"]
    documents = state.get("documents", [])

    logger.info("grading_documents", count=len(documents))

    if not documents:
        return {
            "documents": [],
            "relevant_documents": [],
            "relevance_ratio": 0.0,
            "audit_trail": [
                {
                    "node": "retriever",
                    "action": "grade_documents",
                    "total_documents": 0,
                    "relevant_count": 0,
                    "relevance_ratio": 0.0,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    # Use batch grading for efficiency (single LLM call)
    graded_documents = await _grade_documents_batch(
        query, documents, prefer_cloud=state.get("prefer_cloud", False)
    )

    relevant_documents = [doc for doc in graded_documents if doc["relevant"]]
    total = len(graded_documents)
    relevance_ratio = len(relevant_documents) / total if total > 0 else 0.0

    logger.info(
        "documents_graded",
        total=total,
        relevant=len(relevant_documents),
        relevance_ratio=relevance_ratio,
    )

    return {
        "documents": graded_documents,
        "relevant_documents": relevant_documents,
        "relevance_ratio": relevance_ratio,
        "audit_trail": [
            {
                "node": "retriever",
                "action": "grade_documents",
                "total_documents": total,
                "relevant_count": len(relevant_documents),
                "relevance_ratio": relevance_ratio,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ],
    }


def should_retry(state: GraphState) -> str:
    """Determine whether to retry retrieval or proceed to synthesis.

    Conditional edge function for the corrective RAG loop.

    Args:
        state: Current graph state with relevance_ratio and retry_count.

    Returns:
        "rewrite" if relevance is too low and retries remain, else "generate".
    """
    relevance_ratio = state.get("relevance_ratio", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)

    if relevance_ratio < settings.relevance_retry_threshold and retry_count < max_retries:
        logger.info(
            "retry_decision",
            decision="rewrite",
            relevance_ratio=relevance_ratio,
            retry_count=retry_count,
        )
        return "rewrite"

    logger.info(
        "retry_decision",
        decision="generate",
        relevance_ratio=relevance_ratio,
        retry_count=retry_count,
    )
    return "generate"
