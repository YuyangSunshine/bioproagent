"""Public retrieval interface examples.

This module intentionally exposes only extension points. Private retrieval assets
(e.g., internal embedding index/chunks and production connectors) are not shipped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    source: str
    content: str


class ExampleRetriever:
    """Minimal retriever stub users can replace with their own implementation."""

    def invoke(self, query: str) -> str:
        return (
            "[Retrieval interface placeholder] Local vector index is not included in the public release. "
            "Please configure your own retrieval backend and return relevant chunks here. "
            f"Query: {query}"
        )


class CustomFaissRetriever(ExampleRetriever):
    """Backward-compatible class name kept for public API stability."""


# Public default object used by tool layer.
default_retriever = ExampleRetriever()


def get_retriever(k: int = 3) -> CustomFaissRetriever:
    _ = k
    return CustomFaissRetriever()


def search_pubmed(query: str, max_results: int = 2) -> str:
    _ = max_results
    return (
        "[Retrieval interface placeholder] PubMed connector is intentionally omitted in public release. "
        "Please plug in your own literature retriever. "
        f"Query: {query}"
    )


def Protocol_search(query: str) -> str:
    return (
        "[Retrieval interface placeholder] Protocol repository connector is omitted. "
        "Please configure your own source. "
        f"Query: {query}"
    )


def Web_search(query: str) -> str:
    return (
        "[Retrieval interface placeholder] Web search connector is omitted. "
        "Please configure your own source. "
        f"Query: {query}"
    )
