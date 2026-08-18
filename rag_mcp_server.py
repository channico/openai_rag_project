"""Expose the local company-policy RAG retriever as an MCP server."""

from mcp.server import MCPServer
from pydantic import BaseModel

from rag_app import CALIBRATED_THRESHOLD, load_index, retrieve


mcp_server = MCPServer(
    name="company-policy-rag",
    title="Company Policy RAG",
    description="Search the local company-policy knowledge base for evidence.",
    instructions=(
        "Use search_company_policies to find evidence before answering "
        "questions about company policy. Related evidence may not contain "
        "the requested fact."
    ),
)


class PolicyEvidence(BaseModel):
    """One retrieved policy passage."""

    source: str
    chunk_id: int
    similarity: float
    text: str


class PolicySearchResult(BaseModel):
    """Structured result returned by the policy-search tool."""

    status: str
    evidence: list[PolicyEvidence]


@mcp_server.tool(structured_output=True)
def search_company_policies(question: str, top_k: int = 4) -> PolicySearchResult:
    """Search company policies for evidence relevant to a question.

    Args:
        question: The company-policy question to research.
        top_k: Maximum number of relevant passages to return.

    Returns:
        Structured evidence with source labels and similarity scores.
    """
    records = load_index()
    retrieved_records = retrieve(
        question,
        records,
        top_k=top_k,
        min_similarity=CALIBRATED_THRESHOLD,
    )

    evidence = [
        PolicyEvidence(
            source=record["source"],
            chunk_id=record["chunk_id"],
            similarity=round(float(record["similarity"]), 4),
            text=record["text"],
        )
        for record in retrieved_records
    ]

    return PolicySearchResult(
        status="evidence_found" if evidence else "no_relevant_evidence",
        evidence=evidence,
    )


if __name__ == "__main__":
    mcp_server.run(transport="stdio")
