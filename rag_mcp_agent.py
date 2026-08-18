"""Run the company-policy agent with retrieval supplied through MCP."""

import argparse
import asyncio
import sys
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio

from rag_app import GENERATIONAL_MODEL


PROJECT_ROOT = Path(__file__).resolve().parent
MCP_SERVER_PATH = PROJECT_ROOT / "rag_mcp_server.py"


async def run_policy_agent(question: str) -> str:
    """Run one agent turn using the local MCP retrieval server."""
    server_params = {
        "command": sys.executable,
        "args": [str(MCP_SERVER_PATH)],
        "cwd": str(PROJECT_ROOT),
    }

    async with MCPServerStdio(
        params=server_params,
        cache_tools_list=True,
        name="Company Policy RAG MCP Server",
    ) as mcp_server:
        agent = Agent(
            name="Company Policy MCP Assistant",
            model=GENERATIONAL_MODEL,
            instructions=(
                "You answer questions about company policies. For every "
                "company-policy question, call search_company_policies first. "
                "Use only returned evidence for policy claims. A passage can "
                "be topically related without containing the requested fact. "
                "If the evidence does not explicitly answer the question, say "
                "the available policies do not specify it. Cite evidence as "
                "[filename, chunk N]. For unrelated questions, explain that "
                "you are limited to company-policy questions."
            ),
            mcp_servers=[mcp_server],
        )

        result = await Runner.run(agent, question)
        return str(result.final_output)


def main() -> None:
    """Read a question, run the MCP-backed agent, and print its response."""
    parser = argparse.ArgumentParser(
        description="Company-policy agent backed by a local MCP server"
    )
    parser.add_argument("question", nargs="*", help="Company-policy question")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = input("Ask the MCP policy agent: ").strip()
    if not question:
        raise RuntimeError("A question is required.")

    answer = asyncio.run(run_policy_agent(question))
    print("\nMCP agent response:\n")
    print(answer)


if __name__ == "__main__":
    main()
