# MCP Lab: Reuse a RAG Capability Across Agent Hosts

## Learning objective

Explain and demonstrate the difference between an in-process function tool and
an MCP server that exposes the same capability through a standard protocol.

## Architecture

```text
User question
    |
    v
OpenAI Agents SDK host (rag_mcp_agent.py)
    |
    | MCP over stdio
    v
MCP server (rag_mcp_server.py)
    |
    v
Existing RAG engine (rag_app.py)
    |
    v
Local vector index (rag_index.json)
```

The RAG logic remains in `rag_app.py`. The MCP server only exposes that logic as
a discoverable tool, and the agent remains responsible for deciding when to use
the tool and how to interpret its evidence.

## Run the lab

From the project root:

```bash
.venv/bin/python rag_mcp_agent.py "Who approves remote working days?"
.venv/bin/python rag_mcp_agent.py "What is the maximum daily meal allowance?"
.venv/bin/python rag_mcp_agent.py "What is 2 plus 2?"
```

Expected behavior:

1. The first question retrieves and cites the remote-work policy.
2. The second may retrieve the travel policy but should state that no maximum is
   specified.
3. The third should be declined as outside the company-policy scope.

## Teaching prompts

- What changed when the local function became an MCP tool?
- Which component owns retrieval, tool transport, and final-answer judgment?
- Why does related evidence not always contain the answer?
- Which tools should require human approval in an enterprise workflow?
- How would HTTP transport, authentication, and authorization change deployment?

## Live troubleshooting checklist

- **Server will not start:** confirm the project interpreter and server path.
- **Index error:** run `rag_app.py ingest` and confirm the corpus is unchanged.
- **No tool appears:** inspect the MCP server tool list and its schema.
- **No evidence:** inspect similarity scores and the calibrated threshold.
- **Unsupported answer:** strengthen evidence-only instructions and add an eval.
- **API failure:** distinguish authentication, quota, rate-limit, and model-access
  errors before changing application code.

## Security discussion

This lab uses a read-only retrieval tool and local stdio transport. A production
MCP deployment should authenticate clients, authorize each tool, validate tool
arguments, minimize returned data, log tool activity, and require approval for
actions with side effects.
