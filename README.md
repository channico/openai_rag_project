# RAG and Agents Learning Project

**Nicodemus Chan — independent technical teaching sample**

This repository demonstrates a retrieval-augmented generation (RAG) system built
incrementally with Python and the OpenAI API. It progresses from embeddings and
cosine similarity to persistent retrieval, evaluation, deterministic abstention,
an agent tool, and an optional Model Context Protocol (MCP) integration.

The central teaching example is a realistic retrieval failure: the retriever
selects a relevant travel-policy document for a question about a maximum daily
meal allowance, but that document does not contain the requested amount. The
example distinguishes **document relevance** from **evidence sufficiency** and
shows why an agent must sometimes decline to answer even after successful
retrieval.

> This is an independent teaching sample created by Nicodemus Chan. It is not
> official OpenAI curriculum.

## Suggested review path

1. [RAG teaching deck](lesson_plans_slides/Nicodemus_Chan_RAG_Teaching_Demo.pptx)
2. [RAG lesson plan](lesson_plans_slides/Nicodemus_Chan_60_Minute_RAG_Lesson_Plan.docx)
3. [`rag_app.py`](rag_app.py) — the complete self-managed RAG application
4. [`rag_agent.py`](rag_agent.py) — the RAG retriever exposed as an in-process agent tool
5. [AI Agents teaching deck](lesson_plans_slides/Nicodemus_Chan_AI_Agents_Teaching_Demo.pptx)
6. [AI Agents lesson plan](lesson_plans_slides/Nicodemus_Chan_60_Minute_AI_Agents_Lesson_Plan.docx)
7. [`MCP_LAB.md`](MCP_LAB.md) — optional MCP extension and facilitator prompts

## What the project demonstrates

- Creating embeddings with `text-embedding-3-small`
- Ranking chunks using cosine similarity
- Sentence-aware chunking with overlap
- Grounded response generation with source labels
- Persistent and incremental document indexing
- Relevance-threshold filtering and deterministic abstention
- Retrieval evaluation using labeled calibration and held-out cases
- Separating ranking from threshold filtering during evaluation
- Batch embedding of evaluation questions
- Giving an agent access to retrieval through a function tool
- Exposing the same retrieval capability through a local MCP server

## Architecture

### Core RAG application

```text
Question
   ↓
Question embedding
   ↓
Cosine-similarity ranking
   ↓
Threshold filtering
   ↓
Retrieved policy evidence
   ↓
Grounded answer or abstention
```

### Agent integration options

```text
Direct function tool                 MCP extension

rag_agent.py                         rag_mcp_agent.py
     ↓                                      ↓
Python function call                 MCP over stdio
     ↓                                      ↓
rag_app.py                           rag_mcp_server.py
                                            ↓
                                       rag_app.py
```

`rag_app.py` remains the single implementation of indexing and retrieval. The
agent and MCP examples reuse its functions rather than duplicating the RAG
logic.

## Requirements

- Python 3.10 or later
- An OpenAI API key with access to the models used by the project
- Internet access while creating embeddings or generating answers
- PyCharm or a terminal

API requests may incur usage charges. The generated local vector index is
stored as `rag_index.json` and is intentionally excluded from Git.

## Setup

Run all commands from the repository root.

### 1. Create a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

In PyCharm, you can instead create a virtual environment from the project
interpreter settings and select it as the project interpreter.

### 2. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` uses exact versions to make this teaching demonstration
reproducible. For a maintained production application, dependency upgrades
should be tested regularly rather than remaining pinned indefinitely.

### 3. Configure the environment

macOS or Linux:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder value with your OpenAI API key. Never
commit `.env` or paste an API key into source code.

The generation model can be changed without editing Python:

```dotenv
OPENAI_GENERATIONAL_MODEL=gpt-5.6-luna
```

## Run the core RAG application

### Build the local index

```bash
python rag_app.py ingest
```

The command reads the files in `knowledge_base/`, creates embeddings for new or
changed chunks, and saves the generated index locally.

### Ask a supported policy question

```bash
python rag_app.py ask "Who must approve an employee's remote working days?"
```

### Run retrieval evaluation

```bash
python rag_app.py evaluate
```

The evaluation first sweeps thresholds on the labeled calibration set. It then
uses the frozen threshold of `0.45` on a separate held-out set. In the saved
lesson result, the retriever passed 5 of 6 held-out cases. Its failure retrieved
`travel_policy.txt` with a similarity score of approximately `0.49` for the
daily-meal-allowance question even though the policy did not specify an amount.

That result is intentionally retained as a teaching case: raising the threshold
after observing the held-out result would tune the system to its test data. A
better next step is to expand the labeled set, add evidence-sufficiency checks,
and recalibrate using validation data.

## Run the agent examples

The direct agent exposes policy retrieval as an in-process function tool:

```bash
python rag_agent.py "What is the maximum daily meal allowance?"
```

The agent may receive a topically relevant travel-policy passage, but it should
state that the available policies do not specify the amount.

## Run the optional MCP lab

The MCP-backed agent exposes the same retrieval capability through a local MCP
server:

```bash
python rag_mcp_agent.py "Who approves remote working days?"
python rag_mcp_agent.py "What is the maximum daily meal allowance?"
python rag_mcp_agent.py "What is 2 plus 2?"
```

You do not need to start `rag_mcp_server.py` separately. `rag_mcp_agent.py`
launches it as a child process, communicates with it over standard input/output,
and closes it when the agent run finishes.

Expected behavior:

1. The remote-work question retrieves and cites the relevant policy.
2. The meal-allowance question reports that the requested amount is not specified.
3. The arithmetic question is declined as outside the company-policy scope.

See [`MCP_LAB.md`](MCP_LAB.md) for the architecture, teaching prompts,
troubleshooting checklist, and production-security discussion.

## Learning progression

| File | Concept introduced |
| --- | --- |
| `lesson1_embeddings.py` | Embeddings |
| `lesson2_semantic_search.py` | Cosine similarity and semantic ranking |
| `lesson3_chunking.py` | Chunking and overlap |
| `lesson4_rag.py` | Basic retrieval-augmented generation |
| `lesson5_multiple_documents.py` | Multiple-document retrieval |
| `lesson6_persistent_rag.py` | Persistent local indexing |
| `lesson7_incremental_ingestion.py` | Fingerprints and incremental ingestion |
| `lesson8_relevance_threshold.py` | Relevance thresholds and abstention |
| `lesson9_retrieval_evaluation.py` | Labeled retrieval evaluation and calibration |
| `lesson9b_held_out_retrieval_evaluation.py` | Frozen-threshold held-out evaluation |
| `lesson10_rag_agents.py` | RAG as an agent tool |

The files under `lessons/` preserve the incremental learning journey.
`rag_app.py` and `rag_agent.py` contain the consolidated application used by the
demonstrations.

## Repository structure

```text
.
├── knowledge_base/          # Example company-policy documents
├── lesson_plans_slides/     # Teaching decks and 60-minute lesson plans
├── lessons/                 # Incremental lesson snapshots
├── MCP_LAB.md               # Optional MCP lab guide
├── rag_app.py               # Indexing, retrieval, evaluation, and generation
├── rag_agent.py             # Agents SDK function-tool example
├── rag_mcp_server.py        # MCP wrapper around the existing retriever
├── rag_mcp_agent.py         # Agent that consumes the MCP tool
├── requirements.txt         # Reproducible Python dependencies
└── .env.example             # Safe environment-variable template
```

## Troubleshooting

- **`The RAG Index does not exist`** — run `python rag_app.py ingest`.
- **Index reported as stale** — rerun ingestion after changing policy files or
  indexing settings.
- **Authentication error** — confirm that `.env` contains a valid
  `OPENAI_API_KEY` and that `.env` has not been committed.
- **Model-access error** — set `OPENAI_GENERATIONAL_MODEL` to a model available
  to your OpenAI project, then rerun the command.
- **Commands cannot find the knowledge base** — run them from the repository
  root and confirm that PyCharm uses the project root as its working directory.
- **Running `rag_mcp_server.py` appears to hang** — this is expected when it is
  started alone; it is waiting for an MCP client over stdio. Run
  `rag_mcp_agent.py` instead.

## References

- [OpenAI API quickstart](https://developers.openai.com/api/docs/quickstart)
- [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings)
- [OpenAI retrieval guide](https://developers.openai.com/api/docs/guides/retrieval)
- [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents-sdk)
- [OpenAI MCP and connectors guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)

