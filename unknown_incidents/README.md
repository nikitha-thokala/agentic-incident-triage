# unknown_incidents

The unknown-incident path of [agentic-incident-triage](../README.md). It takes tickets that match no known cluster
tag and runs them through a three-tier cascade, where each tier runs only when the cheaper one could not decide:

| Tier | File | How it decides | Cost |
|---|---|---|---|
| 0 | `tier0_triage.py` | Local open-source embeddings (fastembed, all-MiniLM-L6-v2) + Pinecone search over resolved incidents: exact alert code, then agreeing near neighbours, with near-duplicates collapsed to one vote | no LLM |
| 1 | `tier1_triage.py` | One LLM call (OpenAI Agents SDK, structured output) with the retrieved incidents as context; routes only above a 0.80 confidence gate and never to a class outside the list | 1 call |
| 2 | `tier2_triage.py` | A tool-using investigator that calls a read-only MCP server (`tier2_mcp_server.py`: similar incidents, incident record, KB search, allowlisted store diagnostics) and routes or escalates to L2 with work notes generated from, and citing, what it retrieved (agentic RAG); capped at 8 turns | a few calls |

Only Tier 0 is pure vector search: its neighbours vote and nothing is generated. Tier 1 is retrieval-augmented
classification (retrieved incidents in the prompt, a structured class out). Tier 2 is agentic RAG: the MCP server is
its retrieval layer, and the model writes the evidence and work notes from what the tools return.

Evaluation lives here too: `evaluate.py` replays `eval_set.jsonl` through the cascade and scores routing, wrong
routes, escalations, cost and latency per tier; `judge.py` is an LLM-as-judge for Tier 2 evidence and work notes,
with a self-test of known good and bad cases. Results are saved to `unknown_incidents/eval_results/`; the committed
files are earlier Tier 0 runs (an eval and a threshold sweep) to compare against.

## Setup

This path needs API keys and a few more packages. All commands run from the repo root.

```bash
pip install -r requirements-unknown.txt
cp .env.example .env        # set OPENAI_API_KEY and PINECONE_API_KEY
python -m unknown_incidents.tier0_triage --load   # one time: create the Pinecone index and upload past incidents
```

The first run downloads the small open-source embedding model; after that it runs locally.

## Run

```bash
python -m unknown_incidents.tier0_triage                  # Tier 0 only on the demo tickets (no LLM)
python -m unknown_incidents.tier2_triage                  # full cascade on the demo tickets
python -m unknown_incidents.tier2_triage "your ticket"    # one ticket
```

## Evaluate

```bash
python -m unknown_incidents.evaluate                       # Tier 0 only, free
python -m unknown_incidents.evaluate --sweep               # Tier 0 similarity-threshold sweep, free
python -m unknown_incidents.evaluate --max-tier 2          # full cascade (asks before calling OpenAI)
python -m unknown_incidents.judge --selftest               # check the judge before trusting its scores
python -m unknown_incidents.evaluate --max-tier 2 --judge  # also judge Tier 2 work notes
```

Use a different model for the judge than for Tier 2 (`JUDGE_MODEL` vs `TIER2_MODEL` in `.env`): a model tends to
favour its own answers.
