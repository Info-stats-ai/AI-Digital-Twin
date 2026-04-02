# AI Digital Twin

Production-oriented AI assistant with:
- authenticated per-user memory
- MoE-style expert routing
- async agent orchestration for lower latency
- telemetry for routing/evaluation
- file-based RAG (`PDF` + `image`) with user-isolated retrieval

## What This Project Is

This project is an AI chatbot platform designed for "digital twin" behavior:
- remembers important user context over time
- routes each query to the most suitable expert model
- supports both normal chat and document-assisted chat (RAG)
- surfaces agent execution steps for observability

Core backend is in `backend/server.py`, frontend chat UI in `frontend/components/twin.tsx`.

## Why This Architecture

Single-model chatbots are easy to build but hard to scale in production. They often become:
- too expensive (always calling large models)
- too slow (no routing or async orchestration)
- hard to debug (no route/agent telemetry)
- risky for privacy/compliance (weak memory boundaries)

This project addresses those issues by combining:
- expert routing (small model when possible, fallback when needed)
- strict per-user memory isolation
- async multi-agent pipeline
- route and latency telemetry for continuous tuning

## How It Works (High Level)

### 1) Authentication + User Isolation
- user logs in with JWT auth
- every request is bound to `user_id`
- memory, sessions, and document vectors are scoped by user

### 2) Async Agent Pipeline
For `/chat` and `/chat/rag`, backend runs a pipeline like:
- `PlannerAgent`
- `RouterAgent`
- `MemoryRetrieverAgent`
- `GuardrailAgent`
- `LLMExpertAgent`
- `MemoryWriterAgent`

`RouterAgent` and `MemoryRetrieverAgent` run concurrently using `asyncio.gather(...)` to reduce end-to-end latency.

### 3) Expert Routing (MoE-style)
Queries are routed by:
- LR router prediction + confidence threshold
- heuristic override for expert-specific intents

Expert routes are configured in `EXPERT_TO_PROVIDER_ROUTE` and exposed via `GET /experts`.

### 4) Memory
- short-term: session conversation turns
- long-term: extracted important user facts (`long_memory.json`)
- guardrail sanitization before context injection (basic email/phone redaction)

### 5) File RAG (`/chat/rag`)
User can upload PDF/image with prompt:
- parse content (PDF text extraction, image-to-text)
- chunk text
- create embeddings
- store vectors per-user
- retrieve top-k relevant chunks for current query
- inject retrieved context into expert LLM prompt

## Expert Labels and Model Names

Current expert labels:
- `ml_expert`
- `math_reasoning_expert`
- `dl_expert`
- `genai_expert`
- `research_expert`
- `agentic_ai_expert`
- `rag_expert`
- `llm_eval_expert`
- `friendly_conversation_expert`
- `multimodal_expert`

Also used:
- `memory_factual_expert`
- `technical_expert`
- `gpt_fallback`

Model names are configured in environment variables and route map (`backend/server.py`).  
You can inspect active mapping at runtime with:
- `GET /experts`

### Open-Source Expert Model Mapping (Ollama)

Configured expert env vars and default OSS models:
- `ML_EXPERT_MODEL=llama3.1:8b`
- `MATH_EXPERT_MODEL=deepseek-r1:8b`
- `DL_EXPERT_MODEL=qwen2.5-coder:7b-instruct`
- `GENAI_EXPERT_MODEL=qwen2.5:7b`
- `RESEARCH_EXPERT_MODEL=mistral:7b`
- `AGENTIC_EXPERT_MODEL=llama3.1:8b`
- `RAG_EXPERT_MODEL=qwen2.5:7b`
- `LLM_EVAL_EXPERT_MODEL=llama3.1:8b`
- `FRIENDLY_EXPERT_MODEL=phi3:mini`

### Ollama Runtime Verification

Ollama is installed and working on this machine.

Verification command:
```bash
ollama --version
ollama list
```

Expected model list includes:
- `llama3.1:8b`
- `deepseek-r1:8b`
- `qwen2.5-coder:7b-instruct`
- `qwen2.5:7b`
- `mistral:7b`
- `phi3:mini`

## Is It Fast or Slow?

Short answer: **moderately fast locally, but production speed depends on model choice and document size**.

### Usually Fast
- plain text chat with small local expert model
- short context windows
- no file ingestion

### Usually Slower
- fallback to larger cloud model
- PDF/image ingestion + embedding on request path
- large files, many chunks, high retrieval top-k

### Main latency contributors
- model inference time (largest factor)
- document parsing/OCR
- embedding API calls
- context size sent to LLM

## Production Challenges You Will Face

### 1) Latency Spikes
- file parsing + embedding in request path can increase p95 significantly
- multimodal calls are usually slower than text-only

Mitigation:
- move ingestion to background jobs
- cache doc parsing + embeddings
- cap file size/pages and chunk count

### 2) Cost Growth
- fallback model usage can dominate spend
- embedding many chunks per upload is expensive

Mitigation:
- enforce chunk and upload limits
- tune router threshold with telemetry
- add cost-aware routing policy

### 3) Retrieval Quality Drift
- noisy chunks reduce answer quality
- stale or duplicate long-memory facts can pollute prompts

Mitigation:
- better chunking/reranking
- memory dedup + decay
- regular eval and hard-example retraining

### 4) Privacy and Compliance Risk
- accidental cross-user retrieval is critical severity
- uploaded docs may contain PII/secrets

Mitigation:
- mandatory `user_id` filters in retrieval
- PII scanning/redaction for storage and logs
- encrypted storage and strict retention policies

### 5) Reliability/Operations
- local model runtime availability (Ollama) failures
- external provider/API outages
- uneven performance under concurrency

Mitigation:
- retries with backoff + circuit breaking
- health checks and provider failover
- rate limits and queue-based ingestion

## Known Current Limitations

- hybrid router (LR + heuristics), not a full neural gate yet
- vector store is JSONL/file-based, not a production DB
- basic sanitization only; needs stronger DLP/PII controls
- no streaming token output yet
- ingestion currently in request path (can increase latency)

## Recommended Production Upgrades (Next Steps)

1. Replace file-based vectors with `pgvector`/`Qdrant`
2. Background ingestion workers (Celery/RQ/queue)
3. Add reranker for top-k chunk quality
4. Structured error codes + retry policies
5. Strong observability dashboards (route, cost, p50/p95, fallback rate)
6. Harden security (secrets manager, encryption at rest, audit trails)

## Local Run

### Backend
```bash
cd "/Users/omkarthakur/Desktop/Digital Twin/backend"
source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Ollama (required for local expert routes)
```bash
# terminal 1 (keep running)
ollama serve

# terminal 2 (one-time pulls)
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
ollama pull qwen2.5-coder:7b-instruct
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull phi3:mini
```

### Frontend
```bash
cd "/Users/omkarthakur/Desktop/Digital Twin/frontend"
npm install
npm run dev
```

Open:
- `http://localhost:3000`

## Useful Endpoints

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `POST /chat` (text chat)
- `POST /chat/rag` (prompt + PDF/image files)
- `GET /experts` (expert -> provider/model mapping)
- `GET /sessions`
- `GET /memory/export`

## Evaluation, Training, and Routing Assets

### Router Evaluation / Analysis Scripts
- `scripts/phase2_3_eval_router.py`  
  Offline router evaluation + threshold sweep (accuracy, macro-F1, fallback rate, cost proxy).

- `scripts/phase2_3_analyze_telemetry.py`  
  Analyze live route telemetry and generate summary + misroute candidates.

- `scripts/phase2_3_enrich_from_telemetry.py`  
  Convert hard/low-confidence telemetry samples into enrichment candidates for retraining.

### Neural MoE Router Training Script
- `scripts/train_neural_moe_from_urls.py`  
  Import public data via URL, auto-label expert routes using rules, train a neural MoE-style router, and export artifacts/reports.

### Generated Data and Artifacts
- Dataset: `data/neural_moe_imported_dataset.jsonl`
- Neural model: `artifacts/router_neural_moe.pt`
- Neural metadata: `artifacts/router_neural_moe_meta.json`
- Metrics: `reports/neural_moe/metrics.json`
- Reports: `reports/neural_moe/val_report.json`, `reports/neural_moe/test_report.json`
- Live route telemetry: `memory/route_telemetry.jsonl`

## STAR Method (Project Narrative)

### Situation
Single-model assistants are usually expensive, harder to debug, and brittle under diverse query types. The goal was to build a production-grade AI Digital Twin that is observable, memory-aware, and cost/latency efficient.

### Task
Deliver an AI assistant that:
1. Routes to specialist experts instead of one universal model
2. Maintains user-isolated memory with compliance controls
3. Supports file-assisted chat (PDF/image + prompt)
4. Produces measurable routing/evaluation outputs for iteration

### Action
Implemented:
- MoE-style routing with confidence threshold and fallback
- Async multi-agent orchestration for lower latency (`Planner`, `Router`, `Retriever`, `Guardrail`, `LLM`, `MemoryWriter`)
- User-scoped long memory and telemetry logs
- File RAG ingestion (`/chat/rag`): parse -> chunk -> embed -> retrieve -> inject
- URL-driven neural MoE training pipeline with exported metrics/artifacts

### Result
- Working multi-expert Digital Twin with per-user memory guardrails
- Neural MoE-style router trained across 13 expert classes
- Evaluation outputs (current run):
  - Validation accuracy: `0.9099`
  - Validation macro-F1: `0.9065`
  - Test accuracy: `0.8900`
  - Test macro-F1: `0.8841`
- Strong foundation for production hardening and continuous retraining

## Key Definitions

- **MoE (Mixture of Experts):** A routing architecture where a gate/router selects the best expert model per query.
- **Router confidence threshold:** Minimum confidence required to trust route prediction; otherwise fallback.
- **RAG:** Retrieve relevant chunks from indexed content and provide them as grounding context to generation.
- **Guardrails:** Safety/privacy controls, including user-bound data access and sanitization.
- **Telemetry:** Structured logs for route label, confidence, fallback, latency, and agent execution trace.

## Production Challenges and Mitigations

### 1) Latency spikes
- **Why it happens:** PDF/image parsing, embedding calls, large model invocations, large contexts.
- **Mitigations:** background ingestion workers, chunk/page limits, caching, async orchestration.

### 2) Cost growth
- **Why it happens:** high fallback usage, repeated embeddings, oversized contexts.
- **Mitigations:** threshold tuning, cost-aware route policies, embedding reuse, tighter retrieval budgets.

### 3) Routing drift
- **Why it happens:** user query distribution changes over time.
- **Mitigations:** telemetry review, misroute labeling, periodic retraining and A/B comparisons.

### 4) Privacy/compliance risks
- **Why it happens:** uploaded docs and long memory may contain sensitive data.
- **Mitigations:** strict `user_id` retrieval filters, redaction/sanitization, export/delete flows, audit logs.

### 5) Reliability under load
- **Why it happens:** local model runtime constraints and provider outages.
- **Mitigations:** retries with backoff, circuit-breaking, health checks, queue-based ingestion.

## MoJo Score (Leverage Metric)

### Definition
`MoJo Score = Output / Human Hours`

This tracks how much valuable system output you can produce per hour of human effort.

### Why it matters
A high MoJo score indicates strong leverage: fewer human hours to build and ship meaningful capability.

### Example (Wingman-style leverage)
Given:
- Your effort: `300` hours
- Traditional baseline: team for `6` months

If baseline is:
- `6 engineers * 6 months * 160 hours/month = 5760 hours`
- MoJo uplift vs baseline = `5760 / 300 = 19.2x`

If baseline is:
- `4 engineers * 6 months * 160 hours/month = 3840 hours`
- MoJo uplift = `3840 / 300 = 12.8x`

Interpretation: the delivered output shows roughly `~13x to ~19x` human-time leverage depending on baseline.
