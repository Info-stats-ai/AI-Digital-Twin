#!/usr/bin/env python3
import argparse
import csv
import json
import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -----------------------------
# Config: expert labels + routes
# -----------------------------
EXPERT_LABELS = [
    "memory_factual_expert",
    "technical_expert",
    "gpt_fallback",
]

MODEL_ROUTES = {
    "memory_factual_expert": {
        "easy": "phi-3-mini-4k-instruct",
        "med": "qwen2.5-7b-instruct",
        "hard": "qwen2.5-7b-instruct",
    },
    "technical_expert": {
        "easy": "deepseek-coder-1.3b-instruct",
        "med": "qwen2.5-coder-7b-instruct",
        "hard": "qwen2.5-coder-7b-instruct",
    },
    "gpt_fallback": {
        "easy": "gpt-4o-mini",
        "med": "gpt-4o-mini",
        "hard": "gpt-4.1",
    },
}

# -----------------------------
# Templates
# -----------------------------
MEMORY_TEMPLATES = {
    "easy": [
        "What did I say my role is?",
        "Remind me what my top AI interests are.",
        "What communication style did I ask you to use?",
        "What name should you call me?",
        "What are my key goals for this project?",
    ],
    "med": [
        "Summarize what I told you about my background and current focus.",
        "What did I mention about model evaluation and benchmarking?",
        "Based on my previous messages, what should be my next milestone?",
        "What constraints did I set for latency and cost?",
        "What did I say about using open-source experts and fallback?",
    ],
    "hard": [
        "Use everything I told you about my profile and give a 90-day roadmap with tradeoffs.",
        "Reconcile my earlier goals with my latest constraints and propose a phased execution plan.",
        "Identify conflicts between my memory preferences and current architecture choices.",
        "From past context, infer what I value most and recommend a strategy accordingly.",
        "Synthesize all prior preferences into a production deployment decision memo.",
    ],
}

TECH_TEMPLATES = {
    "easy": [
        "Why am I getting CORS error on localhost?",
        "How do I fix port 8000 already in use?",
        "How to restart backend after .env changes?",
        "What does 401 Unauthorized mean here?",
        "Why does npm say package.json not found?",
    ],
    "med": [
        "How should I structure FastAPI auth middleware for JWT?",
        "How do I design soft-delete for session memory in JSON files?",
        "How to add type guards in TypeScript for API responses?",
        "How to add email verification flow to registration?",
        "How to route between open-source experts and fallback model?",
    ],
    "hard": [
        "Design a low-latency MoE routing architecture with retrieval-aware gating and fallback policy.",
        "How to build evaluation harness for MRR@3, latency, and cost with offline+online signals?",
        "Propose migration from file memory to Postgres+pgvector without downtime.",
        "Design secure multi-tenant auth+memory isolation with audit and compliance constraints.",
        "How to tune routing thresholds using cost-weighted and latency-weighted objectives?",
    ],
}

GPT_FALLBACK_TEMPLATES = {
    "easy": [
        "Give me a polished LinkedIn summary from my profile notes.",
        "Rewrite this project update in a professional tone.",
    ],
    "med": [
        "Draft a product strategy for my digital twin MVP and GTM.",
        "Compare three deployment options with pros/cons and recommendation.",
        "Write a technical design doc for my MoE router service.",
    ],
    "hard": [
        "Create a full architecture proposal with risk matrix, budget estimate, and phased delivery plan.",
        "Analyze system-level tradeoffs among latency, quality, safety, and infra cost for my roadmap.",
        "Produce an executive brief and implementation backlog for a production AI assistant platform.",
    ],
}

NAMES = ["Omkar", "Aarav", "Maya", "Riya", "Arjun", "Neha"]
ROLES = ["Data Scientist", "ML Engineer", "AI Product Engineer"]
TOPICS = ["LLM evaluation", "benchmarking", "RAG", "deployment", "routing"]
ERRORS = ["CORS preflight failed", "401 unauthorized", "500 internal error", "module not found"]


@dataclass
class Row:
    row_id: str
    query: str
    best_expert_label: str
    best_model_route: str
    difficulty: str
    expected_answer_quality: int
    retrieval_quality_label: str
    user_feedback_label: str
    contains_code: int
    error_log_present: int
    memory_needed: int
    multi_hop: int
    estimated_input_tokens: int
    latency_budget_ms: int
    split: str


def pick_expert(rng: random.Random) -> str:
    # Balanced but realistic
    return rng.choices(
        population=EXPERT_LABELS,
        weights=[0.40, 0.38, 0.22],
        k=1,
    )[0]


def pick_difficulty(expert: str, rng: random.Random) -> str:
    if expert == "gpt_fallback":
        return rng.choices(["easy", "med", "hard"], weights=[0.10, 0.40, 0.50], k=1)[0]
    if expert == "technical_expert":
        return rng.choices(["easy", "med", "hard"], weights=[0.30, 0.50, 0.20], k=1)[0]
    return rng.choices(["easy", "med", "hard"], weights=[0.45, 0.40, 0.15], k=1)[0]


def fill_query(template: str, rng: random.Random) -> str:
    q = template
    q = q.replace("{name}", rng.choice(NAMES))
    q = q.replace("{role}", rng.choice(ROLES))
    q = q.replace("{topic}", rng.choice(TOPICS))
    q = q.replace("{error}", rng.choice(ERRORS))
    return q


def make_query(expert: str, difficulty: str, rng: random.Random) -> str:
    if expert == "memory_factual_expert":
        t = rng.choice(MEMORY_TEMPLATES[difficulty])
    elif expert == "technical_expert":
        t = rng.choice(TECH_TEMPLATES[difficulty])
    else:
        t = rng.choice(GPT_FALLBACK_TEMPLATES[difficulty])
    return fill_query(t, rng)


def quality_for(expert: str, difficulty: str, rng: random.Random) -> int:
    base = {
        ("memory_factual_expert", "easy"): 5,
        ("memory_factual_expert", "med"): 4,
        ("memory_factual_expert", "hard"): 3,
        ("technical_expert", "easy"): 5,
        ("technical_expert", "med"): 4,
        ("technical_expert", "hard"): 3,
        ("gpt_fallback", "easy"): 4,
        ("gpt_fallback", "med"): 5,
        ("gpt_fallback", "hard"): 5,
    }[(expert, difficulty)]
    return max(1, min(5, base + rng.choice([0, 0, 0, -1, 1])))


def retrieval_label(expert: str, difficulty: str, rng: random.Random) -> str:
    if expert == "memory_factual_expert":
        return rng.choices(["high", "medium", "low"], weights=[0.60, 0.30, 0.10], k=1)[0]
    if expert == "technical_expert":
        return rng.choices(["high", "medium", "low"], weights=[0.35, 0.45, 0.20], k=1)[0]
    return rng.choices(["high", "medium", "low"], weights=[0.20, 0.45, 0.35], k=1)[0]


def split_for(rng: random.Random) -> str:
    return rng.choices(["train", "val", "test"], weights=[0.80, 0.10, 0.10], k=1)[0]


def make_row(rng: random.Random) -> Row:
    expert = pick_expert(rng)
    difficulty = pick_difficulty(expert, rng)
    query = make_query(expert, difficulty, rng)
    model_route = MODEL_ROUTES[expert][difficulty]

    contains_code = 1 if expert == "technical_expert" and rng.random() < 0.55 else 0
    error_log_present = 1 if expert == "technical_expert" and rng.random() < 0.45 else 0
    memory_needed = 1 if expert == "memory_factual_expert" else (1 if rng.random() < 0.35 else 0)
    multi_hop = 1 if difficulty == "hard" else (1 if rng.random() < 0.15 else 0)

    token_base = {"easy": 40, "med": 110, "hard": 220}[difficulty]
    latency_budget = {"easy": 900, "med": 1800, "hard": 3000}[difficulty]

    return Row(
        row_id=str(uuid.uuid4()),
        query=query,
        best_expert_label=expert,
        best_model_route=model_route,
        difficulty=difficulty,
        expected_answer_quality=quality_for(expert, difficulty, rng),
        retrieval_quality_label=retrieval_label(expert, difficulty, rng),
        user_feedback_label="unknown",  # fill later from live data
        contains_code=contains_code,
        error_log_present=error_log_present,
        memory_needed=memory_needed,
        multi_hop=multi_hop,
        estimated_input_tokens=token_base + rng.randint(-15, 35),
        latency_budget_ms=latency_budget,
        split=split_for(rng),
    )


def to_dict(row: Row) -> Dict[str, object]:
    return {
        "row_id": row.row_id,
        "query": row.query,
        "best_expert_label": row.best_expert_label,
        "best_model_route": row.best_model_route,
        "difficulty": row.difficulty,
        "expected_answer_quality": row.expected_answer_quality,
        "retrieval_quality_label": row.retrieval_quality_label,
        "user_feedback_label": row.user_feedback_label,
        "contains_code": row.contains_code,
        "error_log_present": row.error_log_present,
        "memory_needed": row.memory_needed,
        "multi_hop": row.multi_hop,
        "estimated_input_tokens": row.estimated_input_tokens,
        "latency_budget_ms": row.latency_budget_ms,
        "split": row.split,
    }


def write_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: List[Dict[str, object]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MoE router dataset")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows (e.g., 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--out", type=str, default="router_dataset.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [to_dict(make_row(rng)) for _ in range(args.rows)]

    if args.format == "csv":
        write_csv(rows, args.out)
    else:
        write_jsonl(rows, args.out)

    print(f"Generated {args.rows} rows -> {args.out}")


if __name__ == "__main__":
    main()