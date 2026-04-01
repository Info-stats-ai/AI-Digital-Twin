#!/usr/bin/env python3
"""
Phase 2.3: Data enrichment loop helper.

Reads route telemetry and emits candidate enrichment rows for manual labeling.

Usage:
  python scripts/phase2_3_enrich_from_telemetry.py \
    --telemetry memory/route_telemetry.jsonl \
    --out data/enrichment_candidates.jsonl \
    --max 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_candidate(event: Dict) -> Dict:
    query = event.get("query", "")
    features = {
        "contains_code": int(any(t in query.lower() for t in ["traceback", "exception", "error", "code", "python", "npm", "fastapi", "tsx"])),
        "error_log_present": int(any(t in query.lower() for t in ["traceback", "exception", "error:", "failed", "module not found"])),
        "memory_needed": int(any(t in query.lower() for t in ["remember", "earlier", "previous", "my role", "my goals", "what did i say"])),
        "multi_hop": int(any(t in query.lower() for t in ["compare", "tradeoff", "strategy", "roadmap", "analyze", "design"])),
    }
    est_tokens = max(20, min(500, int(len(query.split()) * 1.5)))
    if est_tokens < 80:
        difficulty = "easy"
        latency_budget_ms = 900
    elif est_tokens < 180:
        difficulty = "med"
        latency_budget_ms = 1800
    else:
        difficulty = "hard"
        latency_budget_ms = 3000

    return {
        "query": query,
        "best_expert_label": event.get("router_label", "gpt_fallback"),  # placeholder for human correction
        "best_model_route": event.get("model_route_alias", "gpt-4o-mini"),
        "difficulty": difficulty,
        "expected_answer_quality": 3,  # placeholder to label
        "retrieval_quality_label": "medium",
        "user_feedback_label": "unknown",
        "contains_code": features["contains_code"],
        "error_log_present": features["error_log_present"],
        "memory_needed": features["memory_needed"],
        "multi_hop": features["multi_hop"],
        "estimated_input_tokens": est_tokens,
        "latency_budget_ms": latency_budget_ms,
        "split": "train",
        "source": "telemetry_enrichment",
        "request_id": event.get("request_id"),
        "fallback_triggered": event.get("fallback_triggered"),
        "fallback_reason": event.get("fallback_reason"),
        "latency_ms": event.get("latency_ms"),
        "note": "Review and correct labels before appending to training data",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max", type=int, default=200)
    args = parser.parse_args()

    events = load_jsonl(args.telemetry)
    if not events:
        raise SystemExit("No telemetry events found.")

    # Prioritize likely misroutes: fallbacks and failed requests.
    prioritized = [
        e for e in events
        if (e.get("fallback_triggered") is True)
        or (e.get("success") is False)
        or (e.get("router_confidence") is not None and e.get("router_confidence") < 0.65)
    ]
    if not prioritized:
        prioritized = events

    seen = set()
    candidates = []
    for ev in prioritized:
        q = ev.get("query", "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        candidates.append(make_candidate(ev))
        if len(candidates) >= args.max:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(candidates)} enrichment candidates to {args.out}")
    print("Next step: manually correct labels and append to training dataset.")


if __name__ == "__main__":
    main()

