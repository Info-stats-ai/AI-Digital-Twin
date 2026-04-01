#!/usr/bin/env python3
"""
Phase 2.3: Route telemetry analysis + misroute report.

Usage:
  python scripts/phase2_3_analyze_telemetry.py \
    --telemetry memory/route_telemetry.jsonl \
    --out reports/phase2_3/telemetry_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--misroutes-out", type=Path, default=Path("reports/phase2_3/misroutes.jsonl"))
    args = parser.parse_args()

    events = load_jsonl(args.telemetry)
    if not events:
        raise SystemExit("No telemetry events found.")

    success_events = [e for e in events if e.get("success") is True]
    latencies = [e.get("latency_ms") for e in success_events if isinstance(e.get("latency_ms"), (int, float))]
    fallback_events = [e for e in events if e.get("fallback_triggered") is True]
    low_conf_events = [e for e in events if isinstance(e.get("router_confidence"), (int, float)) and e["router_confidence"] < 0.65]

    provider_counts = Counter([e.get("route_provider", "unknown") for e in success_events])
    label_counts = Counter([e.get("router_label", "unknown") for e in success_events])

    summary = {
        "total_events": len(events),
        "success_events": len(success_events),
        "error_events": len(events) - len(success_events),
        "fallback_events": len(fallback_events),
        "fallback_rate": len(fallback_events) / max(1, len(events)),
        "low_confidence_events": len(low_conf_events),
        "provider_distribution": dict(provider_counts),
        "router_label_distribution": dict(label_counts),
        "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "latency_ms_avg": float(np.mean(latencies)) if latencies else None,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # "Misroute candidates" are events likely needing relabeling/retraining.
    misroutes = []
    for e in events:
        if e.get("success") is False:
            misroutes.append(e)
            continue
        conf = e.get("router_confidence")
        if e.get("fallback_triggered") is True:
            misroutes.append(e)
        elif isinstance(conf, (int, float)) and conf < 0.65:
            misroutes.append(e)

    args.misroutes_out.parent.mkdir(parents=True, exist_ok=True)
    with args.misroutes_out.open("w", encoding="utf-8") as f:
        for row in misroutes:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Summary written: {args.out}")
    print(f"Misroute candidates written: {args.misroutes_out} ({len(misroutes)} rows)")


if __name__ == "__main__":
    main()

