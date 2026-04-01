#!/usr/bin/env python3
"""
Phase 2.3: Offline router evaluation + threshold sweep.

Usage:
  python scripts/phase2_3_eval_router.py \
    --dataset router_dataset_10k.jsonl \
    --model artifacts/router_tfidf_lr.pkl \
    --out reports/phase2_3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


FEATURE_COLUMNS = [
    "query",
    "contains_code",
    "error_log_present",
    "memory_needed",
    "multi_hop",
    "estimated_input_tokens",
    "latency_budget_ms",
    "difficulty",
    "retrieval_quality_label",
]


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_cost_proxy(pred_label: str) -> float:
    # Rough relative unit cost for threshold tuning comparison
    if pred_label == "gpt_fallback":
        return 1.0
    if pred_label == "technical_expert":
        return 0.35
    return 0.2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/phase2_3"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(load_jsonl(args.dataset))
    model = joblib.load(args.model)

    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if val_df.empty or test_df.empty:
        raise SystemExit("Dataset missing val/test splits.")

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["best_expert_label"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["best_expert_label"]

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    summary = {
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
    }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "classification_report_val.json").write_text(
        json.dumps(classification_report(y_val, val_pred, output_dict=True), indent=2),
        encoding="utf-8",
    )
    (args.out / "classification_report_test.json").write_text(
        json.dumps(classification_report(y_test, test_pred, output_dict=True), indent=2),
        encoding="utf-8",
    )

    cm_val = confusion_matrix(y_val, val_pred, labels=sorted(y_val.unique()))
    cm_test = confusion_matrix(y_test, test_pred, labels=sorted(y_test.unique()))
    pd.DataFrame(cm_val).to_csv(args.out / "confusion_val.csv", index=False)
    pd.DataFrame(cm_test).to_csv(args.out / "confusion_test.csv", index=False)

    # Threshold sweep using predict_proba on validation set.
    classes = model.named_steps["clf"].classes_
    probs = model.predict_proba(X_val)
    true_labels = y_val.to_numpy()

    rows = []
    for th in np.arange(0.45, 0.91, 0.05):
        pred_labels = []
        cost_values = []
        fallback_count = 0
        confidences = probs.max(axis=1)
        raw_idx = probs.argmax(axis=1)
        raw_pred = classes[raw_idx]
        for i in range(len(raw_pred)):
            if confidences[i] < th:
                label = "gpt_fallback"
                fallback_count += 1
            else:
                label = str(raw_pred[i])
            pred_labels.append(label)
            cost_values.append(compute_cost_proxy(label))

        acc = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average="macro")
        fallback_rate = fallback_count / len(pred_labels)
        avg_cost = float(np.mean(cost_values))
        # Weighted objective: quality - cost penalty - fallback penalty
        objective = macro_f1 - 0.15 * avg_cost - 0.1 * fallback_rate
        rows.append(
            {
                "threshold": round(float(th), 2),
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "fallback_rate": float(fallback_rate),
                "avg_cost_proxy": avg_cost,
                "objective": float(objective),
            }
        )

    sweep_df = pd.DataFrame(rows).sort_values("objective", ascending=False)
    sweep_df.to_csv(args.out / "threshold_sweep.csv", index=False)
    best = sweep_df.iloc[0].to_dict()
    (args.out / "best_threshold.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("Saved reports to:", args.out)
    print("Summary:", summary)
    print("Best threshold:", best)


if __name__ == "__main__":
    main()

