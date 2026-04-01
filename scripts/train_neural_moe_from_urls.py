#!/usr/bin/env python3
"""
Import public instruction data via URLs, auto-label expert routes with rules,
train a neural MoE-style router, and report evaluation metrics.

Usage:
  python scripts/train_neural_moe_from_urls.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports" / "neural_moe"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


EXPERTS = [
    "ml_expert",
    "math_reasoning_expert",
    "dl_expert",
    "genai_expert",
    "research_expert",
    "agentic_ai_expert",
    "rag_expert",
    "llm_eval_expert",
    "friendly_conversation_expert",
    "multimodal_expert",
    "memory_factual_expert",
    "technical_expert",
    "gpt_fallback",
]


URLS = {
    "alpaca": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    "gsm8k_train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
}


KEYWORDS = {
    "math_reasoning_expert": ["solve", "equation", "integral", "derivative", "algebra", "calculus", "probability", "theorem"],
    "ml_expert": ["machine learning", "logistic regression", "random forest", "xgboost", "feature engineering", "cross validation"],
    "dl_expert": ["deep learning", "neural network", "cnn", "rnn", "lstm", "transformer", "pytorch", "tensorflow"],
    "genai_expert": ["llm", "prompt", "rlhf", "sft", "inference", "tokenization", "hallucination"],
    "research_expert": ["research paper", "methodology", "ablation", "benchmark", "novelty", "limitations", "arxiv"],
    "agentic_ai_expert": ["agent", "tool calling", "planner", "workflow", "multi-agent", "orchestration"],
    "rag_expert": ["rag", "retrieval", "reranker", "vector db", "embedding", "chunking", "top-k", "mrr"],
    "llm_eval_expert": ["evaluation", "eval", "precision", "recall", "f1", "latency", "cost", "benchmarking"],
    "friendly_conversation_expert": ["hello", "hi", "thanks", "how are you", "good morning", "good evening", "chat"],
    "memory_factual_expert": ["remember", "earlier", "previous", "my role", "my goal", "what did i say"],
    "multimodal_expert": ["image", "figure", "diagram", "screenshot", "pdf", "document", "photo"],
    "technical_expert": ["python", "typescript", "fastapi", "next.js", "debug", "traceback", "error", "api"],
}


def fetch_url(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def parse_alpaca(raw: str) -> List[str]:
    data = json.loads(raw)
    queries = []
    for row in data:
        inst = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        q = f"{inst}\n{inp}".strip()
        if q:
            queries.append(q)
    return queries


def parse_jsonl_questions(raw: str, field: str) -> List[str]:
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        q = str(obj.get(field, "")).strip()
        if q:
            rows.append(q)
    return rows


def infer_label(query: str, source: str) -> str:
    q = query.lower()
    if source == "gsm8k_train":
        return "math_reasoning_expert"
    for label in [
        "memory_factual_expert",
        "math_reasoning_expert",
        "ml_expert",
        "dl_expert",
        "genai_expert",
        "research_expert",
        "agentic_ai_expert",
        "rag_expert",
        "llm_eval_expert",
        "friendly_conversation_expert",
        "multimodal_expert",
        "technical_expert",
    ]:
        if any(k in q for k in KEYWORDS[label]):
            return label
    return "gpt_fallback"


def estimate_features(query: str) -> Dict[str, object]:
    q = query.lower()
    contains_code = int(any(tok in q for tok in ["python", "javascript", "typescript", "code", "fastapi", "next.js", "traceback", "{", "}"]))
    error_log_present = int(any(tok in q for tok in ["traceback", "exception", "error:", "failed", "module not found"]))
    memory_needed = int(any(tok in q for tok in ["remember", "earlier", "previous", "my role", "my goal", "what did i say"]))
    multi_hop = int(any(tok in q for tok in ["compare", "tradeoff", "analyze", "strategy", "design", "evaluate"]))
    has_image = int(any(tok in q for tok in ["image", "figure", "screenshot", "photo", "diagram"]))
    has_pdf = int("pdf" in q or "document" in q)
    est_tokens = max(10, min(600, int(len(re.findall(r"\w+", query)) * 1.3)))
    difficulty = "easy" if est_tokens < 80 else ("med" if est_tokens < 180 else "hard")
    latency_budget_ms = 900 if difficulty == "easy" else (1800 if difficulty == "med" else 3000)
    retrieval_quality_label = "medium"
    return {
        "contains_code": contains_code,
        "error_log_present": error_log_present,
        "memory_needed": memory_needed,
        "multi_hop": multi_hop,
        "has_image": has_image,
        "has_pdf": has_pdf,
        "estimated_input_tokens": est_tokens,
        "difficulty": difficulty,
        "latency_budget_ms": latency_budget_ms,
        "retrieval_quality_label": retrieval_quality_label,
    }


def balance_per_class(df: pd.DataFrame, target_per_class: int = 1200) -> pd.DataFrame:
    parts = []
    for label in EXPERTS:
        sub = df[df["best_expert_label"] == label]
        if sub.empty:
            continue
        if len(sub) >= target_per_class:
            parts.append(sub.sample(target_per_class, random_state=SEED))
        else:
            sampled = sub.sample(target_per_class, random_state=SEED, replace=True)
            parts.append(sampled)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out


class RouterDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class NeuralRouter(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    queries: List[Dict[str, str]] = []
    for name, url in URLS.items():
        raw = fetch_url(url)
        if name == "alpaca":
            rows = parse_alpaca(raw)[:30000]
        elif name == "gsm8k_train":
            rows = parse_jsonl_questions(raw, "question")[:7000]
        else:
            rows = []
        for q in rows:
            queries.append({"query": q, "source": name})

    df = pd.DataFrame(queries).drop_duplicates(subset=["query"]).reset_index(drop=True)
    df["best_expert_label"] = [infer_label(q, s) for q, s in zip(df["query"], df["source"])]
    features_df = pd.DataFrame([estimate_features(q) for q in df["query"]])
    df = pd.concat([df, features_df], axis=1)

    balanced = balance_per_class(df, target_per_class=900)
    balanced.to_json(DATA_DIR / "neural_moe_imported_dataset.jsonl", orient="records", lines=True, force_ascii=False)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    encoder = SentenceTransformer(model_name)
    text_emb = encoder.encode(balanced["query"].tolist(), batch_size=128, show_progress_bar=True, normalize_embeddings=True)

    difficulty_map = {"easy": 0, "med": 1, "hard": 2}
    retrieval_map = {"low": 0, "medium": 1, "high": 2}
    numeric = np.column_stack(
        [
            balanced["contains_code"].to_numpy(dtype=float),
            balanced["error_log_present"].to_numpy(dtype=float),
            balanced["memory_needed"].to_numpy(dtype=float),
            balanced["multi_hop"].to_numpy(dtype=float),
            balanced["has_image"].to_numpy(dtype=float),
            balanced["has_pdf"].to_numpy(dtype=float),
            balanced["estimated_input_tokens"].to_numpy(dtype=float) / 600.0,
            balanced["latency_budget_ms"].to_numpy(dtype=float) / 3000.0,
            balanced["difficulty"].map(difficulty_map).to_numpy(dtype=float) / 2.0,
            balanced["retrieval_quality_label"].map(retrieval_map).to_numpy(dtype=float) / 2.0,
        ]
    )
    x = np.concatenate([text_emb, numeric], axis=1)

    le = LabelEncoder()
    y = le.fit_transform(balanced["best_expert_label"].tolist())

    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

    train_loader = DataLoader(RouterDataset(x_train, y_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(RouterDataset(x_val, y_val), batch_size=512, shuffle=False)
    test_loader = DataLoader(RouterDataset(x_test, y_test), batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralRouter(input_dim=x.shape[1], num_classes=len(le.classes_)).to(device)
    class_counts = np.bincount(y_train, minlength=len(le.classes_))
    class_weights = torch.tensor((class_counts.sum() / np.maximum(class_counts, 1)).astype(np.float32), device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None
    patience = 3
    bad_epochs = 0

    def run_eval(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                ys.append(yb.numpy())
                ps.append(pred)
        return np.concatenate(ys), np.concatenate(ps)

    for epoch in range(18):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        yv, pv = run_eval(val_loader)
        val_f1 = f1_score(yv, pv, average="macro")
        print(f"epoch={epoch+1} loss={epoch_loss/max(1, len(train_loader)):.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    yv, pv = run_eval(val_loader)
    yt, pt = run_eval(test_loader)
    metrics = {
        "val_accuracy": float(accuracy_score(yv, pv)),
        "val_macro_f1": float(f1_score(yv, pv, average="macro")),
        "test_accuracy": float(accuracy_score(yt, pt)),
        "test_macro_f1": float(f1_score(yt, pt, average="macro")),
        "num_rows": int(len(balanced)),
        "num_classes": int(len(le.classes_)),
        "classes": le.classes_.tolist(),
        "encoder_model": model_name,
    }
    print(json.dumps(metrics, indent=2))

    report_val = classification_report(yv, pv, target_names=le.classes_, output_dict=True)
    report_test = classification_report(yt, pt, target_names=le.classes_, output_dict=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_classes": le.classes_.tolist(),
            "input_dim": int(x.shape[1]),
            "encoder_model": model_name,
            "feature_order": [
                "contains_code",
                "error_log_present",
                "memory_needed",
                "multi_hop",
                "has_image",
                "has_pdf",
                "estimated_input_tokens_norm",
                "latency_budget_ms_norm",
                "difficulty_norm",
                "retrieval_quality_norm",
            ],
        },
        ARTIFACTS_DIR / "router_neural_moe.pt",
    )
    (ARTIFACTS_DIR / "router_neural_moe_meta.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS_DIR / "val_report.json").write_text(json.dumps(report_val, indent=2), encoding="utf-8")
    (REPORTS_DIR / "test_report.json").write_text(json.dumps(report_test, indent=2), encoding="utf-8")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

