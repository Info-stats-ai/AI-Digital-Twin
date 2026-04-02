from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, EmailStr, model_validator
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, TypedDict, Callable, Awaitable
import json
import uuid
from pathlib import Path
import re
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
import smtplib
from email.message import EmailMessage
import joblib
import pandas as pd
import numpy as np
import requests
import time
import asyncio
import base64
import mimetypes
import hashlib
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# Load environment variables
load_dotenv(override=True)

app = FastAPI()

# Configure CORS
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
origin_regex = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?$",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI()

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "24"))
EMAIL_VERIFY_EXPIRE_HOURS = int(os.getenv("EMAIL_VERIFY_EXPIRE_HOURS", "24"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:3000")
API_PUBLIC_BASE_URL = os.getenv("API_PUBLIC_BASE_URL", "http://localhost:8000")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "no-reply@digitaltwin.local")
ROUTER_MODEL_PATH = os.getenv("ROUTER_MODEL_PATH", "../artifacts/router_tfidf_lr.pkl")
ROUTER_META_PATH = os.getenv("ROUTER_META_PATH", "../artifacts/router_meta.json")
ROUTER_BACKEND = os.getenv("ROUTER_BACKEND", "neural").lower()
ROUTER_NEURAL_PATH = os.getenv("ROUTER_NEURAL_PATH", "../artifacts/router_neural_moe.pt")
ROUTER_NEURAL_META_PATH = os.getenv("ROUTER_NEURAL_META_PATH", "../artifacts/router_neural_moe_meta.json")
ROUTER_CONF_THRESHOLD = float(os.getenv("ROUTER_CONF_THRESHOLD", "0.62"))
DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", DEFAULT_OPENAI_MODEL)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "30"))
MULTIMODAL_OPENAI_MODEL = os.getenv("MULTIMODAL_OPENAI_MODEL", "gpt-4.1-mini")
MEMORY_RETRIEVAL_TOP_K = int(os.getenv("MEMORY_RETRIEVAL_TOP_K", "3"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_CHUNK_WORDS = int(os.getenv("RAG_CHUNK_WORDS", "180"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "40"))
INGESTION_QUEUE_ENABLED = os.getenv("INGESTION_QUEUE_ENABLED", "true").lower() in {"1", "true", "yes"}

# Memory directory
MEMORY_DIR = Path("../memory")
MEMORY_DIR.mkdir(exist_ok=True)
USERS_FILE = MEMORY_DIR / "users.json"
AUDIT_FILE = MEMORY_DIR / "audit.log"
ROUTE_TELEMETRY_FILE = MEMORY_DIR / "route_telemetry.jsonl"

# In-memory progress + ingestion queue state (single-process dev/prototype runtime).
PROGRESS_STATE: Dict[str, Dict[str, Any]] = {}
INGESTION_JOBS: Dict[str, Dict[str, Any]] = {}
INGESTION_QUEUE: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
AGENT_METRICS: Dict[str, Dict[str, float]] = {}


# Load personality details
def load_personality():
    with open("me.txt", "r", encoding="utf-8") as f:
        return f.read().strip()


PERSONALITY = load_personality()

EXPERT_TO_PROVIDER_ROUTE = {
    "memory_factual_expert": {"provider": "ollama", "model": "phi3:mini"},
    "technical_expert": {"provider": "ollama", "model": "qwen2.5-coder:7b-instruct"},
    "ml_expert": {"provider": "ollama", "model": os.getenv("ML_EXPERT_MODEL", "llama3.1:8b")},
    "math_reasoning_expert": {"provider": "ollama", "model": os.getenv("MATH_EXPERT_MODEL", "deepseek-r1:8b")},
    "dl_expert": {"provider": "ollama", "model": os.getenv("DL_EXPERT_MODEL", "qwen2.5-coder:7b-instruct")},
    "genai_expert": {"provider": "ollama", "model": os.getenv("GENAI_EXPERT_MODEL", "qwen2.5:7b")},
    "research_expert": {"provider": "ollama", "model": os.getenv("RESEARCH_EXPERT_MODEL", "mistral:7b")},
    "agentic_ai_expert": {"provider": "ollama", "model": os.getenv("AGENTIC_EXPERT_MODEL", "llama3.1:8b")},
    "rag_expert": {"provider": "ollama", "model": os.getenv("RAG_EXPERT_MODEL", "qwen2.5:7b")},
    "llm_eval_expert": {"provider": "ollama", "model": os.getenv("LLM_EVAL_EXPERT_MODEL", "llama3.1:8b")},
    "friendly_conversation_expert": {"provider": "ollama", "model": os.getenv("FRIENDLY_EXPERT_MODEL", "phi3:mini")},
    "multimodal_expert": {"provider": "openai", "model": MULTIMODAL_OPENAI_MODEL},
    "gpt_fallback": {"provider": "openai", "model": OPENAI_FALLBACK_MODEL},
}

EXPERT_KEYWORDS = {
    "math_reasoning_expert": [
        "solve", "equation", "integral", "derivative", "matrix", "algebra", "probability",
        "statistics", "bayes", "calculus", "linear algebra", "optimization", "gradient", "theorem",
    ],
    "ml_expert": [
        "machine learning", "sklearn", "xgboost", "random forest", "logistic regression",
        "feature engineering", "cross validation", "roc", "classification report",
    ],
    "dl_expert": [
        "deep learning", "neural network", "cnn", "rnn", "lstm", "transformer",
        "backprop", "pytorch", "tensorflow", "fine-tune model weights",
    ],
    "genai_expert": [
        "prompt engineering", "inference", "llm serving", "sft", "rlhf", "distillation",
        "tokenization", "context window", "hallucination",
    ],
    "research_expert": [
        "research paper", "paper summary", "methodology", "ablation", "sota", "benchmark",
        "novelty", "limitations", "arxiv",
    ],
    "agentic_ai_expert": [
        "agent", "tool calling", "workflow", "planner", "react pattern", "multi agent",
        "orchestration", "autonomous",
    ],
    "rag_expert": [
        "rag", "retrieval", "reranker", "embedding", "vector db", "chunking", "mrr",
        "top-k", "hit@k", "grounded generation",
    ],
    "llm_eval_expert": [
        "evaluation", "eval", "judge model", "latency", "cost", "accuracy", "macro f1",
        "misroute", "threshold sweep", "benchmarking",
    ],
    "friendly_conversation_expert": [
        "how are you", "hello", "hi", "good morning", "good evening", "thanks", "thank you",
        "can we chat", "small talk", "tell me about yourself",
    ],
    "memory_factual_expert": [
        "remember", "earlier", "previous", "my role", "my goals", "what did i say",
    ],
}

class NeuralRouterNet(nn.Module):
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


router_model = None
router_meta: Optional[Dict[str, Any]] = None
router_neural: Optional[Dict[str, Any]] = None

try:
    router_model = joblib.load(ROUTER_MODEL_PATH)
    with open(ROUTER_META_PATH, "r", encoding="utf-8") as f:
        router_meta = json.load(f)
    print(f"[router-lr] loaded model from {ROUTER_MODEL_PATH}")
except Exception as e:
    print(f"[router-lr] not loaded: {e}")

try:
    ckpt = torch.load(ROUTER_NEURAL_PATH, map_location="cpu")
    label_classes = [str(c) for c in ckpt["label_classes"]]
    input_dim = int(ckpt["input_dim"])
    encoder_model_name = str(ckpt.get("encoder_model", "sentence-transformers/all-MiniLM-L6-v2"))
    feature_order = ckpt.get(
        "feature_order",
        [
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
    )

    net = NeuralRouterNet(input_dim=input_dim, num_classes=len(label_classes))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    encoder = SentenceTransformer(encoder_model_name)

    neural_meta: Dict[str, Any] = {}
    if Path(ROUTER_NEURAL_META_PATH).exists():
        with open(ROUTER_NEURAL_META_PATH, "r", encoding="utf-8") as f:
            neural_meta = json.load(f)

    router_neural = {
        "model": net,
        "encoder": encoder,
        "classes": label_classes,
        "feature_order": feature_order,
        "meta": neural_meta,
    }
    print(f"[router-neural] loaded model from {ROUTER_NEURAL_PATH}")
except Exception as e:
    print(f"[router-neural] not loaded: {e}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress_init(request_id: str, user_id: str) -> None:
    PROGRESS_STATE[request_id] = {
        "user_id": user_id,
        "events": [],
        "done": False,
        "updated_at": now_iso(),
    }


def progress_emit(request_id: str, agent: str, status: str, detail: Optional[str] = None) -> None:
    state = PROGRESS_STATE.get(request_id)
    if not state:
        return
    state["events"].append(
        {
            "agent": agent,
            "status": status,
            "detail": detail,
            "timestamp": now_iso(),
        }
    )
    state["updated_at"] = now_iso()


def progress_done(request_id: str) -> None:
    state = PROGRESS_STATE.get(request_id)
    if not state:
        return
    state["done"] = True
    state["updated_at"] = now_iso()


def load_users() -> List[Dict[str, Any]]:
    if not USERS_FILE.exists():
        return []
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users: List[Dict[str, Any]]) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def normalize_email(email: Optional[str]) -> Optional[str]:
    return email.strip().lower() if email else None


def normalize_phone(phone: Optional[str]) -> Optional[str]:
    return phone.strip() if phone else None


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(user_id: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"sub": user_id, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def create_email_verification_token(user_id: str, email: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=EMAIL_VERIFY_EXPIRE_HOURS)
    payload = {"sub": user_id, "email": email, "purpose": "email_verify", "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def send_verification_email(to_email: str, verification_url: str) -> None:
    if not SMTP_HOST:
        # Dev fallback when SMTP isn't configured.
        print(f"[DEV] Email verification link for {to_email}: {verification_url}")
        return

    msg = EmailMessage()
    msg["Subject"] = "Verify your AI Digital Twin account"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content(
        "Welcome to AI Digital Twin.\n\n"
        "Please verify your email by opening this link:\n"
        f"{verification_url}\n\n"
        "If you didn't sign up, you can ignore this email."
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        if SMTP_USER and SMTP_PASS:
            server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    for user in load_users():
        if user["user_id"] == user_id:
            return user
    return None


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


def _user_memory_dir(user_id: str) -> Path:
    user_dir = MEMORY_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def session_file_path(user_id: str, session_id: str) -> Path:
    return _user_memory_dir(user_id) / f"{session_id}.json"


def long_memory_file_path(user_id: str) -> Path:
    return _user_memory_dir(user_id) / "long_memory.json"


def user_docs_dir(user_id: str) -> Path:
    path = _user_memory_dir(user_id) / "docs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_vector_index_path(user_id: str) -> Path:
    return _user_memory_dir(user_id) / "rag_vectors.jsonl"


def load_user_vectors(user_id: str) -> List[Dict[str, Any]]:
    path = user_vector_index_path(user_id)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_user_vectors(user_id: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path = user_vector_index_path(user_id)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name or "upload")
    return cleaned[:120]


def build_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    raw = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"


def extract_text_from_pdf(path: Path) -> str:
    if fitz is None:
        raise HTTPException(status_code=500, detail="PDF parsing unavailable. Install pymupdf.")
    doc = fitz.open(path)
    parts: List[str] = []
    try:
        for page in doc:
            text = page.get_text("text")
            if text:
                parts.append(text)
    finally:
        doc.close()
    return "\n".join(parts).strip()


def image_to_text_with_vision(path: Path, user_prompt: str) -> str:
    data_url = build_data_url(path)
    prompt = (
        "Extract all relevant text and key factual details from this image for question answering. "
        "Be concise but complete.\n\n"
        f"User prompt context: {user_prompt}"
    )
    resp = client.chat.completions.create(
        model=MULTIMODAL_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You convert images into concise, factual text context for retrieval."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def chunk_text(text: str, chunk_words: int = RAG_CHUNK_WORDS, overlap_words: int = RAG_CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    i = 0
    step = max(1, chunk_words - overlap_words)
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_words]).strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def load_long_memory(user_id: str) -> List[Dict[str, Any]]:
    path = long_memory_file_path(user_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_long_memory(user_id: str, records: List[Dict[str, Any]]) -> None:
    path = long_memory_file_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def memory_similarity(query: str, content: str) -> float:
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(content))
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    return overlap / max(1, len(q_tokens))


def sanitize_text(text: str) -> str:
    # Basic PII redaction before passing shared memory context to models.
    redacted = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
    redacted = re.sub(r"\+?\d[\d\-\s]{7,}\d", "[redacted-phone]", redacted)
    return redacted.strip()


def extract_long_memory_facts(query: str, response: str) -> List[Dict[str, Any]]:
    q = query.lower()
    candidates = []
    patterns = [
        "my name is",
        "i am",
        "i'm",
        "my role is",
        "my goal is",
        "i work on",
        "i prefer",
        "remember that",
    ]
    if any(p in q for p in patterns):
        candidates.append({
            "fact_text": sanitize_text(query),
            "importance": 0.9,
            "fact_type": "user_profile_or_preference",
        })
    if "goal" in q or "deadline" in q or "interview" in q:
        candidates.append({
            "fact_text": sanitize_text(query),
            "importance": 0.85,
            "fact_type": "user_goal",
        })
    return candidates


def append_audit_log(action: str, user_id: str, session_id: Optional[str] = None, detail: Optional[str] = None) -> None:
    record = {
        "timestamp": now_iso(),
        "action": action,
        "user_id": user_id,
        "session_id": session_id,
        "detail": detail,
    }
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_route_telemetry(record: Dict[str, Any]) -> None:
    """Append one route execution event as JSONL for phase 2.3 analysis."""
    with open(ROUTE_TELEMETRY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_router_features(query: str, retrieval_quality_label: str = "medium", has_images: bool = False) -> Dict[str, Any]:
    q_lower = query.lower()
    contains_code = int(any(tok in q_lower for tok in ["traceback", "exception", "error", "code", "python", "npm", "fastapi", "tsx", "{", "}"]))
    error_log_present = int(any(tok in q_lower for tok in ["traceback", "exception", "error:", "failed", "module not found"]))
    memory_needed = int(any(tok in q_lower for tok in ["remember", "earlier", "previous", "my role", "my goals", "what did i say"]))
    multi_hop = int(any(tok in q_lower for tok in ["compare", "tradeoff", "strategy", "roadmap", "analyze", "design"]))
    has_pdf = int("pdf" in q_lower or "document" in q_lower or "paper" in q_lower)
    has_image = int(has_images or any(tok in q_lower for tok in ["image", "screenshot", "diagram", "figure", "photo"]))

    estimated_tokens = max(20, min(500, int(len(query.split()) * 1.5)))
    if estimated_tokens < 80:
        difficulty = "easy"
        latency_budget_ms = 900
    elif estimated_tokens < 180:
        difficulty = "med"
        latency_budget_ms = 1800
    else:
        difficulty = "hard"
        latency_budget_ms = 3000

    return {
        "contains_code": contains_code,
        "error_log_present": error_log_present,
        "memory_needed": memory_needed,
        "multi_hop": multi_hop,
        "has_image": has_image,
        "has_pdf": has_pdf,
        "estimated_input_tokens": estimated_tokens,
        "difficulty": difficulty,
        "retrieval_quality_label": retrieval_quality_label,
        "latency_budget_ms": latency_budget_ms,
    }


def infer_router_neural(query: str, retrieval_quality_label: str = "medium", has_images: bool = False) -> Optional[Dict[str, Any]]:
    if router_neural is None:
        return None
    features = build_router_features(query, retrieval_quality_label, has_images)
    difficulty_map = {"easy": 0.0, "med": 0.5, "hard": 1.0}
    retrieval_map = {"low": 0.0, "medium": 0.5, "high": 1.0}

    text_emb = router_neural["encoder"].encode([query], normalize_embeddings=True)[0]
    numeric = np.array(
        [
            float(features["contains_code"]),
            float(features["error_log_present"]),
            float(features["memory_needed"]),
            float(features["multi_hop"]),
            float(features["has_image"]),
            float(features["has_pdf"]),
            float(features["estimated_input_tokens"]) / 600.0,
            float(features["latency_budget_ms"]) / 3000.0,
            difficulty_map.get(str(features["difficulty"]), 0.5),
            retrieval_map.get(str(features["retrieval_quality_label"]), 0.5),
        ],
        dtype=np.float32,
    )
    x = np.concatenate([text_emb.astype(np.float32), numeric], axis=0)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = router_neural["model"](x_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx = int(np.argmax(probs))
    raw_expert = str(router_neural["classes"][idx])
    confidence = float(probs[idx])

    return {
        "raw_expert_label": raw_expert,
        "confidence": confidence,
        "features": features,
    }


def select_heuristic_expert(query: str, has_images: bool = False) -> Optional[str]:
    q = query.lower()
    if has_images:
        return "multimodal_expert"
    # Ordered to prioritize task-specific experts before broad technical route.
    ordered_experts = [
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
    ]
    for expert in ordered_experts:
        if any(tok in q for tok in EXPERT_KEYWORDS[expert]):
            return expert
    return None


def infer_router(query: str, retrieval_quality_label: str = "medium", has_images: bool = False) -> Dict[str, Any]:
    """Predict route using configured backend (neural/lr); fallback if unavailable/low confidence."""
    heuristic_expert = select_heuristic_expert(query, has_images=has_images)
    neural_candidate = infer_router_neural(query, retrieval_quality_label, has_images) if ROUTER_BACKEND == "neural" else None
    lr_available = router_model is not None

    if neural_candidate is None and not lr_available:
        selected = heuristic_expert or "gpt_fallback"
        route = EXPERT_TO_PROVIDER_ROUTE.get(selected, EXPERT_TO_PROVIDER_ROUTE["gpt_fallback"])
        return {
            "expert_label": selected,
            "raw_expert_label": "gpt_fallback",
            "confidence": 0.75 if heuristic_expert else 0.0,
            "fallback_triggered": selected == "gpt_fallback",
            "reason": "heuristic_no_router" if heuristic_expert else "router_not_loaded_any_backend",
            "provider": route["provider"],
            "model_route_alias": route["model"],
        }

    if neural_candidate is not None:
        raw_expert = str(neural_candidate["raw_expert_label"])
        confidence = float(neural_candidate["confidence"])
        features = dict(neural_candidate["features"])
        base_reason = "router_neural_prediction"
    else:
        features = build_router_features(query, retrieval_quality_label, has_images)
        row = pd.DataFrame([{
            "query": query,
            "contains_code": features["contains_code"],
            "error_log_present": features["error_log_present"],
            "memory_needed": features["memory_needed"],
            "multi_hop": features["multi_hop"],
            "estimated_input_tokens": features["estimated_input_tokens"],
            "latency_budget_ms": features["latency_budget_ms"],
            "difficulty": features["difficulty"],
            "retrieval_quality_label": features["retrieval_quality_label"],
        }])
        probs = router_model.predict_proba(row)[0]
        classes = router_model.named_steps["clf"].classes_
        idx = int(np.argmax(probs))
        raw_expert = str(classes[idx])
        confidence = float(probs[idx])
        base_reason = "router_lr_prediction"

    fallback_triggered = confidence < ROUTER_CONF_THRESHOLD
    final_expert = "gpt_fallback" if fallback_triggered else raw_expert
    route_reason = base_reason
    if fallback_triggered:
        route_reason = f"{base_reason}_low_confidence"

    # Heuristic overrides let us support newly added experts before retraining the LR router.
    if heuristic_expert:
        final_expert = heuristic_expert
        fallback_triggered = final_expert == "gpt_fallback"
        route_reason = f"{base_reason}_heuristic_override"

    route = EXPERT_TO_PROVIDER_ROUTE.get(final_expert, EXPERT_TO_PROVIDER_ROUTE["gpt_fallback"])
    return {
        "expert_label": final_expert,
        "raw_expert_label": raw_expert,
        "confidence": confidence,
        "fallback_triggered": fallback_triggered,
        "reason": route_reason,
        "provider": route["provider"],
        "model_route_alias": route["model"],
        "features": features,
    }


def normalize_messages_for_ollama(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            normalized.append({"role": role, "content": " ".join(text_parts).strip()})
            continue
        normalized.append({"role": role, "content": str(content)})
    return normalized


def ollama_chat(model: str, messages: List[Dict[str, Any]]) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {"model": model, "messages": normalize_messages_for_ollama(messages), "stream": False}
    resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def generate_with_route(messages: List[Dict[str, Any]], router_decision: Dict[str, Any]) -> Dict[str, Any]:
    provider = router_decision["provider"]
    model = router_decision["model_route_alias"]
    start = time.time()
    fallback_triggered = False
    fallback_reason = None

    try:
        if provider == "ollama":
            text = ollama_chat(model=model, messages=messages)
        else:
            resp = client.chat.completions.create(model=model, messages=messages)
            text = resp.choices[0].message.content
        return {
            "text": text,
            "provider_used": provider,
            "model_used": model,
            "latency_ms": int((time.time() - start) * 1000),
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
        }
    except Exception as exc:
        fallback_triggered = True
        fallback_reason = f"{provider}_error: {exc}"
        resp = client.chat.completions.create(model=OPENAI_FALLBACK_MODEL, messages=messages)
        text = resp.choices[0].message.content
        return {
            "text": text,
            "provider_used": "openai",
            "model_used": OPENAI_FALLBACK_MODEL,
            "latency_ms": int((time.time() - start) * 1000),
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
        }


def start_agent_event(trace: List[Dict[str, Any]], agent: str, detail: Optional[str] = None) -> int:
    trace.append({
        "agent": agent,
        "status": "running",
        "started_at": now_iso(),
        "ended_at": None,
        "detail": detail,
    })
    return len(trace) - 1


def end_agent_event(trace: List[Dict[str, Any]], idx: int, status: str = "completed", detail: Optional[str] = None) -> None:
    trace[idx]["status"] = status
    trace[idx]["ended_at"] = now_iso()
    if detail:
        trace[idx]["detail"] = detail


async def planner_agent(query: str, has_images: bool) -> Dict[str, Any]:
    hint = select_heuristic_expert(query, has_images=has_images)
    return {
        "expert_hint": hint,
        "needs_memory": True,
        "has_images": has_images,
    }


async def policy_agent(query: str, user_id: str) -> Dict[str, Any]:
    # Simple policy gate placeholder for permissions/safety/rate controls.
    if len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 4000:
        raise HTTPException(status_code=400, detail="Query exceeds max length")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing authenticated user")
    return {"policy_ok": True}


async def router_agent(query: str, retrieval_quality_label: str, has_images: bool) -> Dict[str, Any]:
    return await asyncio.to_thread(
        infer_router,
        query,
        retrieval_quality_label,
        has_images,
    )


async def memory_retriever_agent(user_id: str, session_env: "SessionEnvelope", query: str) -> Dict[str, Any]:
    session_tail = session_env.messages[-6:] if len(session_env.messages) > 6 else session_env.messages
    long_mem = load_long_memory(user_id)
    scored = []
    for item in long_mem:
        content = str(item.get("fact_text", ""))
        scored.append((memory_similarity(query, content), item))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_long = [row for score, row in scored if score > 0][:MEMORY_RETRIEVAL_TOP_K]
    return {
        "session_tail": session_tail,
        "long_memory_hits": top_long,
    }


async def guardrail_agent(memory_result: Dict[str, Any]) -> Dict[str, Any]:
    sanitized_hits = []
    for hit in memory_result.get("long_memory_hits", []):
        copied = dict(hit)
        copied["fact_text"] = sanitize_text(str(hit.get("fact_text", "")))
        sanitized_hits.append(copied)
    return {
        "session_tail": memory_result.get("session_tail", []),
        "long_memory_hits": sanitized_hits,
    }


async def memory_writer_agent(user_id: str, session_id: str, query: str, response: str) -> Dict[str, Any]:
    candidates = extract_long_memory_facts(query, response)
    if not candidates:
        return {"stored": 0}

    existing = load_long_memory(user_id)
    existing_texts = {str(item.get("fact_text", "")).strip().lower() for item in existing}
    stored = 0
    for cand in candidates:
        fact_text = str(cand["fact_text"]).strip()
        if not fact_text or fact_text.lower() in existing_texts:
            continue
        existing.append({
            "memory_id": str(uuid.uuid4()),
            "user_id": user_id,
            "session_id": session_id,
            "fact_text": fact_text,
            "fact_type": cand["fact_type"],
            "importance": cand["importance"],
            "created_at": now_iso(),
            "updated_at": now_iso(),
        })
        existing_texts.add(fact_text.lower())
        stored += 1

    if stored:
        save_long_memory(user_id, existing)
    return {"stored": stored}


async def ingest_raw_files_agent(
    user_id: str,
    session_id: str,
    user_prompt: str,
    raw_files: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    docs_dir = user_docs_dir(user_id)
    for payload in raw_files:
        raw = payload.get("data", b"")
        if not raw:
            continue
        original_name = safe_filename(str(payload.get("filename", "upload")))
        digest = hashlib.sha1(raw).hexdigest()[:10]
        doc_id = f"{session_id}-{digest}"
        saved_path = docs_dir / f"{doc_id}-{original_name}"
        saved_path.write_bytes(raw)

        lower_name = original_name.lower()
        mime = str(payload.get("content_type") or "")
        extracted_text = ""
        source_type = "text"
        if lower_name.endswith(".pdf") or "pdf" in mime:
            source_type = "pdf"
            extracted_text = await asyncio.to_thread(extract_text_from_pdf, saved_path)
        elif any(lower_name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) or mime.startswith("image/"):
            source_type = "image"
            extracted_text = await asyncio.to_thread(image_to_text_with_vision, saved_path, user_prompt)
        else:
            extracted_text = raw.decode("utf-8", errors="ignore")

        if extracted_text.strip():
            docs.append({
                "doc_id": doc_id,
                "file_name": original_name,
                "source_type": source_type,
                "text": extracted_text.strip(),
            })
    return docs


async def file_ingestion_agent(user_id: str, session_id: str, user_prompt: str, files: List[UploadFile]) -> List[Dict[str, Any]]:
    raw_files: List[Dict[str, Any]] = []
    for upload in files:
        raw = await upload.read()
        raw_files.append(
            {
                "filename": upload.filename or "upload",
                "content_type": upload.content_type or "",
                "data": raw,
            }
        )
    return await ingest_raw_files_agent(user_id, session_id, user_prompt, raw_files)


async def chunking_agent(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for doc in documents:
        for idx, chunk in enumerate(chunk_text(str(doc["text"]))):
            chunks.append({
                "doc_id": doc["doc_id"],
                "file_name": doc["file_name"],
                "source_type": doc["source_type"],
                "chunk_id": f"{doc['doc_id']}-c{idx}",
                "text": chunk,
            })
    return chunks


async def embedding_agent(user_id: str, session_id: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [c["text"] for c in chunks]
    vectors = await asyncio.to_thread(embed_texts, texts)
    now = now_iso()
    rows = []
    for c, emb in zip(chunks, vectors):
        rows.append({
            "user_id": user_id,
            "session_id": session_id,
            "doc_id": c["doc_id"],
            "file_name": c["file_name"],
            "source_type": c["source_type"],
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "embedding": emb,
            "created_at": now,
        })
    await asyncio.to_thread(append_user_vectors, user_id, rows)
    return {"stored_chunks": len(rows)}


async def rag_retriever_agent(user_id: str, query: str, top_k: int = RAG_TOP_K) -> Dict[str, Any]:
    rows = await asyncio.to_thread(load_user_vectors, user_id)
    if not rows:
        return {"chunks": [], "doc_ids": []}
    query_embedding = (await asyncio.to_thread(embed_texts, [query]))[0]
    scored: List[Dict[str, Any]] = []
    for row in rows:
        score = cosine_similarity(query_embedding, row.get("embedding", []))
        scored.append({"score": score, "row": row})
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]
    chunks = []
    doc_ids = []
    for item in top:
        row = item["row"]
        chunks.append({
            "doc_id": row.get("doc_id"),
            "file_name": row.get("file_name"),
            "text": row.get("text"),
            "score": item["score"],
        })
        doc_ids.append(str(row.get("doc_id")))
    return {"chunks": chunks, "doc_ids": sorted(list(set(doc_ids)))}

USER_ID_PATTERN = r"^[a-zA-Z0-9_-]{3,64}$"
PHONE_PATTERN = r"^\+?[1-9]\d{7,14}$"


# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    image_urls: Optional[List[str]] = None
    request_id: Optional[str] = None


class AgentTraceItem(BaseModel):
    agent: str
    status: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    detail: Optional[str] = None


class ChatResponse(BaseModel):
    user_id: str
    response: str
    session_id: str
    request_id: Optional[str] = None
    router_label: Optional[str] = None
    router_confidence: Optional[float] = None
    route_provider: Optional[str] = None
    model_route_alias: Optional[str] = None
    runtime_model: Optional[str] = None
    fallback_triggered: Optional[bool] = None
    fallback_reason: Optional[str] = None
    route_reason: Optional[str] = None
    latency_ms: Optional[int] = None
    retrieved_chunks_count: Optional[int] = None
    source_docs_used: List[str] = []
    agent_trace: List[AgentTraceItem] = []


class SessionEnvelope(BaseModel):
    user_id: str
    session_id: str
    created_at: str
    updated_at: str
    is_deleted: bool = False
    deleted_at: Optional[str] = None
    deleted_by: Optional[str] = None
    delete_reason: Optional[str] = None
    messages: List[Dict[str, Any]] = []


class RegisterRequest(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=80)
    last_name: str = Field(..., min_length=1, max_length=80)
    email: EmailStr
    phone: Optional[str] = None
    password: str = Field(..., min_length=8, max_length=128)

    @model_validator(mode="after")
    def check_identity(self):
        if self.phone and not re.match(PHONE_PATTERN, self.phone):
            raise ValueError("Invalid phone format")
        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None


class CurrentUserResponse(BaseModel):
    user_id: str
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None


class RegisterResponse(BaseModel):
    message: str
    user_id: str
    email: str


class SessionDeleteRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    delete_reason: Optional[str] = None


class UserDeleteRequest(BaseModel):
    delete_reason: Optional[str] = None


class SessionRestoreRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    queued_at: str


class ProgressStateResponse(BaseModel):
    request_id: str
    done: bool
    events: List[Dict[str, Any]]


def load_session_envelope(user_id: str, session_id: str) -> SessionEnvelope:
    path = session_file_path(user_id, session_id)
    ts = now_iso()
    if not path.exists():
        return SessionEnvelope(
            user_id=user_id,
            session_id=session_id,
            created_at=ts,
            updated_at=ts,
            messages=[],
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Backward compatibility: old format was just a list of messages.
    if isinstance(data, list):
        return SessionEnvelope(
            user_id=user_id,
            session_id=session_id,
            created_at=ts,
            updated_at=ts,
            messages=data,
        )

    # New format envelope.
    if isinstance(data, dict) and "messages" in data:
        return SessionEnvelope(
            user_id=data.get("user_id", user_id),
            session_id=data.get("session_id", session_id),
            created_at=data.get("created_at", ts),
            updated_at=data.get("updated_at", ts),
            is_deleted=bool(data.get("is_deleted", False)),
            deleted_at=data.get("deleted_at"),
            deleted_by=data.get("deleted_by"),
            delete_reason=data.get("delete_reason"),
            messages=data.get("messages", []),
        )

    raise HTTPException(status_code=500, detail="Invalid session file format")


def save_session_envelope(env: SessionEnvelope) -> None:
    env.updated_at = now_iso()
    path = session_file_path(env.user_id, env.session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(env.model_dump(), f, indent=2, ensure_ascii=False)


async def execute_chat_pipeline(
    *,
    user_id: str,
    session_id: str,
    query: str,
    session_env: SessionEnvelope,
    image_urls: Optional[List[str]] = None,
    rag_chunks: Optional[List[Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    class AgentGraphState(TypedDict, total=False):
        user_id: str
        session_id: str
        query: str
        image_urls: List[str]
        rag_chunks: List[Dict[str, Any]]
        session_env: SessionEnvelope
        plan: Dict[str, Any]
        policy: Dict[str, Any]
        router_decision: Dict[str, Any]
        memory_result: Dict[str, Any]
        guarded_memory: Dict[str, Any]
        messages: List[Dict[str, Any]]
        generation: Dict[str, Any]
        assistant_response: str
        writer_result: Dict[str, Any]
        source_docs_used: List[str]
        retrieved_chunks_count: int
        agent_trace: List[Dict[str, Any]]

    async def run_with_retry(agent_name: str, trace: List[Dict[str, Any]], fn: Callable[[], Awaitable[Any]], detail: Optional[str] = None) -> Any:
        idx = start_agent_event(trace, agent_name, detail=detail)
        if request_id:
            progress_emit(request_id, agent_name, "running", detail)
        attempts = 2
        last_exc: Optional[Exception] = None
        started = time.time()
        AGENT_METRICS.setdefault(agent_name, {"calls": 0.0, "failures": 0.0, "total_latency_ms": 0.0})
        for attempt in range(1, attempts + 1):
            try:
                result = await fn()
                end_agent_event(trace, idx, detail=f"attempt={attempt} success")
                elapsed_ms = int((time.time() - started) * 1000)
                AGENT_METRICS[agent_name]["calls"] += 1
                AGENT_METRICS[agent_name]["total_latency_ms"] += elapsed_ms
                if request_id:
                    progress_emit(request_id, agent_name, "completed", f"attempt={attempt} success")
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    await asyncio.sleep(0.2 * attempt)
                    continue
                end_agent_event(trace, idx, status="failed", detail=f"attempts={attempts};error={exc}")
                AGENT_METRICS[agent_name]["calls"] += 1
                AGENT_METRICS[agent_name]["failures"] += 1
                AGENT_METRICS[agent_name]["total_latency_ms"] += int((time.time() - started) * 1000)
                if request_id:
                    progress_emit(request_id, agent_name, "failed", str(exc))
                raise
        raise RuntimeError(str(last_exc) if last_exc else f"{agent_name} failed")

    async def planner_node(state: AgentGraphState) -> AgentGraphState:
        trace = state["agent_trace"]
        plan = await run_with_retry(
            "PlannerAgent",
            trace,
            lambda: planner_agent(state["query"], has_images=bool(state.get("image_urls"))),
        )
        return {"plan": plan}

    async def policy_node(state: AgentGraphState) -> AgentGraphState:
        trace = state["agent_trace"]
        policy = await run_with_retry(
            "PolicyAgent",
            trace,
            lambda: policy_agent(state["query"], state["user_id"]),
        )
        return {"policy": policy}

    async def route_retrieve_node(state: AgentGraphState) -> AgentGraphState:
        trace = state["agent_trace"]

        async def _run():
            router_task = asyncio.create_task(
                router_agent(
                    state["query"],
                    retrieval_quality_label="medium",
                    has_images=bool(state.get("image_urls")),
                )
            )
            mem_task = asyncio.create_task(
                memory_retriever_agent(state["user_id"], state["session_env"], state["query"])
            )
            router_decision, memory_result = await asyncio.gather(router_task, mem_task)
            return router_decision, memory_result

        router_decision, memory_result = await run_with_retry("RouterAndMemoryAgent", trace, _run)
        return {"router_decision": router_decision, "memory_result": memory_result}

    async def guardrail_node(state: AgentGraphState) -> AgentGraphState:
        trace = state["agent_trace"]
        guarded = await run_with_retry(
            "GuardrailAgent",
            trace,
            lambda: guardrail_agent(state["memory_result"]),
        )
        return {"guarded_memory": guarded}

    async def llm_node(state: AgentGraphState) -> AgentGraphState:
        trace = state["agent_trace"]

        async def _run():
            messages: List[Dict[str, Any]] = [{"role": "system", "content": PERSONALITY}]
            long_hits = state["guarded_memory"].get("long_memory_hits", [])
            if long_hits:
                memory_lines = [f"- {hit.get('fact_text', '')}" for hit in long_hits[:MEMORY_RETRIEVAL_TOP_K]]
                messages.append({
                    "role": "system",
                    "content": "Relevant long-term user memory:\n" + "\n".join(memory_lines),
                })
            rag_chunks_local = state.get("rag_chunks", [])
            if rag_chunks_local:
                rag_lines = [f"- ({chunk.get('file_name','doc')}) {chunk.get('text','')}" for chunk in rag_chunks_local[:RAG_TOP_K]]
                messages.append({
                    "role": "system",
                    "content": "Relevant uploaded document context:\n" + "\n".join(rag_lines),
                })

            messages.extend(state["guarded_memory"].get("session_tail", []))
            if state.get("image_urls"):
                user_content: List[Dict[str, Any]] = [{"type": "text", "text": state["query"]}]
                for url in state["image_urls"][:3]:
                    user_content.append({"type": "image_url", "image_url": {"url": url}})
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": state["query"]})

            gen = await asyncio.to_thread(generate_with_route, messages, state["router_decision"])
            return messages, gen

        messages, generation = await run_with_retry(
            "LLMExpertAgent",
            trace,
            _run,
            detail=str(state["router_decision"].get("expert_label")),
        )
        assistant_response = generation["text"]
        return {"messages": messages, "generation": generation, "assistant_response": assistant_response}

    async def memory_writer_node(state: AgentGraphState) -> AgentGraphState:
        trace = state["agent_trace"]
        writer_result = await run_with_retry(
            "MemoryWriterAgent",
            trace,
            lambda: memory_writer_agent(state["user_id"], state["session_id"], state["query"], state["assistant_response"]),
        )
        return {"writer_result": writer_result}

    async def finalize_node(state: AgentGraphState) -> AgentGraphState:
        if state.get("image_urls"):
            state["session_env"].messages.append({"role": "user", "content": state["messages"][-1]["content"]})
        else:
            state["session_env"].messages.append({"role": "user", "content": state["query"]})
        state["session_env"].messages.append({"role": "assistant", "content": state["assistant_response"]})
        save_session_envelope(state["session_env"])

        source_docs_used = sorted(list({str(c.get("doc_id")) for c in state.get("rag_chunks", []) if c.get("doc_id")}))
        return {
            "source_docs_used": source_docs_used,
            "retrieved_chunks_count": len(state.get("rag_chunks", [])),
        }

    graph = StateGraph(AgentGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("policy", policy_node)
    graph.add_node("route_retrieve", route_retrieve_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("llm", llm_node)
    graph.add_node("memory_writer", memory_writer_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "policy")
    graph.add_edge("policy", "route_retrieve")
    graph.add_edge("route_retrieve", "guardrail")
    graph.add_edge("guardrail", "llm")
    graph.add_edge("llm", "memory_writer")
    graph.add_edge("memory_writer", "finalize")
    graph.add_edge("finalize", END)

    runnable = graph.compile()
    initial_state: AgentGraphState = {
        "user_id": user_id,
        "session_id": session_id,
        "query": query,
        "image_urls": image_urls or [],
        "rag_chunks": rag_chunks or [],
        "session_env": session_env,
        "agent_trace": [],
    }
    final_state = await runnable.ainvoke(initial_state)

    return {
        "assistant_response": final_state["assistant_response"],
        "router_decision": final_state["router_decision"],
        "generation": final_state["generation"],
        "agent_trace": final_state["agent_trace"],
        "retrieved_chunks_count": final_state.get("retrieved_chunks_count", len(rag_chunks or [])),
        "source_docs_used": final_state.get("source_docs_used", []),
    }


async def ingestion_worker() -> None:
    while True:
        job = await INGESTION_QUEUE.get()
        job_id = str(job["job_id"])
        try:
            INGESTION_JOBS[job_id]["status"] = "processing"
            docs = await ingest_raw_files_agent(
                user_id=str(job["user_id"]),
                session_id=str(job["session_id"]),
                user_prompt=str(job.get("message", "")),
                raw_files=list(job.get("files", [])),
            )
            chunks = await chunking_agent(docs)
            emb = await embedding_agent(str(job["user_id"]), str(job["session_id"]), chunks)
            INGESTION_JOBS[job_id]["status"] = "completed"
            INGESTION_JOBS[job_id]["result"] = {
                "parsed_docs": len(docs),
                "chunks": len(chunks),
                "stored_chunks": emb.get("stored_chunks", 0),
            }
            INGESTION_JOBS[job_id]["updated_at"] = now_iso()
        except Exception as exc:
            INGESTION_JOBS[job_id]["status"] = "failed"
            INGESTION_JOBS[job_id]["error"] = str(exc)
            INGESTION_JOBS[job_id]["updated_at"] = now_iso()
        finally:
            INGESTION_QUEUE.task_done()


@app.on_event("startup")
async def _startup_background_workers():
    if INGESTION_QUEUE_ENABLED:
        asyncio.create_task(ingestion_worker())



@app.get("/")
async def root():
    return {"message": "AI Digital Twin API with Memory"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/auth/register", response_model=RegisterResponse)
async def register(request: RegisterRequest):
    users = load_users()
    email = normalize_email(request.email)
    phone = normalize_phone(request.phone)

    for user in users:
        if email and user.get("email") == email:
            raise HTTPException(status_code=409, detail="Email already registered")
        if phone and user.get("phone") == phone:
            raise HTTPException(status_code=409, detail="Phone already registered")

    user_id = str(uuid.uuid4())
    user_record = {
        "user_id": user_id,
        "first_name": request.first_name.strip(),
        "last_name": request.last_name.strip(),
        "email": email,
        "phone": phone,
        "password_hash": hash_password(request.password),
        "is_email_verified": False,
        "created_at": now_iso(),
    }
    users.append(user_record)
    save_users(users)

    verify_token = create_email_verification_token(user_id, email)
    verification_url = f"{API_PUBLIC_BASE_URL}/auth/verify-email?token={verify_token}"
    send_verification_email(email, verification_url)

    return RegisterResponse(
        message="Registration successful. Please verify your email before logging in.",
        user_id=user_id,
        email=email,
    )


@app.get("/auth/verify-email")
async def verify_email(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        if payload.get("purpose") != "email_verify":
            raise HTTPException(status_code=400, detail="Invalid verification token")
        user_id = payload.get("sub")
        email = payload.get("email")
        if not user_id or not email:
            raise HTTPException(status_code=400, detail="Invalid verification token")
    except JWTError as exc:
        raise HTTPException(status_code=400, detail="Invalid or expired verification token") from exc

    users = load_users()
    matched = None
    for user in users:
        if user["user_id"] == user_id and user.get("email") == email:
            matched = user
            break

    if not matched:
        raise HTTPException(status_code=404, detail="User not found for verification")

    matched["is_email_verified"] = True
    matched["email_verified_at"] = now_iso()
    save_users(users)
    return {"message": "Email verified successfully. You can now log in."}


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    users = load_users()
    email = normalize_email(request.email)
    matched = None

    for user in users:
        if user.get("email") == email:
            matched = user
            break

    if not matched or not verify_password(request.password, matched["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not matched.get("is_email_verified", False):
        raise HTTPException(status_code=403, detail="Please verify your email before logging in")

    token = create_access_token(matched["user_id"])
    return AuthResponse(
        access_token=token,
        user_id=matched["user_id"],
        first_name=matched["first_name"],
        last_name=matched["last_name"],
        email=matched.get("email"),
        phone=matched.get("phone"),
    )


@app.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return CurrentUserResponse(
        user_id=current_user["user_id"],
        first_name=current_user["first_name"],
        last_name=current_user["last_name"],
        email=current_user.get("email"),
        phone=current_user.get("phone"),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        request_id = request.request_id or str(uuid.uuid4())
        user_id = current_user["user_id"]
        session_id = request.session_id or str(uuid.uuid4())
        progress_init(request_id, user_id)

        session_env = load_session_envelope(user_id, session_id)
        if session_env.is_deleted:
            raise HTTPException(status_code=404, detail="Session is deleted. Restore it or start a new session.")

        result = await execute_chat_pipeline(
            user_id=user_id,
            session_id=session_id,
            query=request.message,
            session_env=session_env,
            image_urls=request.image_urls,
            rag_chunks=None,
            request_id=request_id,
        )
        router_decision = result["router_decision"]
        gen = result["generation"]
        assistant_response = result["assistant_response"]
        agent_trace = result["agent_trace"]

        append_route_telemetry({
            "timestamp": now_iso(),
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
            "query": request.message,
            "router_label": router_decision["expert_label"],
            "router_raw_label": router_decision["raw_expert_label"],
            "router_confidence": router_decision["confidence"],
            "route_provider": gen["provider_used"],
            "route_model_alias": router_decision["model_route_alias"],
            "route_reason": router_decision.get("reason"),
            "runtime_model": gen["model_used"],
            "fallback_triggered": bool(gen["fallback_triggered"] or router_decision["fallback_triggered"]),
            "fallback_reason": gen["fallback_reason"],
            "latency_ms": gen["latency_ms"],
            "success": True,
            "retrieved_chunks_count": result["retrieved_chunks_count"],
            "source_docs_used": result["source_docs_used"],
            "agent_trace": agent_trace,
        })
        progress_done(request_id)

        return ChatResponse(
            user_id=user_id,
            response=assistant_response,
            session_id=session_id,
            request_id=request_id,
            router_label=router_decision["expert_label"],
            router_confidence=router_decision["confidence"],
            route_provider=gen["provider_used"],
            model_route_alias=router_decision["model_route_alias"],
            runtime_model=gen["model_used"],
            fallback_triggered=gen["fallback_triggered"] or router_decision["fallback_triggered"],
            fallback_reason=gen["fallback_reason"],
            route_reason=router_decision.get("reason"),
            latency_ms=gen["latency_ms"],
            retrieved_chunks_count=result["retrieved_chunks_count"],
            source_docs_used=result["source_docs_used"],
            agent_trace=agent_trace,
        )
    except Exception as e:
        append_route_telemetry({
            "timestamp": now_iso(),
            "request_id": str(uuid.uuid4()),
            "user_id": current_user.get("user_id", "unknown"),
            "session_id": request.session_id,
            "query": request.message,
            "router_label": None,
            "router_raw_label": None,
            "router_confidence": None,
            "route_provider": None,
            "route_model_alias": None,
            "runtime_model": None,
            "fallback_triggered": None,
            "fallback_reason": None,
            "latency_ms": None,
            "success": False,
            "error": str(e),
            "agent_trace": [],
        })
        if request.request_id:
            progress_done(request.request_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/rag", response_model=ChatResponse)
async def chat_with_files(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    request_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    ingestion_trace: List[Dict[str, Any]] = []
    try:
        request_id = request_id or str(uuid.uuid4())
        user_id = current_user["user_id"]
        final_session_id = session_id or str(uuid.uuid4())
        progress_init(request_id, user_id)
        session_env = load_session_envelope(user_id, final_session_id)
        if session_env.is_deleted:
            raise HTTPException(status_code=404, detail="Session is deleted. Restore it or start a new session.")

        docs: List[Dict[str, Any]] = []
        rag_chunks: List[Dict[str, Any]] = []
        source_docs_used: List[str] = []

        if files:
            ingestion_idx = start_agent_event(ingestion_trace, "FileIngestionAgent", detail=f"files={len(files)}")
            docs = await file_ingestion_agent(user_id, final_session_id, message, files)
            end_agent_event(ingestion_trace, ingestion_idx, detail=f"parsed_docs={len(docs)}")

            chunk_idx = start_agent_event(ingestion_trace, "ChunkingAgent")
            chunks = await chunking_agent(docs)
            end_agent_event(ingestion_trace, chunk_idx, detail=f"chunks={len(chunks)}")

            embed_idx = start_agent_event(ingestion_trace, "EmbeddingAgent")
            emb_res = await embedding_agent(user_id, final_session_id, chunks)
            end_agent_event(ingestion_trace, embed_idx, detail=f"stored_chunks={emb_res.get('stored_chunks', 0)}")

        retrieve_idx = start_agent_event(ingestion_trace, "RAGRetrieverAgent")
        rag_result = await rag_retriever_agent(user_id, message, top_k=RAG_TOP_K)
        rag_chunks = rag_result.get("chunks", [])
        source_docs_used = rag_result.get("doc_ids", [])
        end_agent_event(ingestion_trace, retrieve_idx, detail=f"retrieved_chunks={len(rag_chunks)}")

        result = await execute_chat_pipeline(
            user_id=user_id,
            session_id=final_session_id,
            query=message,
            session_env=session_env,
            rag_chunks=rag_chunks,
            request_id=request_id,
        )
        router_decision = result["router_decision"]
        gen = result["generation"]
        assistant_response = result["assistant_response"]
        agent_trace = ingestion_trace + result["agent_trace"]

        append_route_telemetry({
            "timestamp": now_iso(),
            "request_id": request_id,
            "user_id": user_id,
            "session_id": final_session_id,
            "query": message,
            "router_label": router_decision["expert_label"],
            "router_raw_label": router_decision["raw_expert_label"],
            "router_confidence": router_decision["confidence"],
            "route_provider": gen["provider_used"],
            "route_model_alias": router_decision["model_route_alias"],
            "route_reason": router_decision.get("reason"),
            "runtime_model": gen["model_used"],
            "fallback_triggered": bool(gen["fallback_triggered"] or router_decision["fallback_triggered"]),
            "fallback_reason": gen["fallback_reason"],
            "latency_ms": gen["latency_ms"],
            "success": True,
            "retrieved_chunks_count": len(rag_chunks),
            "source_docs_used": source_docs_used,
            "uploaded_docs_count": len(docs),
            "agent_trace": agent_trace,
        })
        progress_done(request_id)

        return ChatResponse(
            user_id=user_id,
            response=assistant_response,
            session_id=final_session_id,
            request_id=request_id,
            router_label=router_decision["expert_label"],
            router_confidence=router_decision["confidence"],
            route_provider=gen["provider_used"],
            model_route_alias=router_decision["model_route_alias"],
            runtime_model=gen["model_used"],
            fallback_triggered=gen["fallback_triggered"] or router_decision["fallback_triggered"],
            fallback_reason=gen["fallback_reason"],
            route_reason=router_decision.get("reason"),
            latency_ms=gen["latency_ms"],
            retrieved_chunks_count=len(rag_chunks),
            source_docs_used=source_docs_used,
            agent_trace=agent_trace,
        )
    except Exception as e:
        append_route_telemetry({
            "timestamp": now_iso(),
            "request_id": str(uuid.uuid4()),
            "user_id": current_user.get("user_id", "unknown"),
            "session_id": session_id,
            "query": message,
            "router_label": None,
            "router_raw_label": None,
            "router_confidence": None,
            "route_provider": None,
            "route_model_alias": None,
            "runtime_model": None,
            "fallback_triggered": None,
            "fallback_reason": None,
            "latency_ms": None,
            "success": False,
            "error": str(e),
            "agent_trace": ingestion_trace,
        })
        if request_id:
            progress_done(request_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experts")
async def list_experts():
    return {
        "experts": EXPERT_TO_PROVIDER_ROUTE,
        "config_env_keys": [
            "ROUTER_BACKEND",
            "ML_EXPERT_MODEL",
            "MATH_EXPERT_MODEL",
            "DL_EXPERT_MODEL",
            "GENAI_EXPERT_MODEL",
            "RESEARCH_EXPERT_MODEL",
            "AGENTIC_EXPERT_MODEL",
            "RAG_EXPERT_MODEL",
            "LLM_EVAL_EXPERT_MODEL",
            "FRIENDLY_EXPERT_MODEL",
            "ROUTER_NEURAL_PATH",
            "ROUTER_NEURAL_META_PATH",
            "ROUTER_MODEL_PATH",
            "ROUTER_META_PATH",
            "MULTIMODAL_OPENAI_MODEL",
            "OPENAI_FALLBACK_MODEL",
            "EMBEDDING_MODEL",
        ],
    }


@app.get("/chat/progress/{request_id}", response_model=ProgressStateResponse)
async def get_chat_progress(request_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    state = PROGRESS_STATE.get(request_id)
    if not state or state.get("user_id") != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Progress request not found")
    return ProgressStateResponse(
        request_id=request_id,
        done=bool(state.get("done", False)),
        events=list(state.get("events", [])),
    )


@app.post("/documents/ingest", response_model=IngestionJobResponse)
async def queue_document_ingestion(
    message: str = Form(""),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    if not INGESTION_QUEUE_ENABLED:
        raise HTTPException(status_code=400, detail="Ingestion queue is disabled")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    user_id = current_user["user_id"]
    final_session_id = session_id or str(uuid.uuid4())
    payload_files: List[Dict[str, Any]] = []
    for upload in files:
        payload_files.append(
            {
                "filename": upload.filename or "upload",
                "content_type": upload.content_type or "",
                "data": await upload.read(),
            }
        )

    job_id = str(uuid.uuid4())
    INGESTION_JOBS[job_id] = {
        "job_id": job_id,
        "user_id": user_id,
        "session_id": final_session_id,
        "status": "queued",
        "queued_at": now_iso(),
        "updated_at": now_iso(),
    }
    await INGESTION_QUEUE.put(
        {
            "job_id": job_id,
            "user_id": user_id,
            "session_id": final_session_id,
            "message": message,
            "files": payload_files,
        }
    )
    return IngestionJobResponse(job_id=job_id, status="queued", queued_at=INGESTION_JOBS[job_id]["queued_at"])


@app.get("/documents/ingest/{job_id}")
async def get_document_ingest_status(job_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    job = INGESTION_JOBS.get(job_id)
    if not job or job.get("user_id") != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job


@app.get("/agent-metrics")
async def get_agent_metrics(current_user: Dict[str, Any] = Depends(get_current_user)):
    out = {}
    for agent, data in AGENT_METRICS.items():
        calls = max(1.0, data.get("calls", 0.0))
        out[agent] = {
            "calls": int(data.get("calls", 0.0)),
            "failures": int(data.get("failures", 0.0)),
            "failure_rate": float(data.get("failures", 0.0) / calls),
            "avg_latency_ms": float(data.get("total_latency_ms", 0.0) / calls),
        }
    return {"agents": out}


@app.get("/sessions")
async def list_sessions(include_deleted: bool = False, current_user: Dict[str, Any] = Depends(get_current_user)):
    """List only one user's sessions"""
    user_id = current_user["user_id"]
    user_dir = MEMORY_DIR / user_id
    if not user_dir.exists() or not user_dir.is_dir():
        return {"user_id": user_id, "sessions": []}

    sessions = []
    for file_path in user_dir.glob("*.json"):
        session_id = file_path.stem
        session_env = load_session_envelope(user_id, session_id)
        if session_env.is_deleted and not include_deleted:
            continue
        sessions.append({
            "user_id": user_id,
            "session_id": session_env.session_id,
            "message_count": len(session_env.messages),
            "last_message": session_env.messages[-1]["content"] if session_env.messages else None,
            "is_deleted": session_env.is_deleted,
            "updated_at": session_env.updated_at,
        })

    return {"user_id": user_id, "sessions": sessions}


@app.get("/memory/export")
async def export_memory(include_deleted: bool = False, current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["user_id"]
    user_dir = MEMORY_DIR / user_id
    sessions: List[Dict[str, Any]] = []
    if user_dir.exists() and user_dir.is_dir():
        for file_path in user_dir.glob("*.json"):
            session_env = load_session_envelope(user_id, file_path.stem)
            if session_env.is_deleted and not include_deleted:
                continue
            sessions.append(session_env.model_dump())

    append_audit_log(
        action="memory_export",
        user_id=user_id,
        detail=f"include_deleted={include_deleted};sessions={len(sessions)}",
    )
    return {"user_id": user_id, "session_count": len(sessions), "sessions": sessions}


@app.post("/memory/session/delete")
async def soft_delete_session(
    request: SessionDeleteRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    env = load_session_envelope(user_id, request.session_id)
    if not session_file_path(user_id, request.session_id).exists():
        raise HTTPException(status_code=404, detail="Session not found")
    if env.is_deleted:
        return {"message": "Session already deleted", "session_id": request.session_id}

    env.is_deleted = True
    env.deleted_at = now_iso()
    env.deleted_by = user_id
    env.delete_reason = request.delete_reason
    save_session_envelope(env)
    append_audit_log("session_soft_delete", user_id, request.session_id, request.delete_reason)
    return {"message": "Session soft deleted", "session_id": request.session_id}


@app.post("/memory/user/delete")
async def soft_delete_user_memory(
    request: UserDeleteRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    user_dir = MEMORY_DIR / user_id
    if not user_dir.exists() or not user_dir.is_dir():
        return {"message": "No memory found", "deleted_sessions": 0}

    deleted_sessions = 0
    for file_path in user_dir.glob("*.json"):
        env = load_session_envelope(user_id, file_path.stem)
        if env.is_deleted:
            continue
        env.is_deleted = True
        env.deleted_at = now_iso()
        env.deleted_by = user_id
        env.delete_reason = request.delete_reason
        save_session_envelope(env)
        deleted_sessions += 1

    append_audit_log("user_soft_delete", user_id, detail=f"sessions={deleted_sessions};reason={request.delete_reason}")
    return {"message": "User memory soft deleted", "deleted_sessions": deleted_sessions}


@app.post("/memory/session/restore")
async def restore_session(
    request: SessionRestoreRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    path = session_file_path(user_id, request.session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    env = load_session_envelope(user_id, request.session_id)
    env.is_deleted = False
    env.deleted_at = None
    env.deleted_by = None
    env.delete_reason = None
    save_session_envelope(env)
    append_audit_log("session_restore", user_id, request.session_id)
    return {"message": "Session restored", "session_id": request.session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
