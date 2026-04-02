from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, EmailStr, model_validator
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
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

# Memory directory
MEMORY_DIR = Path("../memory")
MEMORY_DIR.mkdir(exist_ok=True)
USERS_FILE = MEMORY_DIR / "users.json"
AUDIT_FILE = MEMORY_DIR / "audit.log"
ROUTE_TELEMETRY_FILE = MEMORY_DIR / "route_telemetry.jsonl"


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

router_model = None
router_meta: Optional[Dict[str, Any]] = None
try:
    router_model = joblib.load(ROUTER_MODEL_PATH)
    with open(ROUTER_META_PATH, "r", encoding="utf-8") as f:
        router_meta = json.load(f)
    print(f"[router] loaded model from {ROUTER_MODEL_PATH}")
except Exception as e:
    print(f"[router] not loaded, fallback mode active: {e}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    """Predict route using trained LR router; fallback to GPT route if unavailable/low confidence."""
    heuristic_expert = select_heuristic_expert(query, has_images=has_images)
    if router_model is None:
        selected = heuristic_expert or "gpt_fallback"
        route = EXPERT_TO_PROVIDER_ROUTE.get(selected, EXPERT_TO_PROVIDER_ROUTE["gpt_fallback"])
        return {
            "expert_label": selected,
            "raw_expert_label": "gpt_fallback",
            "confidence": 0.75 if heuristic_expert else 0.0,
            "fallback_triggered": selected == "gpt_fallback",
            "reason": "heuristic_no_router" if heuristic_expert else "router_not_loaded",
            "provider": route["provider"],
            "model_route_alias": route["model"],
        }

    q_lower = query.lower()
    contains_code = int(any(tok in q_lower for tok in ["traceback", "exception", "error", "code", "python", "npm", "fastapi", "tsx", "{", "}"]))
    error_log_present = int(any(tok in q_lower for tok in ["traceback", "exception", "error:", "failed", "module not found"]))
    memory_needed = int(any(tok in q_lower for tok in ["remember", "earlier", "previous", "my role", "my goals", "what did i say"]))
    multi_hop = int(any(tok in q_lower for tok in ["compare", "tradeoff", "strategy", "roadmap", "analyze", "design"]))

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

    row = pd.DataFrame([{
        "query": query,
        "contains_code": contains_code,
        "error_log_present": error_log_present,
        "memory_needed": memory_needed,
        "multi_hop": multi_hop,
        "estimated_input_tokens": estimated_tokens,
        "latency_budget_ms": latency_budget_ms,
        "difficulty": difficulty,
        "retrieval_quality_label": retrieval_quality_label,
    }])

    probs = router_model.predict_proba(row)[0]
    classes = router_model.named_steps["clf"].classes_
    idx = int(np.argmax(probs))
    raw_expert = str(classes[idx])
    confidence = float(probs[idx])

    fallback_triggered = confidence < ROUTER_CONF_THRESHOLD
    final_expert = "gpt_fallback" if fallback_triggered else raw_expert
    route_reason = "router_prediction"
    if fallback_triggered:
        route_reason = "router_low_confidence"

    # Heuristic overrides let us support newly added experts before retraining the LR router.
    if heuristic_expert:
        final_expert = heuristic_expert
        fallback_triggered = final_expert == "gpt_fallback"
        route_reason = "heuristic_override"

    route = EXPERT_TO_PROVIDER_ROUTE.get(final_expert, EXPERT_TO_PROVIDER_ROUTE["gpt_fallback"])
    return {
        "expert_label": final_expert,
        "raw_expert_label": raw_expert,
        "confidence": confidence,
        "fallback_triggered": fallback_triggered,
        "reason": route_reason,
        "provider": route["provider"],
        "model_route_alias": route["model"],
        "features": {
            "contains_code": contains_code,
            "error_log_present": error_log_present,
            "memory_needed": memory_needed,
            "multi_hop": multi_hop,
            "estimated_input_tokens": estimated_tokens,
            "difficulty": difficulty,
            "retrieval_quality_label": retrieval_quality_label,
            "latency_budget_ms": latency_budget_ms,
        },
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


async def file_ingestion_agent(user_id: str, session_id: str, user_prompt: str, files: List[UploadFile]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    docs_dir = user_docs_dir(user_id)
    for upload in files:
        raw = await upload.read()
        if not raw:
            continue
        original_name = safe_filename(upload.filename or "upload")
        digest = hashlib.sha1(raw).hexdigest()[:10]
        doc_id = f"{session_id}-{digest}"
        saved_path = docs_dir / f"{doc_id}-{original_name}"
        saved_path.write_bytes(raw)

        lower_name = original_name.lower()
        mime = upload.content_type or ""
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
) -> Dict[str, Any]:
    agent_trace: List[Dict[str, Any]] = []

    planner_idx = start_agent_event(agent_trace, "PlannerAgent")
    plan = await planner_agent(query, has_images=bool(image_urls))
    end_agent_event(agent_trace, planner_idx, detail=f"expert_hint={plan.get('expert_hint')}")

    router_idx = start_agent_event(agent_trace, "RouterAgent")
    memory_idx = start_agent_event(agent_trace, "MemoryRetrieverAgent")
    router_task = asyncio.create_task(
        router_agent(
            query,
            retrieval_quality_label="medium",
            has_images=bool(image_urls),
        )
    )
    memory_task = asyncio.create_task(memory_retriever_agent(user_id, session_env, query))
    router_decision, memory_result = await asyncio.gather(router_task, memory_task)
    end_agent_event(
        agent_trace,
        router_idx,
        detail=f"{router_decision.get('expert_label')} ({router_decision.get('confidence', 0):.2f})",
    )
    end_agent_event(
        agent_trace,
        memory_idx,
        detail=f"session_tail={len(memory_result.get('session_tail', []))}, long_hits={len(memory_result.get('long_memory_hits', []))}",
    )

    guardrail_idx = start_agent_event(agent_trace, "GuardrailAgent")
    guarded_memory = await guardrail_agent(memory_result)
    end_agent_event(agent_trace, guardrail_idx, detail="sanitized memory context")

    messages: List[Dict[str, Any]] = [{"role": "system", "content": PERSONALITY}]
    long_hits = guarded_memory.get("long_memory_hits", [])
    if long_hits:
        memory_lines = [f"- {hit.get('fact_text', '')}" for hit in long_hits[:MEMORY_RETRIEVAL_TOP_K]]
        messages.append({
            "role": "system",
            "content": "Relevant long-term user memory:\n" + "\n".join(memory_lines),
        })
    if rag_chunks:
        rag_lines = [f"- ({chunk.get('file_name','doc')}) {chunk.get('text','')}" for chunk in rag_chunks[:RAG_TOP_K]]
        messages.append({
            "role": "system",
            "content": "Relevant uploaded document context:\n" + "\n".join(rag_lines),
        })

    messages.extend(guarded_memory.get("session_tail", []))
    if image_urls:
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": query}]
        for url in image_urls[:3]:
            user_content.append({"type": "image_url", "image_url": {"url": url}})
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": query})

    llm_idx = start_agent_event(agent_trace, "LLMExpertAgent", detail=router_decision.get("expert_label"))
    gen = await asyncio.to_thread(generate_with_route, messages, router_decision)
    end_agent_event(
        agent_trace,
        llm_idx,
        detail=f"{gen.get('provider_used')}::{gen.get('model_used')} latency={gen.get('latency_ms')}ms",
    )
    assistant_response = gen["text"]

    writer_idx = start_agent_event(agent_trace, "MemoryWriterAgent")
    writer_result = await memory_writer_agent(user_id, session_id, query, assistant_response)
    end_agent_event(agent_trace, writer_idx, detail=f"stored={writer_result.get('stored', 0)}")

    if image_urls:
        session_env.messages.append({"role": "user", "content": messages[-1]["content"]})
    else:
        session_env.messages.append({"role": "user", "content": query})
    session_env.messages.append({"role": "assistant", "content": assistant_response})
    save_session_envelope(session_env)

    source_docs_used = sorted(list({str(c.get("doc_id")) for c in (rag_chunks or []) if c.get("doc_id")}))
    return {
        "assistant_response": assistant_response,
        "router_decision": router_decision,
        "generation": gen,
        "agent_trace": agent_trace,
        "retrieved_chunks_count": len(rag_chunks or []),
        "source_docs_used": source_docs_used,
    }



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
        request_id = str(uuid.uuid4())
        user_id = current_user["user_id"]
        session_id = request.session_id or str(uuid.uuid4())

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

        return ChatResponse(
            user_id=user_id,
            response=assistant_response,
            session_id=session_id,
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/rag", response_model=ChatResponse)
async def chat_with_files(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    ingestion_trace: List[Dict[str, Any]] = []
    try:
        request_id = str(uuid.uuid4())
        user_id = current_user["user_id"]
        final_session_id = session_id or str(uuid.uuid4())
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

        return ChatResponse(
            user_id=user_id,
            response=assistant_response,
            session_id=final_session_id,
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experts")
async def list_experts():
    return {
        "experts": EXPERT_TO_PROVIDER_ROUTE,
        "config_env_keys": [
            "ML_EXPERT_MODEL",
            "MATH_EXPERT_MODEL",
            "DL_EXPERT_MODEL",
            "GENAI_EXPERT_MODEL",
            "RESEARCH_EXPERT_MODEL",
            "AGENTIC_EXPERT_MODEL",
            "RAG_EXPERT_MODEL",
            "LLM_EVAL_EXPERT_MODEL",
            "FRIENDLY_EXPERT_MODEL",
            "MULTIMODAL_OPENAI_MODEL",
            "OPENAI_FALLBACK_MODEL",
            "EMBEDDING_MODEL",
        ],
    }


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
