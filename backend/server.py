from fastapi import FastAPI, HTTPException, Depends
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


# Load environment variables
load_dotenv(override=True)

app = FastAPI()

# Configure CORS
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

# Memory directory
MEMORY_DIR = Path("../memory")
MEMORY_DIR.mkdir(exist_ok=True)
USERS_FILE = MEMORY_DIR / "users.json"
AUDIT_FILE = MEMORY_DIR / "audit.log"


# Load personality details
def load_personality():
    with open("me.txt", "r", encoding="utf-8") as f:
        return f.read().strip()


PERSONALITY = load_personality()

EXPERT_TO_MODEL_ROUTE = {
    "memory_factual_expert": "phi-3-mini-4k-instruct",
    "technical_expert": "qwen2.5-coder-7b-instruct",
    "gpt_fallback": "gpt-4o-mini",
}

# We currently call OpenAI from this backend. For now, map non-OpenAI expert routes
# to the default OpenAI model until OSS expert serving is wired in.
MODEL_ALIAS_TO_RUNTIME_MODEL = {
    "phi-3-mini-4k-instruct": DEFAULT_OPENAI_MODEL,
    "qwen2.5-coder-7b-instruct": DEFAULT_OPENAI_MODEL,
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1",
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


def infer_router(query: str, retrieval_quality_label: str = "medium") -> Dict[str, Any]:
    """Predict route using trained LR router; fallback to GPT route if unavailable/low confidence."""
    if router_model is None:
        alias = EXPERT_TO_MODEL_ROUTE["gpt_fallback"]
        return {
            "expert_label": "gpt_fallback",
            "raw_expert_label": "gpt_fallback",
            "confidence": 0.0,
            "fallback_triggered": True,
            "reason": "router_not_loaded",
            "model_route_alias": alias,
            "runtime_model": MODEL_ALIAS_TO_RUNTIME_MODEL.get(alias, DEFAULT_OPENAI_MODEL),
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

    alias = EXPERT_TO_MODEL_ROUTE.get(final_expert, EXPERT_TO_MODEL_ROUTE["gpt_fallback"])
    runtime_model = MODEL_ALIAS_TO_RUNTIME_MODEL.get(alias, DEFAULT_OPENAI_MODEL)
    return {
        "expert_label": final_expert,
        "raw_expert_label": raw_expert,
        "confidence": confidence,
        "fallback_triggered": fallback_triggered,
        "model_route_alias": alias,
        "runtime_model": runtime_model,
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

USER_ID_PATTERN = r"^[a-zA-Z0-9_-]{3,64}$"
PHONE_PATTERN = r"^\+?[1-9]\d{7,14}$"


# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    user_id: str
    response: str
    session_id: str
    router_label: Optional[str] = None
    router_confidence: Optional[float] = None
    model_route_alias: Optional[str] = None
    runtime_model: Optional[str] = None
    fallback_triggered: Optional[bool] = None


class SessionEnvelope(BaseModel):
    user_id: str
    session_id: str
    created_at: str
    updated_at: str
    is_deleted: bool = False
    deleted_at: Optional[str] = None
    deleted_by: Optional[str] = None
    delete_reason: Optional[str] = None
    messages: List[Dict[str, str]] = []


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
        user_id = current_user["user_id"]
        session_id = request.session_id or str(uuid.uuid4())

        session_env = load_session_envelope(user_id, session_id)
        if session_env.is_deleted:
            raise HTTPException(status_code=404, detail="Session is deleted. Restore it or start a new session.")

        messages = [{"role": "system", "content": PERSONALITY}]
        messages.extend(session_env.messages)
        messages.append({"role": "user", "content": request.message})

        router_decision = infer_router(request.message, retrieval_quality_label="medium")
        selected_runtime_model = router_decision["runtime_model"]

        response = client.chat.completions.create(
            model=selected_runtime_model,
            messages=messages
        )

        assistant_response = response.choices[0].message.content

        session_env.messages.append({"role": "user", "content": request.message})
        session_env.messages.append({"role": "assistant", "content": assistant_response})
        save_session_envelope(session_env)

        return ChatResponse(
            user_id=user_id,
            response=assistant_response,
            session_id=session_id,
            router_label=router_decision["expert_label"],
            router_confidence=router_decision["confidence"],
            model_route_alias=router_decision["model_route_alias"],
            runtime_model=selected_runtime_model,
            fallback_triggered=router_decision["fallback_triggered"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
