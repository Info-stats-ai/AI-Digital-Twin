from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import uuid
from pathlib import Path

# Load environment variables
load_dotenv(override=True)

app = FastAPI()

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI()

# Memory directory
MEMORY_DIR = Path("../memory")
MEMORY_DIR.mkdir(exist_ok=True)


# Load personality details
def load_personality():
    with open("me.txt", "r", encoding="utf-8") as f:
        return f.read().strip()


PERSONALITY = load_personality()


def _user_memory_dir(user_id: str) -> Path:
    user_dir = MEMORY_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def load_conversation(user_id: str, session_id: str) -> List[Dict]:
    file_path = _user_memory_dir(user_id) / f"{session_id}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_conversation(user_id: str, session_id: str, messages: List[Dict]):
    file_path = _user_memory_dir(user_id) / f"{session_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)

USER_ID_PATTERN = r"^[a-zA-Z0-9_-]{3,64}$"


# Request/Response models
class ChatRequest(BaseModel):
    user_id: str = Field(..., pattern=USER_ID_PATTERN)
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    user_id: str
    response: str
    session_id: str



@app.get("/")
async def root():
    return {"message": "AI Digital Twin API with Memory"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())

        conversation = load_conversation(request.user_id, session_id)

        messages = [{"role": "system", "content": PERSONALITY}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": request.message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        assistant_response = response.choices[0].message.content

        conversation.append({"role": "user", "content": request.message})
        conversation.append({"role": "assistant", "content": assistant_response})

        save_conversation(request.user_id, session_id, conversation)

        return ChatResponse(
            user_id=request.user_id,
            response=assistant_response,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """List all conversation sessions grouped by user"""
    sessions = []
    for user_dir in MEMORY_DIR.iterdir():
        if not user_dir.is_dir():
            continue
        for file_path in user_dir.glob("*.json"):
            session_id = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                conversation = json.load(f)
                sessions.append({
                    "user_id": user_dir.name,
                    "session_id": session_id,
                    "message_count": len(conversation),
                    "last_message": conversation[-1]["content"] if conversation else None
                })
    return {"sessions": sessions}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
