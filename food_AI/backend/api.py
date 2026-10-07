from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your model function
from model import generate_response

app = FastAPI()

# Allow React frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # Supports full conversation history or fallback single message
    messages: Optional[List[MessageItem]] = None
    message: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    flagged: bool = False


# ----------------------------------------------------------------------
# Jailbreak Classifier Hook
# ----------------------------------------------------------------------
def detect_jailbreak(text: str) -> bool:
    """
    Plug your jailbreak classification model or guardrail pipeline here.
    Return True if malicious/adversarial, False if clean.
    """
    # Placeholder: replace with `jailbreak_model.predict(text)`
    return False


@app.get("/")
def root():
    return {"message": "FoodAssist API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # 1. Normalize input to extract the latest prompt and history
    if request.messages and len(request.messages) > 0:
        latest_user_text = request.messages[-1].content
        conversation_payload = [m.model_dump() for m in request.messages]
    elif request.message:
        latest_user_text = request.message
        conversation_payload = [{"role": "user", "content": request.message}]
    else:
        return ChatResponse(response="Please provide a message.", flagged=False)

    # 2. Run your Jailbreak Detection Guardrail
    if detect_jailbreak(latest_user_text):
        return ChatResponse(
            response="⚠️ Security Notice: This prompt has been flagged by safety guardrails.",
            flagged=True,
        )

    # 3. Generate response using Qwen (pass history or latest text)
    # If your model.py expects conversation history, pass `conversation_payload`.
    # If it expects a single string, pass `latest_user_text`.
    try:
        response_text = generate_response(conversation_payload)
    except TypeError:
        # Fallback if your generate_response only accepts a plain string
        response_text = generate_response(latest_user_text)

    return ChatResponse(
        response=response_text,
        flagged=False,
    )