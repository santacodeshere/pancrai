"""
PancrAI — Chat Assistant Routes
Conversational AI endpoint via Groq + Llama 3.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.groq_chat import chat

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Send a message to the AI chat assistant.

    The assistant is aware of the current scan analysis context and can
    answer clinical questions about the prediction results.

    Args:
        request.message: User's question.
        request.history: Previous conversation messages.
        request.prediction_context: Dict with current prediction data.

    Returns:
        AI assistant response.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    history = [m.model_dump() for m in (request.history or [])]

    response = chat(
        message=request.message,
        history=history,
        prediction_context=request.prediction_context,
    )

    return ChatResponse(response=response, role="assistant")
