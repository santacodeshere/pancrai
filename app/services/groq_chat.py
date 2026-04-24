"""
PancrAI — AI Chat Assistant via Groq + Llama 3
Conversational assistant with full scan context awareness.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_ID = "llama-3.1-70b-versatile"
MAX_HISTORY = 10   # keep last 10 message pairs


SYSTEM_PROMPT_TEMPLATE = """You are PancrAI Assistant, an expert AI system specialized in 
pancreatic radiology and oncology. You assist radiologists and oncologists in 
interpreting AI-assisted pancreatic tumor analysis results.

Current Patient Scan Analysis:
{context}

Your role:
- Explain segmentation results, Grad-CAM heatmaps, and uncertainty scores in clinical terms
- Answer questions about pancreatic tumors, PDAC, IPMN, benign tumors
- Provide information on staging, treatment options, and clinical guidelines
- Reference current clinical guidelines (NCCN, ESMO, IAP guidelines)
- Always remind users that AI findings must be validated by a qualified physician

Guidelines:
- Be precise and use appropriate medical terminology
- Provide evidence-based information
- Acknowledge limitations of AI analysis
- Never make definitive diagnoses — present findings as AI-assisted observations
- If asked about surgical options, chemotherapy, or specific treatments, 
  provide general clinical information and recommend specialist consultation

Tone: Professional, helpful, concise but thorough.
"""


def _build_context_string(prediction_context: Optional[Dict[str, Any]]) -> str:
    """Format prediction data as a readable context string for the system prompt."""
    if not prediction_context:
        return "No scan analysis loaded. User is asking general questions."

    ctx_lines = []
    tc = prediction_context.get("tumor_class", "Unknown")
    conf = prediction_context.get("primary_confidence", 0.0)
    risk = prediction_context.get("risk_level", "Unknown")
    unc = prediction_context.get("uncertainty_score", 0.0)
    dice = prediction_context.get("dice_score", 0.0)
    iou = prediction_context.get("iou_score", 0.0)

    ctx_lines.append(f"- Detection result: {tc}")
    ctx_lines.append(f"- Primary confidence: {conf*100:.1f}%")
    ctx_lines.append(f"- Risk level: {risk}")
    ctx_lines.append(f"- Model uncertainty: {unc:.1f}/100")
    ctx_lines.append(f"- Segmentation Dice score: {dice:.4f}")
    ctx_lines.append(f"- Segmentation IoU: {iou:.4f}")

    scores = prediction_context.get("confidence_scores", [])
    if scores and len(scores) == 4:
        ctx_lines.append(
            f"- Class probabilities: No Tumor {scores[0]*100:.1f}%, "
            f"Benign {scores[1]*100:.1f}%, "
            f"Malignant {scores[2]*100:.1f}%, "
            f"Cystic {scores[3]*100:.1f}%"
        )

    m = prediction_context.get("measurements")
    if m:
        ctx_lines.append(
            f"- Tumor area: {m.get('area_cm2', 'N/A')} cm²"
        )
        ctx_lines.append(
            f"- Tumor centroid: ({m.get('centroid_x', 'N/A')}, "
            f"{m.get('centroid_y', 'N/A')}) pixels"
        )

    patient = prediction_context.get("patient", {})
    if patient:
        ctx_lines.append(
            f"- Patient: {patient.get('name', 'N/A')}, "
            f"age {patient.get('age', 'N/A')}, "
            f"sex {patient.get('sex', 'N/A')}"
        )

    return "\n".join(ctx_lines)


def chat(
    message: str,
    history: List[Dict[str, str]],
    prediction_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Send a message to the Groq chat API and get a response.

    Args:
        message: User's current message.
        history: List of previous messages [{"role": "user/assistant", "content": "..."}]
        prediction_context: Current scan prediction data for context injection.

    Returns:
        Assistant response string.
    """
    if not GROQ_API_KEY:
        return _offline_response(message, prediction_context)

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        # Build context-aware system prompt
        context_str = _build_context_string(prediction_context)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_str)

        # Keep only last MAX_HISTORY messages
        recent_history = history[-(MAX_HISTORY * 2):]

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=1024,
            temperature=0.5,
            top_p=0.95,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"[Groq Chat] Error: {e}")
        return _offline_response(message, prediction_context)


def _offline_response(message: str,
                       context: Optional[Dict]) -> str:
    """Offline fallback response when Groq API is unavailable."""
    msg_lower = message.lower()
    ctx = context or {}
    tumor_class = ctx.get("tumor_class", "the detected finding")

    if any(w in msg_lower for w in ["what", "mean", "explain", "segmentation"]):
        return (
            f"The AI segmentation identified **{tumor_class}** in the pancreatic region. "
            "The colored overlay shows the regions the model attributed highest probability "
            "to the tumor class. The Grad-CAM heatmap highlights which image features most "
            "influenced this decision.\n\n"
            "*(Groq API not configured — please add GROQ_API_KEY to .env for full AI chat)*"
        )
    elif any(w in msg_lower for w in ["survival", "prognosis", "stage"]):
        return (
            "Pancreatic ductal adenocarcinoma (PDAC) prognosis varies significantly by stage:\n"
            "- **Stage I**: Localized, 5-year survival ~20-30% with surgical resection\n"
            "- **Stage II**: Regional involvement, 5-year survival ~10-15%\n"
            "- **Stage III**: Locally advanced, surgical resection often not possible\n"
            "- **Stage IV**: Metastatic, median survival 6-12 months with systemic therapy\n\n"
            "Please consult NCCN Pancreatic Adenocarcinoma Guidelines for current management.\n\n"
            "*(Groq API not configured — add GROQ_API_KEY to .env for full AI responses)*"
        )
    elif any(w in msg_lower for w in ["biopsy", "next", "recommend", "treatment"]):
        return (
            "Recommended clinical workup based on AI findings:\n"
            "1. **EUS-FNA** (Endoscopic Ultrasound with Fine Needle Aspiration) — gold standard for tissue diagnosis\n"
            "2. **Serum markers**: CA 19-9, CEA, IgG4\n"
            "3. **MRI/MRCP** if not already performed — better soft tissue characterization\n"
            "4. **Multidisciplinary team (MDT) review** — surgical, medical oncology, gastroenterology\n"
            "5. **PET-CT** if malignancy confirmed — staging assessment\n\n"
            "*(Groq API not configured — add GROQ_API_KEY to .env for full AI responses)*"
        )
    else:
        return (
            "I'm PancrAI's clinical assistant. I can help explain scan results, "
            "tumor characteristics, differential diagnoses, staging, and treatment options "
            "for pancreatic conditions.\n\n"
            "**Note:** Groq API is not configured. Please add `GROQ_API_KEY` to your `.env` "
            "file to enable full AI-powered responses using Llama 3."
        )
