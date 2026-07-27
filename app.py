import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

app = FastAPI(title="SafeSpace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_PROMPT = """
You are SafeSpace, a supportive, empathetic, and reflective conversational assistant focused on mental wellness support for users in Malaysia.

Your role is to:
- Listen with warmth and without judgement.
- Help the user reflect on what they are feeling.
- Offer practical, evidence-informed coping strategies.
- Suggest gentle actions such as breathing exercises, grounding, short walks, journaling, light exercise, sleep routines, or revisiting hobbies.
- Ask calm, helpful follow-up questions when needed.
- Keep the tone professional, reassuring, and clear.

CRITICAL CONSTRAINTS - YOU MUST FOLLOW THESE STRICTLY:
1. You are NOT a doctor, therapist, counsellor, or medical professional.
2. Never provide a diagnosis, psychological assessment, or medication advice.
3. Do not shame, lecture, or overwhelm the user.
4. Keep responses concise, usually 2-4 short paragraphs max.
5. ABSOLUTELY DO NOT include ANY phone numbers, hotline numbers, website links, URLs, or contact details in your responses. 
6. The ONLY exception is if you are responding to a CRISIS situation - then you may ONLY use the exact crisis response provided below.
7. When users ask for numbers or resources, gently explain that you're not a crisis service but you can suggest they search for "mental health support Malaysia" or visit MENTARI's official website.
8. If the user is in crisis, you must respond with the CRISIS_RESPONSE which contains the approved emergency contact details.
9. Do not use US-centric crisis references unless specifically relevant. Prefer Malaysia context.
10. Never create your own phone numbers or URLs - only use what's provided in CRISIS_RESPONSE.
"""

CRISIS_KEYWORDS = [
    r"\bsuicid\w*",
    r"\bkill myself\b",
    r"\bwant to die\b",
    r"\bend my life\b",
    r"\bself[- ]?harm\b",
    r"\bhurt myself\b",
    r"\bcutting myself\b",
    r"\boverdose\b",
    r"\bcan't go on\b",
    r"\bcan not go on\b",
    r"\bwant to disappear\b",
    r"\bemergency\b",
    r"\bneed help now\b",
    r"\bhelp me\b",
    r"\bnumber for mental health\b",  # Added to catch requests for numbers
    r"\bphone number\b",              # Added to catch requests for numbers
    r"\bcall\b.*\bhelp\b",            # Added to catch "call for help"
    r"\bcontact\b.*\bhelp\b",         # Added to catch "contact for help"
    r"\breach out\b.*\bhelp\b",       # Added to catch "reach out for help"
    r"\bhotline\b",
    r"\bhelpline\b",
]

CRISIS_RESPONSE = (
    "I'm really sorry you're carrying this much pain right now. "
    "You deserve immediate support from a real person.\n\n"
    "Please reach out right now:\n"
    "- **Malaysia emergency:** Call **999** if you may be in immediate danger.\n"
    "- **Befrienders KL:** **+603-7627 2929** (24 hours, free, confidential).\n"
    "- **Talian HEAL:** **15555**.\n"
    "- **Talian Kasih:** **15999**.\n\n"
    "If possible, stay near someone you trust, move away from anything you could use to hurt yourself, "
    "and go to the nearest hospital emergency department or ask someone to take you there now."
)

# Matches URLs and likely phone numbers in normal assistant responses.
CONTACT_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|(?:\+?\d[\d\s().-]{6,}\d))",
    re.IGNORECASE
)

# Pattern to detect if the response contains any crisis keywords (to prevent numbers in non-crisis responses)
RESPONSE_CRISIS_PATTERN = re.compile(
    r"\b(?:emergency|help now|can't go on|suicidal|self-harm)\b",
    re.IGNORECASE
)

def check_crisis_trigger(text: str) -> bool:
    """Scan input against crisis patterns."""
    text_lower = text.lower()
    for pattern in CRISIS_KEYWORDS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False

def remove_contact_details(text: str) -> str:
    """Remove phone numbers and URLs from non-crisis responses."""
    # Remove any phone numbers, URLs, or contact details
    cleaned = CONTACT_PATTERN.sub("", text)
    
    # Remove any remaining contact-related text
    cleaned = re.sub(r"(call|contact|reach out to|phone|hotline|helpline|number)\s*:?\s*\d+", "", cleaned, flags=re.IGNORECASE)
    
    # Remove multiple spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    return cleaned.strip()

def is_crisis_response_needed(user_input: str, history: list) -> bool:
    """Check if the conversation context suggests a crisis response is needed."""
    # Check current user input
    if check_crisis_trigger(user_input):
        return True
    
    # Check recent history for crisis context
    if history:
        recent_messages = history[-4:]  # Check last 4 messages
        for msg in recent_messages:
            if msg.get("role") == "user":
                if check_crisis_trigger(msg.get("content", "")):
                    return True
    
    return False

def sanitize_bot_response(response: str, was_crisis: bool) -> str:
    """Clean the bot response to ensure no contact details unless it's a crisis response."""
    if was_crisis:
        # If it's a crisis response, ensure it contains exactly the CRISIS_RESPONSE
        # or at least the approved numbers
        return CRISIS_RESPONSE
    
    # For non-crisis responses, aggressively remove any contact details
    cleaned = remove_contact_details(response)
    
    # If after cleaning the response is empty or too short, provide a fallback
    if len(cleaned.strip()) < 10:
        cleaned = "I understand you're asking for resources. While I can't provide specific numbers, I encourage you to search online for mental health support services in Malaysia or visit the MENTARI portal for official resources."
    
    return cleaned

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    history: list[ChatMessage]
    user_input: str

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html not found</h1>"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # Convert history to dict format for easier checking
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
    
    # Step 1: Check if crisis response is needed based on user input and history
    is_crisis = is_crisis_response_needed(request.user_input, history_dicts)
    
    if is_crisis:
        return {"response": CRISIS_RESPONSE, "flagged_crisis": True}

    # Step 2: Build prompt with enhanced instructions
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.user_input})

    # Step 3: Call model
    try:
        completion = client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
            temperature=0.6,
            max_tokens=400,
            extra_headers={
                "HTTP-Referer": "https://supportive-ai.onrender.com",
                "X-Title": "SafeSpace",
            }
        )

        bot_response = completion.choices[0].message.content or ""
        
        # Step 4: Sanitize the response to ensure no contact details
        bot_response = sanitize_bot_response(bot_response, was_crisis=False)

        return {"response": bot_response, "flagged_crisis": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")

app.mount("/", StaticFiles(directory=".", html=False), name="static")

if __name__ == "__main__": 
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)