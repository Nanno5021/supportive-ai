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

CRITICAL CONSTRAINTS:
1. You are NOT a doctor, therapist, counsellor, or medical professional.
2. Never provide a diagnosis, psychological assessment, or medication advice.
3. Do not shame, lecture, or overwhelm the user.
4. Keep responses concise, usually 2–4 short paragraphs max.
5. Do not include any phone numbers, hotline numbers, website links, URLs, or contact details in normal responses.
6. Only the exact CRISIS_RESPONSE may include approved emergency contact details.
7. When appropriate, encourage reaching out to trusted people or local professional support without giving any numbers or links.
8. Do not use US-centric crisis references unless specifically relevant. Prefer Malaysia context.
9. If the user may be in crisis, respond with support, encourage immediate human help, and avoid excessive detail.
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
]

CRISIS_RESPONSE = (
    "I’m really sorry you’re carrying this much pain right now. "
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

def check_crisis_trigger(text: str) -> bool:
    """Scan input against crisis patterns."""
    for pattern in CRISIS_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def remove_contact_details(text: str) -> str:
    """Remove phone numbers and URLs from non-crisis responses."""
    cleaned = CONTACT_PATTERN.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

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
    # Step 1: Crisis check BEFORE model call
    if check_crisis_trigger(request.user_input):
        return {"response": CRISIS_RESPONSE, "flagged_crisis": True}

    # Step 2: Build prompt
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
        bot_response = remove_contact_details(bot_response)

        return {"response": bot_response, "flagged_crisis": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")

# Serve static files from /static instead of /
# Put logo.jpg, styles.css, etc. in the same folder and reference them as needed.
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
