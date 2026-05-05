"""
STEP 5: api/server.py
FastAPI backend — exposes /ask and /answer endpoints consumed by the frontend.
Run: uvicorn api.server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

from retrieval.engine import get_engine
from llm.generator import generate_questions

app = FastAPI(title="AI Exam Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


# ── Schemas ─────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200, examples=["Servlet"])
    top_k: int = Field(default=8, ge=1, le=20)

class RetrievedQ(BaseModel):
    question: str
    score: float

class AskResponse(BaseModel):
    topic: str
    retrieved: list[RetrievedQ]
    llm_output: str
    total_in_db: int

class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

class AnswerResponse(BaseModel):
    question: str
    answer: str


# ── Endpoints ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
def serve_ui():
    return FileResponse("frontend/static/index.html")


@app.get("/health")
def health():
    engine = get_engine()
    return {"status": "ok", "questions_in_db": len(engine.questions)}


@app.post("/answer", response_model=AnswerResponse)
def answer_question(req: AnswerRequest):
    """Generate a detailed exam answer for a single retrieved PYQ."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are an expert Java/Web Technologies exam answer writer.

Write a detailed, exam-ready answer for this question:

{req.question}

Format your answer with:
- Clear headings where needed (use ### for headings)
- Numbered points or bullet points for lists
- Code examples if relevant (wrap in triple backticks with language)
- Keep it concise but complete as expected in a university exam answer
- Aim for 250-400 words

Do NOT repeat the question. Start directly with the answer."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise exam answer writer for Java/Web Technologies. Write structured answers suitable for university exams.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=900,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    return AnswerResponse(question=req.question, answer=answer)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    engine = get_engine()
    retrieved = engine.retrieve(req.topic.strip(), top_k=req.top_k)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No similar questions found.")

    try:
        llm_output = generate_questions(req.topic.strip(), retrieved)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    return AskResponse(
        topic=req.topic,
        retrieved=[RetrievedQ(question=r.question, score=r.score) for r in retrieved],
        llm_output=llm_output,
        total_in_db=len(engine.questions),
    )


































































































































