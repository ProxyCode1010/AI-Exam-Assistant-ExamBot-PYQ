"""
STEP 4: llm/generator.py
Calls Groq (llama-3.3-70b-versatile) with retrieved PYQs as context.
Returns ranked + deduplicated + predicted exam questions.
Run: python -m llm.generator
"""

import os
from groq import Groq
from dotenv import load_dotenv
from retrieval.engine import RetrievedQuestion

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def build_prompt(topic: str, retrieved: list[RetrievedQuestion]) -> str:
    pyq_block = "\n".join(
        f"{i+1}. {r.question}  [similarity: {r.score:.3f}]"
        for i, r in enumerate(retrieved)
    )

    return f"""You are an expert exam question analyst for a Java/Web Technologies course.

A student is studying the topic: **{topic}**

Below are the most similar Past Year Questions (PYQs) retrieved from the question bank:

--- RETRIEVED PYQs ---
{pyq_block}
--- END PYQs ---

Your task (respond in this exact structure):

### Important PYQs (filtered & deduplicated)
List the most exam-relevant questions from the PYQs above. Remove duplicates, keep only the most complete/specific version. Number them.

### Predicted Questions (likely to appear)
Based on exam patterns in the PYQs, predict 3–5 new questions the student should prepare. These must NOT be copies of the PYQs above — generate original predictions based on topic gaps and trends.

### Quick Tips for "{topic}"
Give 3 bullet points on what aspects to focus on for this topic in exams (what types of questions appear most: definition, architecture, code, comparison, etc.).

Be concise and exam-focused. Do not explain your reasoning — just give the questions and tips.
"""


def generate_questions(topic: str, retrieved: list[RetrievedQuestion]) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = build_prompt(topic, retrieved)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise exam assistant. Always respond using the exact "
                    "section headers provided. Be concise and exam-focused."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,      # lower = more factual/consistent
        max_tokens=1024,
        top_p=0.9,
    )

    return response.choices[0].message.content


# ── Standalone test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from retrieval.engine import get_engine

    engine = get_engine()
    topic = "Servlet"
    retrieved = engine.retrieve(topic)

    print(f"[generator] Calling Groq ({GROQ_MODEL}) for topic: '{topic}'…\n")
    output = generate_questions(topic, retrieved)
    print(output)