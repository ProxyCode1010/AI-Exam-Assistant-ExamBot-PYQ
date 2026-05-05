# AI Exam Assistant
### RAG + LLM pipeline for exam question prediction

---

## How It Works

```
questions.txt  →  [Embed]  →  FAISS Vector DB
                                    ↓
User enters topic  →  [Retrieve top-K]  →  [Groq Llama 3.3]  →  Ranked + Predicted Qs
```

- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, cosine similarity)
- **Vector DB**: FAISS IndexFlatIP (exact inner-product search)
- **LLM**: `llama-3.3-70b-versatile` via Groq API
- **Backend**: FastAPI
- **Frontend**: Vanilla HTML/CSS/JS (served by FastAPI)

---

## Folder Structure

```
ai-exam-assistant/
│
├── data/
│   ├── questions.txt           ← your raw PYQ file (edit this)
│   ├── questions_clean.json    ← auto-generated after Step 1
│   ├── loader.py               ← Step 1: clean + dedup
│   └── __init__.py
│
├── embeddings/
│   ├── builder.py              ← Step 2: embed + build FAISS index
│   ├── faiss.index             ← auto-generated after Step 2
│   ├── metadata.pkl            ← auto-generated after Step 2
│   └── __init__.py
│
├── retrieval/
│   ├── engine.py               ← Step 3: retrieval engine (used by API)
│   └── __init__.py
│
├── llm/
│   ├── generator.py            ← Step 4: Groq LLM call
│   └── __init__.py
│
├── api/
│   ├── server.py               ← Step 5: FastAPI server
│   └── __init__.py
│
├── frontend/
│   └── static/
│       └── index.html          ← Full UI (dark theme)
│
├── .env                        ← YOUR API KEY goes here
├── requirements.txt
├── setup.py                    ← One-time setup runner
└── README.md
```

---

## Sequential Execution Guide

### PRE-REQUISITE: Get Groq API Key
1. Go to https://console.groq.com
2. Sign up (free) → Create API Key
3. Open `.env` and replace `your_groq_api_key_here` with your key

---

### STEP 0 — Create virtual environment

```bash
# Navigate to project folder
cd ai-exam-assistant

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

---

### STEP 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> First time installs ~1.5GB (PyTorch + sentence-transformers). Takes 2–5 min.

---

### STEP 2 — Run one-time setup (loads data + builds index)

```bash
python setup.py
```

This runs two sub-steps automatically:
- **2a** `data/loader.py` — reads `questions.txt`, deduplicates, saves `questions_clean.json`
- **2b** `embeddings/builder.py` — embeds all questions using MiniLM, saves `faiss.index` + `metadata.pkl`

Expected output:
```
[loader] Raw lines     : 72
[loader] After dedup   : 45
[loader] Saved 45 questions → data/questions_clean.json
[builder] Loading embedding model: all-MiniLM-L6-v2
[builder] Encoding questions...
[builder] FAISS index total vectors: 45
[builder] Saved index  → embeddings/faiss.index
```

> You only need to run `setup.py` once, or again if you add new questions to `questions.txt`.

---

### STEP 3 — (Optional) Test retrieval standalone

```bash
python -m retrieval.engine
```

Expected output for topic "Hibernate":
```
Top 8 PYQs for topic: 'Hibernate'
   1. [0.872] Explain the Architecture of Hibernate in detail
   2. [0.831] What is Hibernate
   ...
```

---

### STEP 4 — (Optional) Test LLM standalone

```bash
python -m llm.generator
```

Calls Groq API and prints the ranked + predicted questions for "Servlet".

---

### STEP 5 — Start the server

```bash
uvicorn api.server:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

---

### STEP 6 — Open the UI

Open your browser and go to:
```
http://localhost:8000
```

Type any topic (Servlet, Hibernate, Spring, etc.) → click **Generate ↗**

---

### STEP 7 — (Optional) Test the API directly

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"topic": "Servlet", "top_k": 8}'
```

---

## Adding New Questions

1. Add questions to `data/questions.txt` (one per line)
2. Re-run: `python setup.py`
3. Restart the server

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY not set` | Edit `.env` and add your key |
| `faiss.index not found` | Run `python setup.py` first |
| Slow first startup | MiniLM downloads ~90MB on first use |
| Port 8000 in use | Use `--port 8001` in uvicorn command |
| CUDA errors | The project uses CPU-only FAISS, safe to ignore GPU warnings |