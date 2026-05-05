#!/usr/bin/env python3
"""
setup.py — One-time setup script.
Runs Step 1 (load) and Step 2 (build embeddings) in sequence.
After this, start the server with:  uvicorn api.server:app --reload --port 8000
"""

import sys
import os

print("=" * 55)
print("  AI Exam Assistant — Setup")
print("=" * 55)

# ── Check .env ──────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
    print("\n[!] ERROR: Set your GROQ_API_KEY in .env first!")
    print("    Get it free at: https://console.groq.com\n")
    sys.exit(1)

# ── Step 1: Load & clean questions ──────────────────────────
print("\n[1/2] Loading and deduplicating questions…")
from data.loader import load_and_clean, save_cleaned
questions = load_and_clean()
save_cleaned(questions)

# ── Step 2: Build FAISS index ───────────────────────────────
print("\n[2/2] Building embeddings and FAISS index…")
from embeddings.builder import build_index
build_index()

print("\n" + "=" * 55)
print("  Setup complete!")
print("  Start the server:")
print("  uvicorn api.server:app --reload --port 8000")
print("  Then open: http://localhost:8000")
print("=" * 55 + "\n")