"""
STEP 1: data/loader.py
Loads questions.txt, deduplicates, and saves cleaned list as JSON.
Run: python -m data.loader
"""

import json
import os


def load_and_clean(filepath: str = "data/questions.txt") -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    seen = set()
    cleaned = []
    for line in raw_lines:
        q = line.strip()
        if not q:
            continue
        key = q.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(q)

    print(f"[loader] Raw lines     : {len(raw_lines)}")
    print(f"[loader] After dedup   : {len(cleaned)}")
    return cleaned


def save_cleaned(questions: list[str], out_path: str = "data/questions_clean.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"[loader] Saved {len(questions)} questions → {out_path}")


if __name__ == "__main__":
    qs = load_and_clean()
    save_cleaned(qs)