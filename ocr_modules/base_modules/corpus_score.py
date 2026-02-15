# ocr_modules/base_modules/corpus_score.py
import json
import os
import re
from shared.path_utils import project_path

CORPUS_PATH = project_path("resources", "corpus_freqs.json")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    raw_freqs = json.load(f)

# Normalize keys to lowercase
CORPUS_FREQS = {
    word.lower(): float(freq) if isinstance(freq, (int, float)) else 0.0
    for word, freq in raw_freqs.items()
}

# Build percentile rank map
sorted_words = sorted(CORPUS_FREQS.items(), key=lambda x: x[1], reverse=True)
RANK_PERCENTILE = {
    word: round(1 - (i / len(sorted_words)), 4)
    for i, (word, _) in enumerate(sorted_words)
}
for word, score in RANK_PERCENTILE.items():
    if not isinstance(score, (int, float)):
        print(f"⚠️ Non-numeric score in RANK_PERCENTILE: {word} → {score}")

UNKNOWN_TOKENS = set()

def tokenize(text):
    # Lowercase and extract alphabetic words only
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())

def score_word(word):
    score = RANK_PERCENTILE.get(word)
    if score is None:
        UNKNOWN_TOKENS.add(word)
        return 0.0
    return score

def corpus_score(text, verbose=False):
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    scores = [score_word(token) for token in tokens]
    if verbose:
        print(f"\n🔍 Corpus Scoring: {text}")
        for token, score in zip(tokens, scores):
            print(f"  {token}: {score}")
    return round(sum(scores) / len(scores), 2)
