# ocr_modules/base_modules/parsers.py

from ocr_modules.base_modules.reliability import is_reliable
from ocr_modules.base_modules.corpus_score import corpus_score
import numpy as np

def filter_cipher_output(text: str) -> bool:
    """
    Heuristic filter to reject gibberish/cipher-like outputs.
    Returns True if text looks valid, False if it's likely junk.
    """
    import re
    if not text:
        return False

    # Ratio of alphabetic characters to total
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)

    if alpha_ratio < 0.5:
        return False
    if re.search(r"[=]{2,}", text):  # repeated '='
        return False
    if re.search(r"[A-Z][a-z][A-Z][a-z]", text):  # alternating case pattern
        return False
    if len(text) < 3:
        return False

    return True


def parse_tesseract_output(raw):
    """
    Parse raw Tesseract output into normalized dict.
    raw: dict with keys "text" (list of strings) and "conf" (list of confidences).
    """
    lines, confidences, alpha_count = [], [], 0
    for text, conf in zip(raw.get("text", []), raw.get("conf", [])):
        if text.strip() and conf != "-1":
            lines.append(text)
            try:
                conf_val = float(conf)
            except ValueError:
                conf_val = 0.0
            confidences.append(conf_val)
            alpha_count += sum(c.isalpha() for c in text)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    scaled_conf = avg_conf / 100.0
    text = " ".join(lines).strip()
    corpus = corpus_score(text)

    passes_filter = filter_cipher_output(text)

    return {
        "text": text,
        "confidence": round(scaled_conf, 2),
        "corpus_score": corpus,
        "reliable": passes_filter and scaled_conf >= 0.6 and corpus >= 0.5
    }

def parse_easyocr_output(raw, min_token_conf=0.6):
    lines, confidences = [], []
    kept, dropped = [], []

    for entry in raw:
        if len(entry) >= 3:
            text = entry[1].strip()
            conf = float(entry[2])
            if text:
                if conf >= min_token_conf:
                    lines.append(text)
                    confidences.append(conf)
                    kept.append((text, conf))
                else:
                    dropped.append((text, conf))

    # print(f"🔎 EasyOCR tokens kept: {kept}")
    # print(f"⚠️ EasyOCR tokens dropped: {dropped}")

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    text = " ".join(lines).strip()
    corpus = corpus_score(text)

    return {
        "text": text,
        "confidence": round(avg_conf, 2),
        "corpus_score": corpus,
        "reliable": avg_conf >= 0.6 and corpus >= 0.5
    }

def parse_paddleocr_output(raw):
    """
    Parse PaddleOCR output into normalized dict.
    raw: list of dicts returned by reader.ocr(), each with 'rec_texts' and 'rec_scores'.
    """
    lines, confidences = [], []
    for res in raw:
        if isinstance(res, dict):
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            for txt, conf in zip(texts, scores):
                if txt and txt.strip():
                    lines.append(txt.strip())
                    confidences.append(conf)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    text = " ".join(lines).strip()

    # optional corpus scoring if you have a function defined
    try:
        corpus = corpus_score(text)
    except Exception:
        corpus = 0.0

    return {
        "text": text,
        "confidence": round(avg_conf, 2),
        "corpus_score": corpus,
        "reliable": avg_conf >= 0.6 and corpus >= 0.5
    }


def parse_east_output(raw):
    """
    Parse EAST detector output into normalized dict.
    raw: dict with "regions" and "region_count".
    """
    regions = raw.get("regions", [])
    region_count = raw.get("region_count", 0)

    return {
        "regions": regions,
        "region_count": region_count,
        "reliable": region_count > 0
    }
