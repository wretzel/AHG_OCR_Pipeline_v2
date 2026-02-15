# voice/punctuation.py
QUESTION_STARTERS = {
    "who", "what", "when", "where", "why", "how",
    "do", "does", "did",
    "are", "is", "can", "should", "would", "will"
}

EXCLAMATION_WORDS = {
    "wow", "oh", "oh my god", "no way", "stop", "hey", "look out"
}


def infer_punctuation(text: str) -> str:
    """
    Infer punctuation based on heuristics.
    """

    if not text:
        return ""

    lower = text.lower()

    # Question detection
    first_word = lower.split(" ", 1)[0]
    if first_word in QUESTION_STARTERS:
        return text + "?"

    # Exclamation detection
    for phrase in EXCLAMATION_WORDS:
        if phrase in lower:
            return text + "!"

    # Default
    return text + "."
