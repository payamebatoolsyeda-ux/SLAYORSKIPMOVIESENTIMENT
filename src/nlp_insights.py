"""Innovative NLP signals: Sarcasm & Irony Radar, Auto-Spoiler Shield, Emotion Spectrum, Popcorn Meter."""
import re
from typing import Dict, List, Tuple

from .aspects import split_sentences

# ------------------------------------------------------- Sarcasm & Irony Radar
_SARCASM_PATTERNS = [
    (re.compile(r"\b(masterpiece|brilliant|genius|10/10|best)\b.{0,45}\b(insomnia|nap|sleep|torture|paint dry|time back|regret)\b", re.I),
     "Praises film as a sleep aid / time-waster"),
    (re.compile(r"\bif you('re| are)? (trying|looking|hoping) to (cure insomnia|fall asleep|waste)\b", re.I),
     "Backhanded conditional praise"),
    (re.compile(r"\b(would|will) (nap|sleep|snooze) again\b", re.I),
     "Ironic rating pattern"),
    (re.compile(r"\b(amazingly|brilliantly|spectacularly|wonderfully)\s+(bad|awful|terrible|boring|dull|horrible|lifeless)\b", re.I),
     "Oxymoronic positive-negative contrast"),
    (re.compile(r"\b(yeah|sure|totally),?\s+(because|as if)\b.{0,40}\b(makes sense|believ|genius|oscar)\b", re.I),
     "Sarcastic concession phrasing"),
    (re.compile(r"\bcongratulations\b.{0,40}\b(making|turning|wasting|ruining)\b", re.I),
     "Ironic congratulations"),
    (re.compile(r'"(?:masterpiece|classic|brilliant|genius|deep|thrilling)"', re.I),
     "Scare-quoted praise word"),
]


def detect_sarcasm(text: str) -> dict:
    """Detect sarcastic or ironic review phrasing and explain the trigger."""
    s = str(text)
    reasons = []
    for pat, reason in _SARCASM_PATTERNS:
        if pat.search(s):
            reasons.append(reason)
    score = min(1.0, len(reasons) * 0.45 + (0.15 if reasons and ("!" in s or "..." in s) else 0.0))
    return {
        "is_sarcastic": bool(reasons),
        "score": round(score, 2),
        "reason": "; ".join(reasons) if reasons else "",
    }


def adjust_sentiment_for_sarcasm(label: str, confidence: float, sarcasm: dict) -> Tuple[str, float]:
    """Invert apparent positive/neutral polarity when strong irony/sarcasm is detected."""
    if not sarcasm.get("is_sarcastic"):
        return label, confidence
    base_conf = 0.7 if confidence is None else float(confidence)
    if label == "positive":
        return "negative", round(max(0.55, min(0.95, 0.55 + 0.35 * sarcasm["score"])), 3)
    if label == "neutral":
        return "negative", round(max(0.52, min(0.85, base_conf)), 3)
    return label, round(min(0.98, max(base_conf, 0.75)), 3)


# ---------------------------------------------------------- Auto-Spoiler Shield
_SPOILER_PATTERNS = [
    (re.compile(r"\bspoiler\b", re.I), "explicit spoiler warning"),
    (re.compile(r"\b(dies|died|killed|murdered|sacrifices? (?:himself|herself|themselves))\b", re.I), "character death"),
    (re.compile(r"\b(turns out|reveals? that|revealed to be|was actually|plot twist|final twist)\b", re.I), "plot twist reveal"),
    (re.compile(r"\b(in the end|at the end|final scene|post[- ]credits?|ending where)\b", re.I), "ending detail"),
    (re.compile(r"\b(betrays?|betrayed|culprit|killer was|survived)\b", re.I), "major plot outcome"),
]


def detect_spoiler(text: str) -> dict:
    """Identify sentences likely containing plot spoilers."""
    triggers: List[str] = []
    spoiler_sentences: List[str] = []
    for sent in split_sentences(text):
        sent_hits = [label for pat, label in _SPOILER_PATTERNS if pat.search(sent)]
        if sent_hits:
            spoiler_sentences.append(sent)
            for h in sent_hits:
                if h not in triggers:
                    triggers.append(h)
    return {
        "has_spoiler": bool(spoiler_sentences),
        "triggers": triggers,
        "spoiler_sentences": spoiler_sentences,
    }


# ----------------------------------------------------- Emotion & Vibe Spectrum
EMOTION_LEXICON: Dict[str, List[str]] = {
    "Hype": [
        "triumph", "masterpiece", "best", "thrilling", "electrifying", "jaw-dropping",
        "exhilarating", "phenomenal", "must-see", "recommend", "unforgettable", "brilliant",
        "dazzling", "electric", "kinetic", "spectacular",
    ],
    "Nostalgia": [
        "classic", "nostalgic", "retro", "timeless", "vintage", "memories", "childhood",
        "throwback", "warm", "golden", "echoes", "old-school", "reminiscent", "summer",
    ],
    "Suspense": [
        "suspense", "tense", "tension", "mystery", "haunting", "twist", "twists",
        "edge", "eerie", "chilling", "unpredictable", "paranoia", "thriller", "dark",
    ],
    "Emotional Depth": [
        "emotional", "moving", "cried", "tears", "heartbreaking", "touching", "poignant",
        "tender", "soulful", "human", "believable", "intimate", "grief", "chemistry",
    ],
    "Humor": [
        "funny", "hilarious", "witty", "clever", "comedy", "laugh", "laughed", "charming",
        "playful", "sharp", "banter", "sarcastic", "amusing",
    ],
    "Disappointment": [
        "disappointing", "letdown", "boring", "sloppy", "rushed", "thin", "weak",
        "wooden", "forgettable", "predictable", "terrible", "waste", "drags", "mess",
        "unfocused", "cheap",
    ],
}

_EMOTION_PATTERNS = {
    emo: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    for emo, words in EMOTION_LEXICON.items()
}


def score_emotions(texts: List[str]) -> Dict[str, float]:
    """Map reviews onto a 6-axis Emotion & Vibe Spectrum (0-100 scale)."""
    if not texts:
        return {emo: 0.0 for emo in EMOTION_LEXICON}
    raw_hits = {emo: 0 for emo in EMOTION_LEXICON}
    for t in texts:
        s = str(t)
        for emo, pat in _EMOTION_PATTERNS.items():
            raw_hits[emo] += len(pat.findall(s))

    n = max(1, len(texts))
    # Scale hits per review so typical rich sets span 15..95 cleanly without saturating at 100
    max_density = max(0.35, max(raw_hits.values()) / n)
    return {
        emo: round(min(100.0, (hits / n) / max_density * 88.0 + (8.0 if hits > 0 else 0.0)), 1)
        for emo, hits in raw_hits.items()
    }


# --------------------------------------------------------------- Popcorn Meter
def compute_popcorn_meter(pct: Dict[str, float]) -> dict:
    """Compute Popcorn Meter score (0-100) and theater-themed verdict tier."""
    score = round(pct.get("positive", 0.0) + 0.45 * pct.get("neutral", 0.0), 1)
    score = max(0.0, min(100.0, score))
    if score >= 68.0:
        return {
            "score": score,
            "tier": "Freshly Popped",
            "stamp": "FRESHLY POPPED",
            "color": "#FACC15",
            "subtitle": "Golden theater triumph — audiences came back for seconds.",
        }
    if score >= 45.0:
        return {
            "score": score,
            "tier": "Half-Popped",
            "stamp": "HALF-POPPED",
            "color": "#38BDF8",
            "subtitle": "Divided matinee — bright highlights mixed with uneven kernels.",
        }
    return {
        "score": score,
        "tier": "Burnt Kernels",
        "stamp": "BURNT KERNELS",
        "color": "#F43F5E",
        "subtitle": "Tough crowd — audience reception scorched the box office.",
    }
