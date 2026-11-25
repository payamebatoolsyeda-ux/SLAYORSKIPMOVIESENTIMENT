"""Aspect-based analysis across 5 cinematic dimensions."""
import re
from typing import List

import pandas as pd

ASPECT_LABELS = {
    "acting": "Acting",
    "storyline_pacing": "Storyline & Pacing",
    "direction": "Direction",
    "cinematography": "Cinematography",
    "soundtrack_vfx": "Soundtrack & VFX",
}

ASPECT_KEYWORDS = {
    "acting": [
        "acting", "actor", "actors", "actress", "actresses", "cast", "performance",
        "performances", "portrayal", "portrayed", "played", "plays", "lead", "chemistry",
    ],
    "storyline_pacing": [
        "story", "storyline", "plot", "script", "screenplay", "writing", "dialogue",
        "narrative", "twist", "twists", "ending", "pacing", "pace", "characters",
        "character", "drag", "drags", "rushed", "slow burn",
    ],
    "direction": [
        "direction", "director", "directed", "directing", "tone", "vision",
        "filmmaker", "filmmaking", "editing", "helmed", "auteur",
    ],
    "cinematography": [
        "cinematography", "visuals", "visual", "camera", "shots", "shot",
        "lighting", "scenery", "photography", "frames", "framing", "palette",
    ],
    "soundtrack_vfx": [
        "score", "soundtrack", "music", "sound", "audio", "songs", "synth",
        "effects", "vfx", "cgi", "special effects", "practical effects", "orchestra",
    ],
}

_PATTERNS = {
    a: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b", re.IGNORECASE)
    for a, kws in ASPECT_KEYWORDS.items()
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text)) if len(s.strip()) > 3]


def extract_aspect_sentences(texts: List[str]) -> pd.DataFrame:
    rows = []
    for i, text in enumerate(texts):
        for sent in split_sentences(text):
            for aspect, pat in _PATTERNS.items():
                if pat.search(sent):
                    rows.append({"review_idx": i, "aspect": aspect, "sentence": sent})
    return pd.DataFrame(rows, columns=["review_idx", "aspect", "sentence"])


def analyze_aspects(texts: List[str], model) -> pd.DataFrame:
    """Adds a 'label' column (sentiment of each aspect sentence)."""
    df = extract_aspect_sentences(texts)
    if df.empty:
        df["label"] = []
        return df
    df["label"] = [p["label"] for p in model.predict(df["sentence"].tolist())]
    return df


def summarize_aspects(aspect_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["aspect", "aspect_name", "positive", "neutral", "negative", "mentions", "net_score"]
    rows = []
    for aspect in ASPECT_KEYWORDS:
        sub = aspect_df[aspect_df["aspect"] == aspect] if not aspect_df.empty else aspect_df
        pos = int((sub["label"] == "positive").sum()) if len(sub) else 0
        neu = int((sub["label"] == "neutral").sum()) if len(sub) else 0
        neg = int((sub["label"] == "negative").sum()) if len(sub) else 0
        n = pos + neu + neg
        rows.append({
            "aspect": aspect,
            "aspect_name": ASPECT_LABELS[aspect],
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "mentions": n,
            "net_score": round((pos - neg) / n, 2) if n else 0.0,
        })
    return pd.DataFrame(rows, columns=cols)
