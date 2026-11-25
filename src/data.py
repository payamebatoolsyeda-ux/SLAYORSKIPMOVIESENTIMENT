"""Data loading for SST-5 training and flexible CSV/JSON user review ingestion."""
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DATA_DIR, ID2LABEL
from .preprocess import clean_text


def five_to_three(label: int) -> int:
    # SST-5: 0 very neg, 1 neg, 2 neutral, 3 pos, 4 very pos
    if label <= 1:
        return 0
    if label == 2:
        return 1
    return 2


def load_sst5_3class(use_cache: bool = True) -> dict:
    """Returns {'train','validation','test'} DataFrames with columns text, label, label_name."""
    DATA_DIR.mkdir(exist_ok=True)
    splits = ["train", "validation", "test"]
    cache = {s: DATA_DIR / f"sst5_3class_{s}.csv" for s in splits}
    if use_cache and all(p.exists() for p in cache.values()):
        return {s: pd.read_csv(p) for s, p in cache.items()}

    from datasets import load_dataset

    ds = load_dataset("SetFit/sst5")
    out = {}
    for s in splits:
        df = ds[s].to_pandas()[["text", "label"]]
        df["text"] = df["text"].map(clean_text)
        df["label"] = df["label"].map(five_to_three)
        df["label_name"] = df["label"].map(ID2LABEL)
        df.to_csv(cache[s], index=False)
        out[s] = df
    return out


def _normalize_reviews_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).lower(): c for c in df.columns}
    if "text" not in cols:
        for alt in ("review", "content", "comment", "body"):
            if alt in cols:
                df = df.rename(columns={cols[alt]: "text"})
                break
        else:
            raise ValueError("File needs a 'text' (or 'review') column.")
    else:
        df = df.rename(columns={cols["text"]: "text"})

    for optional, alts in (
        ("movie", ("movie", "title", "film")),
        ("genre", ("genre", "category")),
        ("date", ("date", "timestamp", "created_at")),
    ):
        for alt in alts:
            if alt in cols:
                df = df.rename(columns={cols[alt]: optional})
                break

    df["text"] = df["text"].astype(str).map(clean_text)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


def load_reviews_csv(path_or_buffer) -> pd.DataFrame:
    """User-supplied reviews CSV. Needs a 'text' column; optional 'movie' and 'date' columns."""
    return _normalize_reviews_df(pd.read_csv(path_or_buffer))


def load_reviews_json(path_or_buffer) -> pd.DataFrame:
    """User-supplied reviews JSON (array of objects, object with 'reviews' key, or JSON lines)."""
    if isinstance(path_or_buffer, (str, Path)):
        raw = Path(path_or_buffer).read_text(encoding="utf-8")
    else:
        raw_bytes = path_or_buffer.read()
        raw = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)

    payload = json.loads(raw)
    if isinstance(payload, dict):
        for key in ("reviews", "data", "items"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("JSON must be a list of review objects or an object containing 'reviews'.")
    return _normalize_reviews_df(pd.DataFrame(payload))


def load_reviews_file(path_or_buffer, filename: Optional[str] = None) -> pd.DataFrame:
    """Load CSV or JSON reviews into a normalized DataFrame."""
    name = (filename or getattr(path_or_buffer, "name", "") or str(path_or_buffer)).lower()
    if name.endswith(".json"):
        return load_reviews_json(path_or_buffer)
    return load_reviews_csv(path_or_buffer)
