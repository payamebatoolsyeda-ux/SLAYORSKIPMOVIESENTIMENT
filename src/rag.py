"""Vector store over reviews and aspect sentences (SentenceTransformers + FAISS with TF-IDF FAISS fallback)."""
from typing import List, Optional

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import EMBED_MODEL

_encoder = None
_encoder_failed = False


def get_encoder():
    global _encoder, _encoder_failed
    if _encoder_failed:
        return None
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer

            _encoder = SentenceTransformer(EMBED_MODEL)
        except Exception:
            _encoder_failed = True
            return None
    return _encoder


class ReviewIndex:
    """Each document: {'text', 'label', 'kind': 'review'|'aspect_sentence', 'aspect': str|None}"""

    def __init__(self):
        self.docs: List[dict] = []
        self.index: Optional[faiss.Index] = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self.backend_name: str = "TF-IDF + FAISS"

    def _encode_docs(self, texts: List[str]) -> np.ndarray:
        enc = get_encoder()
        if enc is not None:
            self.backend_name = "SentenceTransformers + FAISS"
            vecs = enc.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vecs.astype("float32")

        self.backend_name = "TF-IDF + FAISS"
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        mat = self._vectorizer.fit_transform(texts)
        return normalize(mat, norm="l2").toarray().astype("float32")

    def _encode_query(self, query: str) -> np.ndarray:
        enc = get_encoder()
        if enc is not None and self._vectorizer is None:
            vecs = enc.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vecs.astype("float32")

        mat = self._vectorizer.transform([query])
        return normalize(mat, norm="l2").toarray().astype("float32")

    def build(self, docs: List[dict]):
        self.docs = docs
        self._vectorizer = None
        if not docs:
            self.index = None
            return
        vecs = self._encode_docs([d["text"] for d in docs])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(
        self,
        query: str,
        k: int = 3,
        label: Optional[str] = None,
        aspect: Optional[str] = None,
        kind: Optional[str] = None,
        movie: Optional[str] = None,
        genre: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[dict]:
        if self.index is None:
            return []
        fetch = len(self.docs) if (movie or genre) else min(len(self.docs), max(k * 25, 50))
        scores, ids = self.index.search(self._encode_query(query), fetch)
        hits = []
        movie_low = movie.lower() if movie else None
        genre_low = genre.lower() if genre else None
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            if min_score is not None and float(score) < min_score:
                continue
            d = self.docs[i]
            if label and d["label"] != label:
                continue
            if aspect and d.get("aspect") != aspect:
                continue
            if kind and d["kind"] != kind:
                continue
            if movie_low and str(d.get("movie", "")).lower() != movie_low:
                continue
            if genre_low:
                doc_genre = str(d.get("genre", "")).lower()
                if genre_low != doc_genre and genre_low not in doc_genre:
                    continue
            hits.append({**d, "similarity": float(score)})
            if len(hits) == k:
                break
        return hits
