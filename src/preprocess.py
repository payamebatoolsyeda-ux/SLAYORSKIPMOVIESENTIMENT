"""Text preprocessing, tokenization, stopword filtering, and corpus statistics."""
import html
import re
from typing import Dict, List

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can",
    "could", "did", "do", "does", "doing", "done", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def clean_text(text: str) -> str:
    """Light cleaning: unescape HTML, drop tags, collapse whitespace."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_lemmatize(token: str) -> str:
    """Lightweight offline lemma-style normalization for analytics & vocabulary stats."""
    w = token.lower().strip("'")
    if len(w) <= 3:
        return w
    for suffix, repl in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)] + repl
    return w


def tokenize(text: str, remove_stopwords: bool = False, lemmatize: bool = False) -> List[str]:
    """Tokenize text with optional stopword removal and lightweight lemmatization."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(clean_text(text))]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if lemmatize:
        tokens = [simple_lemmatize(t) for t in tokens]
    return [t for t in tokens if t]


def compute_corpus_stats(texts: List[str]) -> Dict[str, float]:
    """Compute token count, stopword removal, and vocabulary stats across reviews."""
    raw_tokens: List[str] = []
    content_tokens: List[str] = []
    lemmas = set()
    for text in texts:
        toks = tokenize(text, remove_stopwords=False)
        raw_tokens.extend(toks)
        kept = [t for t in toks if t not in STOPWORDS]
        content_tokens.extend(kept)
        lemmas.update(simple_lemmatize(t) for t in kept)

    n_reviews = len(texts)
    total_tokens = len(raw_tokens)
    kept_tokens = len(content_tokens)
    removed = total_tokens - kept_tokens
    return {
        "n_reviews": n_reviews,
        "total_tokens": total_tokens,
        "content_tokens": kept_tokens,
        "stopwords_removed": removed,
        "stopword_pct": round(100.0 * removed / total_tokens, 1) if total_tokens else 0.0,
        "vocab_size": len(lemmas),
        "avg_tokens": round(total_tokens / n_reviews, 1) if n_reviews else 0.0,
    }
