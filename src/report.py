"""Analysis pipeline, RAG Q&A, cinema-styled charts, and Box Office Intelligence Dossier (PDF/DOCX)."""
import json
import os
import re
from typing import List, Optional
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aspects import ASPECT_KEYWORDS, ASPECT_LABELS, analyze_aspects, summarize_aspects
from .config import LABELS, REPORTS_DIR
from .nlp_insights import (adjust_sentiment_for_sarcasm, compute_popcorn_meter,
                           detect_sarcasm, detect_spoiler, score_emotions)
from .preprocess import clean_text, compute_corpus_stats
from .rag import ReviewIndex

ASPECT_QUERIES = {
    "acting": "the acting and performances of the cast",
    "storyline_pacing": "the story, plot, pacing and script",
    "direction": "the direction, editing and filmmaking",
    "cinematography": "the cinematography, framing, lighting and visuals",
    "soundtrack_vfx": "the soundtrack, score, audio and visual effects",
}


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "movie"


def _short(text: str, n: int = 280) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------- analysis
def analyze_movie(title: str, reviews: pd.DataFrame, model, adjust_sarcasm: bool = True) -> dict:
    """Analyze a movie's reviews with 3-class ML, sarcasm/spoiler flags, 5-aspect ABSA, emotions, and FAISS RAG."""
    df = reviews.copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("No reviews to analyze.")

    texts = df["text"].tolist()
    preds = model.predict(texts)
    raw_labels = [p["label"] for p in preds]
    raw_scores = [p["score"] for p in preds]

    sarcasm_info = [detect_sarcasm(t) for t in texts]
    spoiler_info = [detect_spoiler(t) for t in texts]

    final_labels, final_scores = [], []
    for lbl, sc, sarc in zip(raw_labels, raw_scores, sarcasm_info):
        if adjust_sarcasm:
            adj_lbl, adj_sc = adjust_sentiment_for_sarcasm(lbl, sc, sarc)
        else:
            adj_lbl, adj_sc = lbl, sc
        final_labels.append(adj_lbl)
        final_scores.append(adj_sc)

    df["raw_label"] = raw_labels
    df["label"] = final_labels
    df["confidence"] = final_scores
    df["is_sarcastic"] = [s["is_sarcastic"] for s in sarcasm_info]
    df["sarcasm_score"] = [s["score"] for s in sarcasm_info]
    df["sarcasm_reason"] = [s["reason"] for s in sarcasm_info]
    df["has_spoiler"] = [s["has_spoiler"] for s in spoiler_info]
    df["spoiler_triggers"] = [", ".join(s["triggers"]) for s in spoiler_info]
    df["sentiment_score"] = [{"positive": 1, "neutral": 0, "negative": -1}[l] for l in final_labels]

    aspect_df = analyze_aspects(texts, model)
    aspect_summary = summarize_aspects(aspect_df)
    emotions = score_emotions(texts)
    corpus_stats = compute_corpus_stats(texts)

    # RAG index: whole reviews + aspect sentences tagged with sentiment & spoiler metadata
    docs = [
        {
            "text": r.text,
            "label": r.label,
            "kind": "review",
            "aspect": None,
            "has_spoiler": bool(r.has_spoiler),
            "is_sarcastic": bool(r.is_sarcastic),
        }
        for r in df.itertuples()
    ]
    docs += [
        {
            "text": r.sentence,
            "label": r.label,
            "kind": "aspect_sentence",
            "aspect": r.aspect,
            "has_spoiler": detect_spoiler(r.sentence)["has_spoiler"],
            "is_sarcastic": detect_sarcasm(r.sentence)["is_sarcastic"],
        }
        for r in aspect_df.itertuples()
    ]
    index = ReviewIndex()
    index.build(docs)

    evidence = {
        "positive": index.search("a great movie that I highly recommend", k=3, label="positive", kind="review"),
        "negative": index.search("a disappointing movie, not worth watching", k=3, label="negative", kind="review"),
        "aspects": {},
    }
    for a in ASPECT_KEYWORDS:
        evidence["aspects"][a] = {
            "positive": index.search(ASPECT_QUERIES[a] + " was excellent", k=2, label="positive",
                                     aspect=a, kind="aspect_sentence"),
            "negative": index.search(ASPECT_QUERIES[a] + " was bad", k=2, label="negative",
                                     aspect=a, kind="aspect_sentence"),
        }

    counts = {l: int((df["label"] == l).sum()) for l in LABELS}
    n = len(df)
    pct = {l: 100.0 * c / n for l, c in counts.items()}
    popcorn = compute_popcorn_meter(pct)

    return {
        "title": title,
        "n_reviews": n,
        "counts": counts,
        "pct": pct,
        "popcorn": popcorn,
        "reviews": df,
        "aspect_df": aspect_df,
        "aspect_summary": aspect_summary,
        "emotions": emotions,
        "corpus_stats": corpus_stats,
        "sarcasm_count": int(df["is_sarcastic"].sum()),
        "spoiler_count": int(df["has_spoiler"].sum()),
        "evidence": evidence,
        "rag_index": index,
        "rag_backend": index.backend_name,
    }


# ------------------------------------------------------- RAG Q&A
_GREETING_RE = re.compile(
    r"^(?:hi+|hello+|hey+|hiya|yo|sup|howdy|good\s+(?:morning|afternoon|evening)|what'?s\s+up|greetings)\b[\s,!?.]*",
    re.I,
)
_PURE_SMALL_TALK = {
    "greeting": re.compile(
        r"^(?:hi+|hello+|hey+|hiya|yo|sup|howdy|good\s+(?:morning|afternoon|evening)|what'?s\s+up|greetings)[!.\s?]*$",
        re.I,
    ),
    "wellbeing": re.compile(
        r"^(?:how\s+(?:are\s+(?:you|u)|r\s+u|are\s+you\s+doing|is\s+it\s+going|'s\s+it\s+going)|you\s+good)[!.\s?]*$",
        re.I,
    ),
    "identity": re.compile(
        r"^(?:who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|how\s+do\s+you\s+work|help(?:\s+me)?)[!.\s?]*$",
        re.I,
    ),
    "thanks": re.compile(
        r"^(?:thanks?(?:\s+you|\s+a\s+lot|\s+so\s+much)?|thx|ty|cheers|appreciate\s+it|ok(?:ay)?|cool|nice|awesome|great|got\s+it|sounds\s+good|makes\s+sense)[!.\s]*$",
        re.I,
    ),
    "bye": re.compile(
        r"^(?:bye+|goodbye|see\s+(?:you|ya)|later|cya|have\s+a\s+(?:good|nice)\s+(?:day|one))[!.\s]*$",
        re.I,
    ),
}

_STOP_AND_GENERIC = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "has", "have", "had", "can", "could", "would", "should", "will", "shall", "may", "might",
    "i", "me", "my", "we", "our", "you", "your", "they", "them", "their", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "in", "on", "at", "to", "for", "of", "with", "from", "by", "about", "as", "into", "like", "through", "over",
    "and", "or", "but", "if", "so", "than", "too", "very", "just", "really", "actually", "any", "some", "all", "many",
    "there", "here", "more", "most", "much", "lot", "anything", "something", "nothing", "everything",
    "movie", "movies", "film", "films", "cinema", "review", "reviews", "reviewer", "reviewers",
    "audience", "audiences", "people", "person", "viewer", "viewers", "critic", "critics", "crowd", "everyone", "anyone",
    "think", "thought", "say", "said", "saying", "feel", "felt", "feeling", "feelings", "react", "reacted", "reaction", "reactions",
    "talk", "talked", "talking", "mention", "mentioned", "opinion", "opinions", "feedback", "comment", "comments",
    "good", "bad", "great", "best", "worst", "better", "worse", "well", "tell", "know", "give", "show", "get",
    "info", "information", "detail", "details", "document", "documents", "doc", "docs", "file", "files", "pdf", "csv", "json", "txt",
    "database", "db", "data", "available", "found", "find", "search", "check", "read", "load", "please", "hey", "hi", "hello",
}

_GENERIC_SUBJECTS = _STOP_AND_GENERIC | {
    "acting", "actor", "actors", "actress", "actresses", "cast", "performance", "performances", "lead", "leads", "chemistry",
    "story", "storyline", "plot", "script", "screenplay", "writing", "dialogue", "narrative", "twist", "twists",
    "ending", "beginning", "middle", "pacing", "pace", "character", "characters",
    "direction", "director", "directing", "tone", "vision", "editing", "filmmaking",
    "cinematography", "visual", "visuals", "camera", "shot", "shots", "lighting", "scenery", "frames", "framing",
    "soundtrack", "score", "music", "sound", "audio", "songs", "vfx", "cgi", "effects", "special effects",
    "spoiler", "spoilers", "sarcasm", "irony", "vibe", "mood", "emotion", "emotions", "tone",
    "overall", "summary", "overview", "verdict", "rating", "score", "popcorn", "popcorn meter", "strengths", "weaknesses",
}

_TARGET_PATTERNS = [
    re.compile(r"\b(?:reviews?|info(?:rmation)?|data|details?|documents?|docs?|files?)\s+(?:of|for|on|about|called|named)\s+(?:the\s+(?:movie|film|document|doc|file)\s+)?[\"']?([A-Za-z0-9][A-Za-z0-9\s:'\-]{1,42})[\"']?", re.I),
    re.compile(r"\b(?:tell\s+me\s+about|what\s+about|how\s+about|how\s+is|how\s+was|have\s+you\s+seen|do\s+you\s+have|should\s+i\s+watch)\s+(?:the\s+(?:movie|film|document|doc|file)\s+)?[\"']?([A-Za-z0-9][A-Za-z0-9\s:'\-]{1,42})[\"']?", re.I),
    re.compile(r"[\"']([A-Za-z0-9][A-Za-z0-9\s:'\-]{1,42})[\"']\s+(?:movie|film|reviews?|document|doc|file)\b", re.I),
    re.compile(r"\b([A-Z][A-Za-z0-9:'\-]+(?:\s+[A-Z0-9][A-Za-z0-9:'\-]+){0,4})\s+(?:movie|film|reviews?|document|doc|file)\b"),
]


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _distinctive_tokens(text: str) -> List[str]:
    return [t for t in _tokenize_words(text) if t not in _STOP_AND_GENERIC and len(t) > 1]


def _token_matches_corpus(tok: str, corpus_tokens: set) -> bool:
    if tok in corpus_tokens:
        return True
    if len(tok) >= 4:
        pref = tok[:4]
        return any(ct.startswith(pref) or tok.startswith(ct[:4]) for ct in corpus_tokens if len(ct) >= 4)
    return False


def _clean_subject(raw: str) -> str:
    s = re.sub(r"[?.!,;:\"']+$", "", raw.strip())
    s = re.sub(r"^(?:the|a|an|this|that|my|our)\s+", "", s, flags=re.I).strip()
    return s


def _classify_small_talk(question: str) -> Optional[str]:
    q = question.strip()
    for kind, pat in _PURE_SMALL_TALK.items():
        if pat.match(q):
            return kind
    return None


def _small_talk_response(kind: str, analysis: dict) -> str:
    title = analysis["title"]
    n = analysis["n_reviews"]
    if kind == "greeting":
        return (
            f"Hey there! I'm doing great — I have **{n} audience reviews** for **{title}** loaded up right here. "
            "Ask me anything about what viewers thought (the acting, ending, visuals, spoilers, or whether it's worth a watch)!"
        )
    if kind == "wellbeing":
        return (
            f"I'm doing well, thanks for asking! Always happy to talk movies. "
            f"Want to know the overall vibe on **{title}**, or are you curious about something specific?"
        )
    if kind == "identity":
        return (
            f"I'm **SlayOrSkip**, your conversational movie-review assistant. Right now I'm looking at **{n} reviews** for **{title}**. "
            "You can chat with me normally — ask for a quick verdict, dig into acting/story/music, check for spoilers, "
            "or ask specific questions. If you ask about a movie or document that isn't in the database, I'll let you know straight up."
        )
    if kind == "thanks":
        return f"Anytime! Let me know if you want to explore anything else about **{title}**."
    return f"Catch you later! Enjoy the show 🎬"


def _not_in_db_response(analysis: dict, subject: Optional[str] = None, available_movies: Optional[List[str]] = None) -> str:
    title = analysis["title"]
    other_list = [m for m in (available_movies or []) if m.lower() != title.lower()]
    if subject:
        msg = (
            f"**{subject}** was **not found in the database** (information not available). "
            f"Right now I only have audience reviews loaded for **{title}**."
        )
    else:
        msg = (
            f"That information is **not available in the database** for **{title}** — "
            "none of the loaded reviews or documents cover that topic."
        )
    if other_list:
        msg += f" Other movies currently in the curated database: {', '.join(f'**{m}**' for m in other_list)} (you can switch films in the sidebar, or upload your own CSV/JSON)."
    else:
        msg += " Try asking about the film's acting, storyline, ending, direction, visuals, or upload a dataset containing that movie/document."
    return msg


def _check_out_of_db_entity(question: str, analysis: dict, available_movies: Optional[List[str]]) -> Optional[dict]:
    """Detect if the user is asking about a different movie or an unindexed movie/document."""
    cur_title = analysis["title"]
    q_low = question.lower()

    # 1. Check if user specifically named another movie that IS in the database
    for m in available_movies or []:
        if m.lower() == cur_title.lower():
            continue
        if re.search(rf"\b{re.escape(m.lower())}\b", q_low):
            return {
                "answer": (
                    f"**{m}** is in our database, though I'm currently tuned into **{cur_title}**. "
                    f"Select **{m}** under **Select Film** in the sidebar and I'll gladly break down its reviews for you!"
                ),
                "citations": [],
                "generator": "Database Lookup",
            }

    corpus_tokens = set(_tokenize_words(" ".join(analysis["reviews"]["text"].tolist())))
    title_tokens = set(_tokenize_words(cur_title))

    # 2. Explicit movie/document target phrases
    for pat in _TARGET_PATTERNS:
        m = pat.search(question)
        if not m:
            continue
        raw_target = _clean_subject(m.group(1))
        if not raw_target or raw_target.lower() in _GENERIC_SUBJECTS:
            continue
        # Remove trailing generic words like "reviews" or "movie"
        target_words = [w for w in _tokenize_words(raw_target) if w not in _STOP_AND_GENERIC]
        if not target_words:
            continue
        if all(w in title_tokens for w in target_words):
            continue
        if not any(_token_matches_corpus(w, corpus_tokens) for w in target_words):
            display_name = " ".join(w.capitalize() if w.islower() else w for w in raw_target.split())
            return {
                "answer": _not_in_db_response(analysis, subject=display_name, available_movies=available_movies),
                "citations": [],
                "generator": "Database Lookup (Not Found)",
            }

    # 3. Mid-sentence capitalized proper nouns (e.g., "Is Inception worth watching?")
    words_raw = re.findall(r"[A-Za-z0-9:'\-]+", question)
    for idx, w in enumerate(words_raw):
        if idx == 0:
            continue
        w_low = w.lower()
        if w[0].isupper() and w_low not in _GENERIC_SUBJECTS and w_low not in title_tokens and w_low != "i":
            if not _token_matches_corpus(w_low, corpus_tokens):
                return {
                    "answer": _not_in_db_response(analysis, subject=w, available_movies=available_movies),
                    "citations": [],
                    "generator": "Database Lookup (Not Found)",
                }

    return None


def _detect_aspects_in_question(question: str) -> List[str]:
    q_low = question.lower()
    matched = []
    for aspect, kws in ASPECT_KEYWORDS.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", q_low):
                matched.append(aspect)
                break
    return matched


def _conversational_summary_reply(question: str, analysis: dict, mode: str) -> dict:
    title = analysis["title"]
    pct = analysis["pct"]
    pop = analysis["popcorn"]
    n = analysis["n_reviews"]
    ranked = analysis["aspect_summary"].sort_values("net_score", ascending=False)
    best = ranked[ranked["net_score"] > 0.15]["aspect_name"].tolist()
    weak = ranked[ranked["net_score"] < -0.15]["aspect_name"].tolist()

    if mode == "positive":
        hits = analysis["evidence"]["positive"][:3]
        praise_txt = f" Viewers especially praised **{', '.join(best)}**." if best else ""
        ans = (
            f"On the positive side, **{pct['positive']:.0f}%** of the {n} reviews for **{title}** are upbeat.{praise_txt}"
        )
        if hits:
            ans += f" One standout viewer summed it up nicely: \"{_short(hits[0]['text'], 180)}\""
        return {"answer": ans, "citations": hits, "generator": f"Conversational RAG ({analysis['rag_backend']})"}

    if mode == "negative":
        hits = analysis["evidence"]["negative"][:3]
        crit_txt = f" The main sore spots were **{', '.join(weak)}**." if weak else ""
        ans = (
            f"Roughly **{pct['negative']:.0f}%** of the {n} reviews for **{title}** lean negative.{crit_txt}"
        )
        if hits:
            ans += f" A typical critique reads: \"{_short(hits[0]['text'], 180)}\""
        return {"answer": ans, "citations": hits, "generator": f"Conversational RAG ({analysis['rag_backend']})"}

    if mode == "spoilers":
        sp_df = analysis["reviews"][analysis["reviews"]["has_spoiler"]]
        if sp_df.empty:
            return {
                "answer": f"Good news — none of the {n} reviews for **{title}** triggered the Auto-Spoiler Shield, so you should be safe from major plot reveals!",
                "citations": [],
                "generator": "Conversational NLP Insight",
            }
        cites = [{"text": r.text, "label": r.label, "has_spoiler": True, "similarity": 1.0} for r in sp_df.head(3).itertuples()]
        return {
            "answer": (
                f"Heads up! **{len(sp_df)}** of the {n} reviews for **{title}** contain potential plot spoilers "
                "(blurred below behind the Spoiler Shield if you'd rather not peek)."
            ),
            "citations": cites,
            "generator": "Conversational NLP Insight",
        }

    if mode == "sarcasm":
        sc_df = analysis["reviews"][analysis["reviews"]["is_sarcastic"]]
        if sc_df.empty:
            return {
                "answer": f"None of the {n} reviews for **{title}** were flagged by the Sarcasm & Irony Radar — viewers seem pretty straightforward here.",
                "citations": [],
                "generator": "Conversational NLP Insight",
            }
        cites = [{"text": r.text, "label": r.label, "has_spoiler": bool(r.has_spoiler), "similarity": 1.0} for r in sc_df.head(3).itertuples()]
        return {
            "answer": (
                f"The Sarcasm & Irony Radar caught **{len(sc_df)}** sarcastic review(s) for **{title}** — "
                f"for example: \"{_short(sc_df.iloc[0]['text'], 160)}\""
            ),
            "citations": cites,
            "generator": "Conversational NLP Insight",
        }

    # General overall / recommendation / verdict
    tone = _tone(pct)
    top_emo = sorted(analysis["emotions"].items(), key=lambda kv: kv[1], reverse=True)[0]
    parts = [
        f"Overall, **{title}** is getting a **{tone}** reception — it sits at **{pop['score']:.0f}% on the Popcorn Meter ({pop['tier']})** across {n} reviews ({pct['positive']:.0f}% positive, {pct['neutral']:.0f}% neutral, {pct['negative']:.0f}% negative)."
    ]
    if best:
        parts.append(f"Audiences really appreciated the **{', '.join(best[:2])}**")
        if weak:
            parts[-1] += f", though **{', '.join(weak[:2])}** drew some criticism."
        else:
            parts[-1] += "."
    elif weak:
        parts.append(f"The biggest complaints centered around **{', '.join(weak[:2])}**.")
    parts.append(f"The strongest emotional vibe is **{top_emo[0]}**.")
    if re.search(r"\b(watch|worth|recommend|should\s+i)\b", question, re.I):
        if pop["score"] >= 68:
            parts.append("If this genre sounds like your kind of film, it's definitely worth checking out!")
        elif pop["score"] >= 45:
            parts.append("It's a bit of a split crowd, so it depends on whether its strengths appeal to you.")
        else:
            parts.append("Most viewers came away disappointed, so you might want to temper expectations.")

    cites = (analysis["evidence"]["positive"][:2] + analysis["evidence"]["negative"][:1])[:3]
    return {
        "answer": " ".join(parts),
        "citations": cites,
        "generator": f"Conversational RAG ({analysis['rag_backend']})",
    }


def _conversational_aspect_reply(aspects: List[str], analysis: dict, index: ReviewIndex) -> dict:
    title = analysis["title"]
    summ = analysis["aspect_summary"].set_index("aspect")
    parts = []
    citations = []
    seen_texts = set()

    for asp in aspects[:2]:
        row = summ.loc[asp]
        asp_name = row["aspect_name"]
        mentions = int(row["mentions"])
        if mentions == 0:
            parts.append(f"None of the loaded reviews for **{title}** specifically mention **{asp_name}**, so that info isn't available in the database.")
            continue

        pos, neu, neg = int(row["positive"]), int(row["neutral"]), int(row["negative"])
        net = float(row["net_score"])
        if net >= 0.3:
            stance = f"audiences were really enthusiastic about the **{asp_name}** ({pos} positive vs. {neg} negative mentions)"
        elif net <= -0.3:
            stance = f"viewers were fairly critical of the **{asp_name}** ({neg} negative vs. {pos} positive mentions)"
        else:
            stance = f"opinions on the **{asp_name}** were divided ({pos} positive, {neu} neutral, {neg} negative)"
        parts.append(f"For **{asp_name}**, {stance}.")

        ev = analysis["evidence"]["aspects"].get(asp, {})
        asp_hits = (ev.get("positive", [])[:2] + ev.get("negative", [])[:2])
        if not asp_hits:
            min_sc = 0.25 if "SentenceTransformers" in index.backend_name else 0.05
            asp_hits = index.search(ASPECT_QUERIES[asp], k=2, kind="review", min_score=min_sc)
        for h in asp_hits:
            if h["text"] not in seen_texts:
                seen_texts.add(h["text"])
                citations.append(h)

    if citations:
        parts.append(f"For instance, one reviewer wrote: \"{_short(citations[0]['text'], 180)}\"")
        if len(citations) > 1 and citations[1]["label"] != citations[0]["label"]:
            parts.append(f"On the flip side, another noted: \"{_short(citations[1]['text'], 160)}\"")

    return {
        "answer": " ".join(parts),
        "citations": citations[:4],
        "generator": f"Conversational RAG ({analysis['rag_backend']})" if citations else "Database Lookup (Not Found)",
    }


def answer_audience_question(
    question: str,
    analysis: dict,
    use_llm: bool = False,
    available_movies: Optional[List[str]] = None,
    chat_history: Optional[List[dict]] = None,
) -> dict:
    """Conversational RAG Q&A grounded in FAISS-retrieved reviews, with natural chat & out-of-DB handling."""
    q_raw = (question or "").strip()
    if not q_raw:
        return {
            "answer": f"Ask me anything about **{analysis['title']}** — I'm ready whenever you are!",
            "citations": [],
            "generator": "Conversational Assistant",
        }

    # 1. Pure small talk (greetings, how are you, thanks, help, bye)
    st_kind = _classify_small_talk(q_raw)
    if st_kind:
        return {
            "answer": _small_talk_response(st_kind, analysis),
            "citations": [],
            "generator": "Conversational Assistant",
        }

    # Strip leading greeting if followed by a real question (e.g., "Hey, how was the acting?")
    greet_prefix = ""
    q_clean = q_raw
    gm = _GREETING_RE.match(q_raw)
    if gm and len(q_raw[gm.end():].strip()) > 1:
        greet_prefix = "Hey! "
        q_clean = q_raw[gm.end():].strip()

    # Handle simple follow-ups ("why?", "tell me more", "elaborate") using prior turn
    if re.match(r"^(?:why\??|tell\s+me\s+more|elaborate|explain(?:\s+more)?|more\s+details?)[?.!\s]*$", q_clean, re.I):
        if chat_history:
            prev = chat_history[-1]
            if prev.get("citations"):
                extra = " ".join(f"\"{ _short(c['text'], 160) }\"" for c in prev["citations"][:2])
                return {
                    "answer": f"{greet_prefix}Sure — here's a bit more detail straight from what viewers said: {extra}",
                    "citations": prev["citations"],
                    "generator": "Conversational Follow-up",
                }
        q_clean = "overall summary strengths and weaknesses"

    # 2. Check for another movie in DB or an unknown movie/document not in DB
    out_of_db = _check_out_of_db_entity(q_clean, analysis, available_movies)
    if out_of_db:
        return out_of_db

    # 3. Retrieve relevant reviews with similarity threshold
    index: ReviewIndex = analysis["rag_index"]
    is_dense = "SentenceTransformers" in index.backend_name
    min_score = 0.27 if is_dense else 0.08
    hits = index.search(q_clean, k=4, kind="review", min_score=min_score)
    if not hits:
        hits = index.search(q_clean, k=4, min_score=min_score)

    # Guard against TF-IDF matching only generic words ("movie", "reviews") on an unknown query
    q_distinct = _distinctive_tokens(q_clean)
    if not is_dense and hits and q_distinct:
        hits_tokens = set(_tokenize_words(" ".join(h["text"] for h in hits)))
        matched_in_hits = [t for t in q_distinct if _token_matches_corpus(t, hits_tokens)]
        if not matched_in_hits:
            hits = []

    # 4. Optional Anthropic LLM mode (conversational + strict DB grounding)
    if use_llm and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            excerpts = [{"label": h["label"], "similarity": round(h["similarity"], 3), "text": _short(h["text"], 320)} for h in hits]
            summary_ctx = {
                "active_movie": analysis["title"],
                "available_movies_in_database": available_movies or [analysis["title"]],
                "n_reviews": analysis["n_reviews"],
                "popcorn_meter": analysis["popcorn"],
                "sentiment_pct": {k: round(v, 1) for k, v in analysis["pct"].items()},
                "aspect_summary": analysis["aspect_summary"].to_dict("records"),
                "sarcasm_count": analysis["sarcasm_count"],
                "spoiler_count": analysis["spoiler_count"],
            }
            recent_turns = [
                {"user": item["q"], "assistant": item["answer"]}
                for item in (chat_history or [])[-4:]
            ]
            prompt = (
                f"You are SlayOrSkip, a friendly, natural-sounding movie review assistant chatting with a user about '{analysis['title']}'.\n"
                "Rules:\n"
                "1. Talk like a normal, helpful person — natural, warm, and conversational (not robotic).\n"
                "2. Base any claims strictly on the database context and retrieved review excerpts below. Do NOT invent plot details or outside facts.\n"
                "3. If the user asks about a different movie, document, or detail that is NOT in the active movie's reviews or database, "
                "clearly state that it is 'not found in the database' or 'information is not available' (and mention if another movie in available_movies_in_database can be selected in the sidebar).\n\n"
                f"DATABASE SUMMARY:\n{json.dumps(summary_ctx, indent=2)}\n\n"
                f"RECENT CHAT:\n{json.dumps(recent_turns, indent=2)}\n\n"
                f"RETRIEVED REVIEWS:\n{json.dumps(excerpts, indent=2)}\n\n"
                f"USER QUESTION: {q_raw}\n\n"
                "Write a helpful, natural 2-4 sentence reply."
            )
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return {
                "answer": msg.content[0].text.strip(),
                "citations": hits,
                "generator": f"LLM ({analysis['rag_backend']})",
            }
        except Exception as e:
            print(f"[rag_qa] LLM Q&A failed, falling back to conversational RAG: {e}")

    # 5. Deterministic conversational intents (spoilers, sarcasm, pros/cons, overall verdict, aspects)
    if re.search(r"\b(spoilers?|spoiler)\b", q_clean, re.I):
        res = _conversational_summary_reply(q_clean, analysis, "spoilers")
        res["answer"] = greet_prefix + res["answer"]
        return res
    if re.search(r"\b(sarcas\w*|irony|ironic)\b", q_clean, re.I):
        res = _conversational_summary_reply(q_clean, analysis, "sarcasm")
        res["answer"] = greet_prefix + res["answer"]
        return res
    if re.search(r"\b(what\s+did\s+people\s+like|best\s+part|strengths?|positive\s+reviews?|praise|highlights?)\b", q_clean, re.I):
        res = _conversational_summary_reply(q_clean, analysis, "positive")
        res["answer"] = greet_prefix + res["answer"]
        return res
    if re.search(r"\b(what\s+did\s+people\s+dislike|worst\s+part|weakness(?:es)?|negative\s+reviews?|complaints?|criticism|flaws?)\b", q_clean, re.I):
        res = _conversational_summary_reply(q_clean, analysis, "negative")
        res["answer"] = greet_prefix + res["answer"]
        return res
    if re.search(r"\b(worth\s+watching|should\s+i\s+watch|recommend\w*|is\s+it\s+good|is\s+it\s+bad|how\s+is\s+(?:it|the\s+(?:movie|film))|overall|summary|summarize|overview|verdict|popcorn\s+(?:meter|score)|rating|overall\s+score|how\s+many\s+reviews|vibe|emotion)\b", q_clean, re.I) and not (set(q_distinct) - {"worth", "watching", "watch", "recommend", "recommended", "overall", "summary", "summarize", "overview", "verdict", "popcorn", "meter", "rating", "score", "vibe", "emotion", "emotions"}):
        res = _conversational_summary_reply(q_clean, analysis, "overall")
        res["answer"] = greet_prefix + res["answer"]
        return res

    aspects_hit = _detect_aspects_in_question(q_clean)
    broad_aspect_terms = {
        "acting", "actor", "actors", "actress", "cast", "performance", "performances",
        "story", "storyline", "plot", "pacing", "direction", "director",
        "cinematography", "visual", "visuals", "soundtrack", "score", "music", "vfx", "cgi",
    }
    if aspects_hit and (not hits or set(q_distinct).issubset(broad_aspect_terms)):
        res = _conversational_aspect_reply(aspects_hit, analysis, index)
        res["answer"] = greet_prefix + res["answer"]
        return res

    # 6. No relevant hits found -> explicitly report info not available / not in database
    if not hits:
        return {
            "answer": greet_prefix + _not_in_db_response(analysis, available_movies=available_movies),
            "citations": [],
            "generator": "Database Lookup (Not Found)",
        }

    # Check if the user asked about a specific entity/concept that isn't actually in the corpus
    # even if another generic word matched (e.g., "ending of Titanic" or "budget numbers")
    corpus_tokens = set(_tokenize_words(" ".join(analysis["reviews"]["text"].tolist())))
    title_tokens = set(_tokenize_words(analysis["title"]))
    missing_tokens = [
        t for t in q_distinct
        if t not in title_tokens and not _token_matches_corpus(t, corpus_tokens)
    ]
    # If query has specific non-aspect content words and NONE of the capitalized/rare terms exist
    if missing_tokens and len(missing_tokens) == len(q_distinct):
        return {
            "answer": greet_prefix + _not_in_db_response(analysis, available_movies=available_movies),
            "citations": [],
            "generator": "Database Lookup (Not Found)",
        }

    # 7. Natural, human-sounding RAG synthesis for matching reviews
    pos = sum(1 for h in hits if h["label"] == "positive")
    neu = sum(1 for h in hits if h["label"] == "neutral")
    neg = sum(1 for h in hits if h["label"] == "negative")
    lead = hits[0]

    if pos > neg and neg == 0:
        intro = f"Viewers seem genuinely upbeat about that in **{analysis['title']}**"
    elif neg > pos and pos == 0:
        intro = f"Audience feedback on that point is mostly critical for **{analysis['title']}**"
    elif pos > neg:
        intro = f"Reactions lean positive on that ({pos} positive, {neg} negative, {neu} neutral)"
    elif neg > pos:
        intro = f"Reactions lean negative on that ({neg} negative, {pos} positive, {neu} neutral)"
    else:
        intro = f"Viewers are pretty split on that ({pos} positive, {neg} negative, {neu} neutral)"

    answer = f"{greet_prefix}{intro}. One closely matching review ({lead['label']}) puts it like this: \"{_short(lead['text'], 200)}\""
    contrasting = next((h for h in hits[1:] if h["label"] != lead["label"]), None)
    if contrasting:
        answer += f" On the other hand, a {contrasting['label']} review mentions: \"{_short(contrasting['text'], 160)}\""
    elif len(hits) > 1:
        answer += f" Another viewer also noted: \"{_short(hits[1]['text'], 150)}\""

    return {
        "answer": answer,
        "citations": hits,
        "generator": f"Conversational RAG ({analysis['rag_backend']})",
    }


# ------------------------------------------------------- Global 6-Movie RAG
_GENRE_ALIASES = {
    "sci-fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "mystery / drama": "Mystery / Drama",
    "mystery/drama": "Mystery / Drama",
    "mystery": "Mystery / Drama",
    "drama": "Mystery / Drama",
    "noir": "Mystery / Drama",
    "action": "Action",
    "horror": "Horror",
    "romance": "Romance",
    "romantic": "Romance",
    "comedy": "Comedy",
    "comedic": "Comedy",
}


def build_global_rag_bundle(df_all: pd.DataFrame, model, adjust_sarcasm: bool = True) -> dict:
    """Build one unified, cached RAG index and per-movie summaries across all curated movies."""
    from .config import MOVIE_PROFILES

    df = df_all.copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    if "movie" not in df.columns:
        df["movie"] = "Unknown Movie"
    if "genre" not in df.columns:
        df["genre"] = df["movie"].map(lambda m: MOVIE_PROFILES.get(m, {}).get("genre", "General"))
    else:
        df["genre"] = df.apply(
            lambda r: r["genre"] if pd.notna(r["genre"]) and str(r["genre"]).strip()
            else MOVIE_PROFILES.get(r["movie"], {}).get("genre", "General"),
            axis=1,
        )
    if "date" not in df.columns:
        df["date"] = ""

    texts = df["text"].tolist()
    preds = model.predict(texts)
    raw_labels = [p["label"] for p in preds]
    raw_scores = [p["score"] for p in preds]
    sarcasm_info = [detect_sarcasm(t) for t in texts]
    spoiler_info = [detect_spoiler(t) for t in texts]

    final_labels = []
    for lbl, sc, sarc in zip(raw_labels, raw_scores, sarcasm_info):
        adj_lbl, _ = adjust_sentiment_for_sarcasm(lbl, sc, sarc) if adjust_sarcasm else (lbl, sc)
        final_labels.append(adj_lbl)

    df["label"] = final_labels
    df["is_sarcastic"] = [s["is_sarcastic"] for s in sarcasm_info]
    df["has_spoiler"] = [s["has_spoiler"] for s in spoiler_info]

    docs = [
        {
            "text": r.text,
            "label": r.label,
            "kind": "review",
            "aspect": None,
            "movie": str(r.movie),
            "genre": str(r.genre),
            "date": str(getattr(r, "date", "")),
            "has_spoiler": bool(r.has_spoiler),
            "is_sarcastic": bool(r.is_sarcastic),
        }
        for r in df.itertuples()
    ]
    index = ReviewIndex()
    index.build(docs)

    movie_genres: dict = {}
    genre_movies: dict = {}
    movie_stats: dict = {}
    for movie_name, grp in df.groupby("movie", sort=False):
        genre_name = str(grp["genre"].iloc[0])
        movie_genres[movie_name] = genre_name
        genre_movies.setdefault(genre_name, []).append(movie_name)
        counts = {l: int((grp["label"] == l).sum()) for l in LABELS}
        n = len(grp)
        pct = {l: 100.0 * c / n for l, c in counts.items()}
        popcorn = compute_popcorn_meter(pct)
        asp_df = analyze_aspects(grp["text"].tolist(), model)
        asp_summ = summarize_aspects(asp_df)
        movie_stats[movie_name] = {
            "title": movie_name,
            "genre": genre_name,
            "n_reviews": n,
            "counts": counts,
            "pct": pct,
            "popcorn": popcorn,
            "aspect_summary": asp_summ,
            "emotions": score_emotions(grp["text"].tolist()),
            "sarcasm_count": int(grp["is_sarcastic"].sum()),
            "spoiler_count": int(grp["has_spoiler"].sum()),
            "reviews": grp.reset_index(drop=True),
        }

    return {
        "reviews": df,
        "rag_index": index,
        "rag_backend": index.backend_name,
        "movie_genres": movie_genres,
        "genre_movies": genre_movies,
        "movie_stats": movie_stats,
    }


def detect_query_scope(question: str, movie_genres: dict) -> dict:
    """Detect whether the query targets a specific movie, a genre, or all curated movies."""
    q_low = question.lower()
    # Match longest movie names first
    for movie_name in sorted(movie_genres.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(movie_name.lower())}\b", q_low):
            g = movie_genres[movie_name]
            return {"type": "movie", "movie": movie_name, "genre": g, "label": f"Movie: {movie_name} ({g})"}

    # Match genre aliases
    available_genres = set(movie_genres.values())
    for alias in sorted(_GENRE_ALIASES.keys(), key=len, reverse=True):
        canonical = _GENRE_ALIASES[alias]
        if canonical in available_genres and re.search(rf"\b{re.escape(alias)}\b", q_low):
            return {"type": "genre", "movie": None, "genre": canonical, "label": f"Genre: {canonical}"}

    return {"type": "all", "movie": None, "genre": None, "label": "All Movies (6 Genres)"}


def _global_catalog_summary(movie_genres: dict) -> str:
    return ", ".join(f"**{m}** ({g})" for m, g in movie_genres.items())


def _global_not_in_db_reply(global_bundle: dict, subject: Optional[str] = None) -> dict:
    catalog = _global_catalog_summary(global_bundle["movie_genres"])
    if subject:
        ans = (
            f"**{subject}** was **not found in the database** (information not available). "
            f"The curated 6-genre database currently covers: {catalog}."
        )
    else:
        ans = (
            "That information is **not available in the database** — none of the reviews across the 6 curated movies cover it. "
            f"Try asking about acting, plot, pacing, visuals, soundtrack, spoilers, or any of: {catalog}."
        )
    return {
        "answer": ans,
        "citations": [],
        "scope": "Database Lookup (Not Found)",
        "generator": "Database Lookup (Not Found)",
    }


def _check_global_out_of_db(question: str, global_bundle: dict) -> Optional[dict]:
    """Detect references to unindexed movies/documents when scope is 'all'."""
    corpus_tokens = set(_tokenize_words(" ".join(global_bundle["reviews"]["text"].tolist())))
    all_title_tokens = set()
    for m, g in global_bundle["movie_genres"].items():
        all_title_tokens.update(_tokenize_words(m))
        all_title_tokens.update(_tokenize_words(g))

    for pat in _TARGET_PATTERNS:
        m = pat.search(question)
        if not m:
            continue
        raw_target = _clean_subject(m.group(1))
        if not raw_target or raw_target.lower() in _GENERIC_SUBJECTS:
            continue
        target_words = [w for w in _tokenize_words(raw_target) if w not in _STOP_AND_GENERIC]
        if not target_words or all(w in all_title_tokens for w in target_words):
            continue
        if not any(_token_matches_corpus(w, corpus_tokens) for w in target_words):
            display_name = " ".join(w.capitalize() if w.islower() else w for w in raw_target.split())
            return _global_not_in_db_reply(global_bundle, subject=display_name)

    words_raw = re.findall(r"[A-Za-z0-9:'\-]+", question)
    for idx, w in enumerate(words_raw):
        if idx == 0:
            continue
        w_low = w.lower()
        if w[0].isupper() and w_low not in _GENERIC_SUBJECTS and w_low not in all_title_tokens and w_low != "i":
            if not _token_matches_corpus(w_low, corpus_tokens):
                return _global_not_in_db_reply(global_bundle, subject=w)

    return None


def _global_scope_overview_reply(question: str, scope: dict, global_bundle: dict, mode: str) -> dict:
    index: ReviewIndex = global_bundle["rag_index"]
    stats_map = global_bundle["movie_stats"]
    movie_genres = global_bundle["movie_genres"]

    if scope["type"] == "movie":
        target_movies = [scope["movie"]]
    elif scope["type"] == "genre":
        target_movies = global_bundle["genre_movies"].get(scope["genre"], [])
    else:
        target_movies = list(movie_genres.keys())

    # If narrowed to 1 movie (either direct movie match or single movie per genre)
    if len(target_movies) == 1:
        st_m = stats_map[target_movies[0]]
        title, genre = st_m["title"], st_m["genre"]
        pop, pct, n = st_m["popcorn"], st_m["pct"], st_m["n_reviews"]
        ranked = st_m["aspect_summary"].sort_values("net_score", ascending=False)
        best = ranked[ranked["net_score"] > 0.15]["aspect_name"].tolist()
        weak = ranked[ranked["net_score"] < -0.15]["aspect_name"].tolist()

        if mode == "spoilers":
            sp_df = st_m["reviews"][st_m["reviews"]["has_spoiler"]]
            if sp_df.empty:
                return {"answer": f"No plot spoilers were flagged in **{title}** ({genre}).", "citations": [], "scope": scope["label"], "generator": "Conversational NLP Insight"}
            cites = [{"text": r.text, "label": r.label, "movie": title, "genre": genre, "has_spoiler": True, "similarity": 1.0} for r in sp_df.head(3).itertuples()]
            return {"answer": f"Heads up — **{len(sp_df)}** review(s) for **{title}** ({genre}) contain plot spoilers (shielded below).", "citations": cites, "scope": scope["label"], "generator": "Conversational NLP Insight"}

        if mode == "sarcasm":
            sc_df = st_m["reviews"][st_m["reviews"]["is_sarcastic"]]
            if sc_df.empty:
                return {"answer": f"No sarcastic reviews were flagged for **{title}** ({genre}).", "citations": [], "scope": scope["label"], "generator": "Conversational NLP Insight"}
            cites = [{"text": r.text, "label": r.label, "movie": title, "genre": genre, "has_spoiler": bool(r.has_spoiler), "similarity": 1.0} for r in sc_df.head(3).itertuples()]
            return {"answer": f"The Sarcasm Radar caught **{len(sc_df)}** ironic review(s) for **{title}** ({genre}) — e.g. \"{_short(sc_df.iloc[0]['text'], 160)}\"", "citations": cites, "scope": scope["label"], "generator": "Conversational NLP Insight"}

        if mode == "positive":
            cites = index.search("great movie highly recommend loved", k=3, label="positive", movie=title)
            praise = f" Standout strengths: **{', '.join(best[:2])}**." if best else ""
            ans = f"**{title}** ({genre}) has **{pct['positive']:.0f}% positive** reviews.{praise}"
            if cites:
                ans += f" A top viewer raves: \"{_short(cites[0]['text'], 170)}\""
            return {"answer": ans, "citations": cites, "scope": scope["label"], "generator": f"Global RAG ({global_bundle['rag_backend']})"}

        if mode == "negative":
            cites = index.search("disappointing bad boring wasted", k=3, label="negative", movie=title)
            flaws = f" Main pain points: **{', '.join(weak[:2])}**." if weak else ""
            ans = f"About **{pct['negative']:.0f}%** of reviews for **{title}** ({genre}) lean negative.{flaws}"
            if cites:
                ans += f" One critic noted: \"{_short(cites[0]['text'], 170)}\""
            return {"answer": ans, "citations": cites, "scope": scope["label"], "generator": f"Global RAG ({global_bundle['rag_backend']})"}

        tone = _tone(pct)
        ans = (
            f"**{title}** ({genre}) has a **{tone}** reception with a **{pop['score']:.0f}% Popcorn Meter ({pop['tier']})** "
            f"across {n} reviews ({pct['positive']:.0f}% positive, {pct['neutral']:.0f}% neutral, {pct['negative']:.0f}% negative)."
        )
        if best:
            ans += f" Viewers praised **{', '.join(best[:2])}**"
            ans += f", while **{', '.join(weak[:2])}** drew some pushback." if weak else "."
        elif weak:
            ans += f" Most criticism targeted **{', '.join(weak[:2])}**."
        cites = (
            index.search("recommend great worth watching", k=2, label="positive", movie=title)
            + index.search("disappointing weak", k=1, label="negative", movie=title)
        )[:3]
        return {"answer": ans, "citations": cites, "scope": scope["label"], "generator": f"Global RAG ({global_bundle['rag_backend']})"}

    # All 6 movies leaderboard / cross-movie summary
    ranked_movies = sorted(stats_map.values(), key=lambda s: s["popcorn"]["score"], reverse=True)
    if mode == "spoilers":
        sp_rows = global_bundle["reviews"][global_bundle["reviews"]["has_spoiler"]]
        cites = [{"text": r.text, "label": r.label, "movie": r.movie, "genre": r.genre, "has_spoiler": True, "similarity": 1.0} for r in sp_rows.head(4).itertuples()]
        return {"answer": f"Across all 6 curated movies, **{len(sp_rows)}** reviews contain plot spoilers (all hidden behind the Spoiler Shield below).", "citations": cites, "scope": scope["label"], "generator": "Conversational NLP Insight"}

    if mode == "sarcasm":
        sc_rows = global_bundle["reviews"][global_bundle["reviews"]["is_sarcastic"]]
        cites = [{"text": r.text, "label": r.label, "movie": r.movie, "genre": r.genre, "has_spoiler": bool(r.has_spoiler), "similarity": 1.0} for r in sc_rows.head(4).itertuples()]
        return {"answer": f"Across all 6 movies, the Sarcasm Radar detected **{len(sc_rows)}** ironic/backhanded reviews.", "citations": cites, "scope": scope["label"], "generator": "Conversational NLP Insight"}

    top = ranked_movies[0]
    runner_up = ranked_movies[1]
    board = "; ".join(f"**{s['title']}** ({s['genre']}, {s['popcorn']['score']:.0f}%)" for s in ranked_movies)
    ans = (
        f"Across all 6 genres, the crowd favorite right now is **{top['title']}** ({top['genre']}) at "
        f"**{top['popcorn']['score']:.0f}% Popcorn Meter ({top['popcorn']['tier']})**, followed by **{runner_up['title']}** "
        f"({runner_up['popcorn']['score']:.0f}%). Full standings: {board}."
    )
    cites = index.search("best recommend masterpiece triumph", k=3, label="positive")
    return {"answer": ans, "citations": cites, "scope": scope["label"], "generator": f"Global RAG ({global_bundle['rag_backend']})"}


def answer_global_question(
    question: str,
    global_bundle: dict,
    use_llm: bool = False,
    chat_history: Optional[List[dict]] = None,
) -> dict:
    """Global, non-reloading conversational RAG Q&A with automatic movie & genre scope detection."""
    q_raw = (question or "").strip()
    movie_genres = global_bundle["movie_genres"]
    if not q_raw:
        return {
            "answer": f"Ask me anything across all 6 genres ({_global_catalog_summary(movie_genres)})!",
            "citations": [],
            "scope": "All Movies (6 Genres)",
            "generator": "Conversational Assistant",
        }

    # 1. Pure small talk
    st_kind = _classify_small_talk(q_raw)
    if st_kind:
        if st_kind == "greeting":
            ans = (
                "Hey there! I'm doing great — I've got audience reviews across all **6 genres** loaded right here: "
                f"{_global_catalog_summary(movie_genres)}. Ask about a specific movie, a genre like *Comedy* or *Horror*, or compare them all!"
            )
        elif st_kind == "wellbeing":
            ans = "I'm doing well, thanks for asking! Ready to chat movies whenever you are — which film or genre are you curious about?"
        elif st_kind == "identity":
            ans = (
                f"I'm **SlayOrSkip**, your global movie-review assistant covering **6 genres** ({_global_catalog_summary(movie_genres)}). "
                "Mention any movie name or genre to auto-filter my search, or ask across all films. If something isn't in the database, I'll let you know!"
            )
        elif st_kind == "thanks":
            ans = "Anytime! Feel free to ask about any of the 6 movies or genres whenever you're ready."
        else:
            ans = "See you at the movies! Enjoy the show 🎬"
        return {"answer": ans, "citations": [], "scope": "Conversational Assistant", "generator": "Conversational Assistant"}

    greet_prefix = ""
    q_clean = q_raw
    gm = _GREETING_RE.match(q_raw)
    if gm and len(q_raw[gm.end():].strip()) > 1:
        greet_prefix = "Hey! "
        q_clean = q_raw[gm.end():].strip()

    # Follow-up support
    if re.match(r"^(?:why\??|tell\s+me\s+more|elaborate|explain(?:\s+more)?|more\s+details?)[?.!\s]*$", q_clean, re.I):
        if chat_history:
            prev = chat_history[-1]
            if prev.get("citations"):
                extra = " ".join(f"[{c.get('movie', '')}] \"{_short(c['text'], 150)}\"" for c in prev["citations"][:2])
                return {
                    "answer": f"{greet_prefix}Here's a closer look from those reviews: {extra}",
                    "citations": prev["citations"],
                    "scope": prev.get("scope", "All Movies (6 Genres)"),
                    "generator": "Conversational Follow-up",
                }
        q_clean = "overall summary and best rated movie"

    # 2. Automatic movie & genre scope detection
    scope = detect_query_scope(q_clean, movie_genres)

    # 3. If no curated movie/genre matched, check for unindexed external movies/documents
    if scope["type"] == "all":
        out_of_db = _check_global_out_of_db(q_clean, global_bundle)
        if out_of_db:
            return out_of_db

    # 4. Retrieve scoped reviews
    index: ReviewIndex = global_bundle["rag_index"]
    is_dense = "SentenceTransformers" in index.backend_name
    min_score = 0.25 if is_dense else 0.06
    hits = index.search(q_clean, k=4, kind="review", movie=scope["movie"], genre=scope["genre"], min_score=min_score)

    q_distinct = _distinctive_tokens(q_clean)
    # Remove movie/genre tokens from distinctive check so "How is the Comedy movie?" doesn't require "comedy" inside review text
    scope_tokens = set()
    if scope["movie"]:
        scope_tokens.update(_tokenize_words(scope["movie"]))
    if scope["genre"]:
        scope_tokens.update(_tokenize_words(scope["genre"]))
        scope_tokens.update({"romantic", "comedic", "scifi", "sci"})
    topic_distinct = [t for t in q_distinct if t not in scope_tokens]

    if not is_dense and hits and topic_distinct:
        hits_tokens = set(_tokenize_words(" ".join(h["text"] for h in hits)))
        if not any(_token_matches_corpus(t, hits_tokens) for t in topic_distinct):
            hits = []

    # 5. Optional LLM mode
    if use_llm and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            excerpts = [
                {"movie": h.get("movie"), "genre": h.get("genre"), "label": h["label"], "similarity": round(h["similarity"], 3), "text": _short(h["text"], 300)}
                for h in hits
            ]
            compact_stats = {
                m: {"genre": s["genre"], "popcorn_score": s["popcorn"]["score"], "tier": s["popcorn"]["tier"], "n_reviews": s["n_reviews"]}
                for m, s in global_bundle["movie_stats"].items()
            }
            recent_turns = [{"user": t["q"], "assistant": t["answer"]} for t in (chat_history or [])[-4:]]
            prompt = (
                "You are SlayOrSkip, a friendly, natural-sounding movie review assistant.\n"
                f"Detected Query Scope: {scope['label']}\n"
                "Use ONLY the 6-movie database stats and retrieved excerpts below. Do not invent facts.\n"
                "If the user asks about a movie, document, or detail NOT in the database, clearly state it is 'not found in the database' / 'information not available'.\n\n"
                f"CURATED MOVIES:\n{json.dumps(compact_stats, indent=2)}\n\n"
                f"RECENT CHAT:\n{json.dumps(recent_turns, indent=2)}\n\n"
                f"RETRIEVED REVIEWS:\n{json.dumps(excerpts, indent=2)}\n\n"
                f"USER QUESTION: {q_raw}\n\n"
                "Reply naturally in 2-4 sentences."
            )
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return {
                "answer": msg.content[0].text.strip(),
                "citations": hits,
                "scope": scope["label"],
                "generator": f"LLM ({global_bundle['rag_backend']})",
            }
        except Exception as e:
            print(f"[global_rag_qa] LLM failed, using conversational RAG: {e}")

    # 6. Summary / recommendation / spoiler / sarcasm / positive / negative / aspect intents
    if re.search(r"\b(spoilers?|spoiler)\b", q_clean, re.I):
        res = _global_scope_overview_reply(q_clean, scope, global_bundle, "spoilers")
        res["answer"] = greet_prefix + res["answer"]
        return res
    if re.search(r"\b(sarcas\w*|irony|ironic)\b", q_clean, re.I):
        res = _global_scope_overview_reply(q_clean, scope, global_bundle, "sarcasm")
        res["answer"] = greet_prefix + res["answer"]
        return res
    if re.search(r"\b(what\s+did\s+people\s+like|best\s+part|strengths?|positive\s+reviews?|praise|highlights?)\b", q_clean, re.I):
        res = _global_scope_overview_reply(q_clean, scope, global_bundle, "positive")
        res["answer"] = greet_prefix + res["answer"]
        return res
    if re.search(r"\b(what\s+did\s+people\s+dislike|worst\s+part|weakness(?:es)?|negative\s+reviews?|complaints?|criticism|flaws?)\b", q_clean, re.I):
        res = _global_scope_overview_reply(q_clean, scope, global_bundle, "negative")
        res["answer"] = greet_prefix + res["answer"]
        return res

    summary_words = {
        "worth", "watching", "watch", "recommend", "recommended", "overall", "summary",
        "summarize", "overview", "verdict", "popcorn", "meter", "rating", "score",
        "vibe", "emotion", "emotions", "rank", "compare", "highest", "rated",
        "tonight", "pick", "best", "worst", "top", "favorite", "favourite",
    }
    if (
        (not topic_distinct and scope["type"] != "all")
        or (
            re.search(r"\b(worth\s+watching|should\s+i\s+watch|recommend\w*|is\s+it\s+good|is\s+it\s+bad|how\s+is\b|which\s+(?:movie|film)|what\s+should\s+i\s+watch|overall|summary|summarize|overview|verdict|popcorn\s+(?:meter|score)|rating|overall\s+score|rank|compare|vibe|best\s+(?:movie|film|reviews?))\b", q_clean, re.I)
            and not (set(topic_distinct) - summary_words)
        )
    ):
        res = _global_scope_overview_reply(q_clean, scope, global_bundle, "overall")
        res["answer"] = greet_prefix + res["answer"]
        return res

    # Aspect question scoped to movie/genre/all
    aspects_hit = _detect_aspects_in_question(q_clean)
    broad_aspect_terms = {
        "acting", "actor", "actors", "actress", "cast", "performance", "performances",
        "story", "storyline", "plot", "pacing", "direction", "director",
        "cinematography", "visual", "visuals", "soundtrack", "score", "music", "vfx", "cgi",
    }
    if aspects_hit and (not hits or set(topic_distinct).issubset(broad_aspect_terms)):
        target_movies = (
            [scope["movie"]] if scope["type"] == "movie"
            else (global_bundle["genre_movies"].get(scope["genre"], []) if scope["type"] == "genre" else list(movie_genres.keys()))
        )
        if len(target_movies) == 1:
            st_m = global_bundle["movie_stats"][target_movies[0]]
            summ = st_m["aspect_summary"].set_index("aspect")
            parts, cites, seen = [], [], set()
            for asp in aspects_hit[:2]:
                row = summ.loc[asp]
                asp_name, mentions = row["aspect_name"], int(row["mentions"])
                pos, neu, neg, net = int(row["positive"]), int(row["neutral"]), int(row["negative"]), float(row["net_score"])
                if mentions == 0:
                    parts.append(f"None of the loaded reviews for **{st_m['title']}** discuss **{asp_name}**.")
                elif net >= 0.3:
                    parts.append(f"For **{st_m['title']}**, viewers loved the **{asp_name}** ({pos} positive vs. {neg} negative).")
                elif net <= -0.3:
                    parts.append(f"For **{st_m['title']}**, viewers were critical of the **{asp_name}** ({neg} negative vs. {pos} positive).")
                else:
                    parts.append(f"Opinions on **{asp_name}** in **{st_m['title']}** were mixed ({pos} positive, {neu} neutral, {neg} negative).")
                pref_label = "positive" if net >= 0.3 else ("negative" if net <= -0.3 else None)
                asp_hits = index.search(ASPECT_QUERIES[asp], k=2, label=pref_label, movie=st_m["title"], min_score=min_score)
                if not asp_hits:
                    asp_hits = index.search(ASPECT_QUERIES[asp], k=2, movie=st_m["title"], min_score=0.01)
                for h in asp_hits:
                    if h["text"] not in seen:
                        seen.add(h["text"])
                        cites.append(h)
            if cites:
                parts.append(f"One reviewer noted: \"{_short(cites[0]['text'], 170)}\"")
            return {"answer": greet_prefix + " ".join(parts), "citations": cites[:4], "scope": scope["label"], "generator": f"Global RAG ({global_bundle['rag_backend']})"}
        else:
            asp_cites = index.search(q_clean, k=4, movie=scope["movie"], genre=scope["genre"])
            if asp_cites:
                pos = sum(1 for h in asp_cites if h["label"] == "positive")
                neg = sum(1 for h in asp_cites if h["label"] == "negative")
                lead = asp_cites[0]
                ans = f"{greet_prefix}Across **{scope['label']}**, feedback on **{', '.join(ASPECT_LABELS[a] for a in aspects_hit[:2])}** leans {'positive' if pos >= neg else 'critical'}. For example, on **{lead['movie']}** ({lead['genre']}): \"{_short(lead['text'], 170)}\""
                return {"answer": ans, "citations": asp_cites, "scope": scope["label"], "generator": f"Global RAG ({global_bundle['rag_backend']})"}

    # 7. No matching reviews found -> info not available
    if not hits:
        return _global_not_in_db_reply(global_bundle)

    # 8. Natural RAG synthesis over scoped hits
    pos = sum(1 for h in hits if h["label"] == "positive")
    neu = sum(1 for h in hits if h["label"] == "neutral")
    neg = sum(1 for h in hits if h["label"] == "negative")
    lead = hits[0]
    scope_desc = f"**{scope['movie']}**" if scope["type"] == "movie" else (f"the **{scope['genre']}** reviews" if scope["type"] == "genre" else "our 6-movie database")

    if pos > neg and neg == 0:
        intro = f"Viewers are genuinely upbeat about that in {scope_desc}"
    elif neg > pos and pos == 0:
        intro = f"Feedback on that point is mostly critical in {scope_desc}"
    elif pos > neg:
        intro = f"Reactions in {scope_desc} lean positive ({pos} positive, {neg} negative, {neu} neutral)"
    elif neg > pos:
        intro = f"Reactions in {scope_desc} lean negative ({neg} negative, {pos} positive, {neu} neutral)"
    else:
        intro = f"Viewers in {scope_desc} are split on that ({pos} positive, {neg} negative, {neu} neutral)"

    ans = f"{greet_prefix}{intro}. Closest match from **{lead['movie']}** ({lead['label']}): \"{_short(lead['text'], 185)}\""
    contrasting = next((h for h in hits[1:] if h["label"] != lead["label"]), None)
    if contrasting:
        ans += f" Meanwhile, a {contrasting['label']} review for **{contrasting['movie']}** adds: \"{_short(contrasting['text'], 150)}\""
    elif len(hits) > 1:
        ans += f" Another review (**{hits[1]['movie']}**) adds: \"{_short(hits[1]['text'], 145)}\""

    return {
        "answer": ans,
        "citations": hits,
        "scope": scope["label"],
        "generator": f"Global RAG ({global_bundle['rag_backend']})",
    }


# ------------------------------------------------------- text generation
def _tone(pct: dict) -> str:
    if pct["positive"] >= 65:
        return "overwhelmingly positive"
    if pct["positive"] >= 50 and pct["negative"] < 25:
        return "generally positive"
    if pct["negative"] >= 65:
        return "overwhelmingly negative"
    if pct["negative"] >= 50 and pct["positive"] < 25:
        return "generally negative"
    return "mixed"


def template_sections(a: dict) -> dict:
    """Deterministic RAG-grounded fallback report text (works offline)."""
    pct, n, title = a["pct"], a["n_reviews"], a["title"]
    pop = a["popcorn"]
    summ = a["aspect_summary"].set_index("aspect")
    overview = (
        f"Across {n} reviews of {title}, {pct['positive']:.0f}% are positive, "
        f"{pct['neutral']:.0f}% neutral/mixed, and {pct['negative']:.0f}% negative — "
        f"earning a {pop['score']:.0f}% Popcorn Meter ({pop['tier']}). "
        f"Overall audience reception is {_tone(pct)}."
    )

    strengths, weaknesses = [], []
    ranked = summ[summ["mentions"] >= 1].sort_values("net_score", ascending=False)
    for aspect, r in ranked.iterrows():
        label_name = ASPECT_LABELS.get(aspect, aspect.capitalize())
        ev = a["evidence"]["aspects"][aspect]
        if r["net_score"] > 0.15:
            line = f"{label_name}: {int(r['positive'])} of {int(r['mentions'])} mentions are positive."
            if ev["positive"]:
                line += f" e.g. \"{_short(ev['positive'][0]['text'], 150)}\""
            strengths.append(line)
        elif r["net_score"] < -0.15:
            line = f"{label_name}: {int(r['negative'])} of {int(r['mentions'])} mentions are negative."
            if ev["negative"]:
                line += f" e.g. \"{_short(ev['negative'][0]['text'], 150)}\""
            weaknesses.append(line)

    if not strengths:
        strengths = ["No single cinematic dimension stood out as a dominant strength."]
    if not weaknesses:
        weaknesses = ["No single cinematic dimension stood out as a dominant weakness."]

    top_emo = sorted(a["emotions"].items(), key=lambda kv: kv[1], reverse=True)[:2]
    takeaways = [
        f"Dominant emotional vibe: {top_emo[0][0]} ({top_emo[0][1]:.0f}/100) and {top_emo[1][0]} ({top_emo[1][1]:.0f}/100).",
        f"NLP safety signals: {a['sarcasm_count']} sarcastic/ironic review(s) and {a['spoiler_count']} plot-spoiler warning(s) flagged.",
        f"Corpus profile: {a['corpus_stats']['total_tokens']} total tokens across {n} reviews ({a['corpus_stats']['vocab_size']} unique lemmatized terms).",
    ]

    best_area = ASPECT_LABELS.get(ranked.index[0], ranked.index[0]) if len(ranked) and ranked.iloc[0]["net_score"] > 0.15 else ""
    weak_area = ASPECT_LABELS.get(ranked.index[-1], ranked.index[-1]) if len(ranked) > 1 and ranked.iloc[-1]["net_score"] < -0.15 else ""
    verdict = f"Audiences find {title} {_tone(pct)} ({pop['tier']})."
    if best_area:
        verdict += f" Its strongest craft element is {best_area}."
    if weak_area:
        verdict += f" Its most criticized area is {weak_area}."

    return {
        "overview": overview,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "takeaways": takeaways,
        "verdict": verdict.strip(),
    }


def llm_sections(a: dict) -> Optional[dict]:
    """Synthesize report narrative strictly from ML metrics + RAG-retrieved quotes."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        def q(hits):
            return [_short(h["text"], 300) for h in hits]

        payload = {
            "movie": a["title"],
            "n_reviews": a["n_reviews"],
            "popcorn_meter": a["popcorn"],
            "sentiment_pct": {k: round(v, 1) for k, v in a["pct"].items()},
            "aspect_summary": a["aspect_summary"].to_dict("records"),
            "emotions": a["emotions"],
            "evidence": {
                "positive_reviews": q(a["evidence"]["positive"]),
                "negative_reviews": q(a["evidence"]["negative"]),
                "aspects": {
                    k: {"positive": q(v["positive"]), "negative": q(v["negative"])}
                    for k, v in a["evidence"]["aspects"].items()
                },
            },
        }
        prompt = (
            "You write concise, insightful movie-review intelligence dossiers. Use ONLY the ML statistics "
            "and retrieved review excerpts below; do not invent facts or plot details.\n\n"
            f"DATA:\n{json.dumps(payload, indent=2)}\n\n"
            "Respond with ONLY a JSON object with keys: "
            '"overview" (2-3 sentences), "strengths" (list of 2-4 short strings), '
            '"weaknesses" (list of 2-4 short strings), "takeaways" (list of 2-3 short strings), '
            '"verdict" (2 sentences). No markdown fences.'
        )
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        out = json.loads(text)
        if all(k in out for k in ("overview", "strengths", "weaknesses", "verdict")):
            out.setdefault("takeaways", [])
            return out
    except Exception as e:
        print(f"[report] LLM generation failed, using template: {e}")
    return None


def build_report(a: dict, use_llm: bool = True) -> dict:
    sections = llm_sections(a) if use_llm else None
    generator = "LLM (RAG-grounded)" if sections else "Deterministic RAG Template"
    sections = sections or template_sections(a)

    REPORTS_DIR.mkdir(exist_ok=True)
    slug = slugify(a["title"])
    chart = REPORTS_DIR / f"{slug}_chart.png"
    radar = REPORTS_DIR / f"{slug}_radar.png"
    make_chart(a, chart)
    make_emotion_radar(a, radar)
    return {
        **a,
        **sections,
        "generator": generator,
        "chart_path": str(chart),
        "radar_path": str(radar),
    }


# ------------------------------------------------------------------ charts
def make_chart(a: dict, path) -> None:
    colors = {"negative": "#F43F5E", "neutral": "#38BDF8", "positive": "#FACC15"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), facecolor="#0B0F19")
    for ax in axes:
        ax.set_facecolor("#151932")
        ax.tick_params(colors="#E2E8F0", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    axes[0].bar([l.capitalize() for l in LABELS], [a["counts"][l] for l in LABELS],
                color=[colors[l] for l in LABELS], width=0.55)
    axes[0].set_title("3-Class Sentiment Distribution", color="#F8FAFC", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("Reviews", color="#CBD5E1", fontsize=9)

    s = a["aspect_summary"]
    names = s["aspect_name"].tolist() if "aspect_name" in s.columns else s["aspect"].tolist()
    bottom = [0] * len(s)
    for l in LABELS:
        axes[1].barh(names, s[l], left=bottom, color=colors[l], label=l.capitalize(), height=0.58)
        bottom = [b + v for b, v in zip(bottom, s[l])]
    axes[1].set_title("Cinematic Aspect Mentions", color="#F8FAFC", fontsize=10.5, fontweight="bold")
    leg = axes[1].legend(fontsize=7.5, facecolor="#0B0F19", edgecolor="#334155")
    for txt in leg.get_texts():
        txt.set_color("#E2E8F0")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_emotion_radar(a: dict, path) -> None:
    emotions = a["emotions"]
    labels = list(emotions.keys())
    values = list(emotions.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_loop = values + values[:1]
    angles_loop = angles + angles[:1]

    fig = plt.figure(figsize=(4.8, 4.4), facecolor="#0B0F19")
    ax = fig.add_subplot(111, polar=True, facecolor="#151932")
    ax.plot(angles_loop, values_loop, color="#FACC15", linewidth=2.2)
    ax.fill(angles_loop, values_loop, color="#38BDF8", alpha=0.28)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color="#F8FAFC", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color="#94A3B8", fontsize=7)
    ax.grid(color="#334155", alpha=0.7)
    ax.spines["polar"].set_color("#475569")
    ax.set_title("Emotion & Vibe Spectrum", color="#FACC15", fontsize=10.5, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


# ----------------------------------------------------------------- export
def _evidence_lines(rep: dict) -> List[str]:
    lines = []
    for label in ("positive", "negative"):
        for h in rep["evidence"][label][:2]:
            prefix = "[SPOILER-TAGGED] " if h.get("has_spoiler") else ""
            lines.append(f"[{label.upper()}] {prefix}\"{_short(h['text'], 240)}\"")
    return lines


def export_pdf(rep: dict, path) -> str:
    """Export a cinema-styled 'Box Office Intelligence Dossier' PDF."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    midnight = colors.HexColor("#0B0F19")
    indigo = colors.HexColor("#151932")
    gold = colors.HexColor("#FACC15")
    sky = colors.HexColor("#38BDF8")
    slate = colors.HexColor("#E2E8F0")

    st = getSampleStyleSheet()
    st.add(ParagraphStyle("DossierBannerTitle", parent=st["Title"], textColor=gold, fontSize=18, leading=22, alignment=0))
    st.add(ParagraphStyle("DossierSub", parent=st["Normal"], textColor=slate, fontSize=9.5, leading=13))
    st.add(ParagraphStyle("DossierH2", parent=st["Heading2"], textColor=indigo, spaceBefore=10, spaceAfter=4))

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=f"SlayOrSkip - {rep['title']} Box Office Intelligence Dossier",
    )

    P = lambda t, s="BodyText": Paragraph(escape(str(t)), st[s])
    pop = rep["popcorn"]

    banner = Table(
        [[
            P(f"SLAYORSKIP // BOX OFFICE INTELLIGENCE DOSSIER", "DossierSub"),
            P(f"POPCORN METER: {pop['score']:.0f}% ({pop['tier']})", "DossierSub"),
        ], [
            P(f"{rep['title']}", "DossierBannerTitle"),
            P(f"{rep['n_reviews']} reviews | {rep['generator']}", "DossierSub"),
        ]],
        colWidths=[11.4 * cm, 6.0 * cm],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), midnight),
        ("BOX", (0, 0), (-1, -1), 1.5, gold),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))

    flow = [
        banner,
        Spacer(1, 10),
        P("Executive Consensus", "DossierH2"),
        P(rep["overview"]),
        Spacer(1, 8),
    ]

    tbl = Table(
        [["Sentiment Class", "Reviews", "Audience Share"]]
        + [[l.capitalize(), rep["counts"][l], f"{rep['pct'][l]:.1f}%"] for l in LABELS],
        colWidths=[6.0 * cm, 4.0 * cm, 4.0 * cm],
        hAlign="LEFT",
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), indigo),
        ("TEXTCOLOR", (0, 0), (-1, 0), gold),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F8FAFC")]),
    ]))
    flow += [tbl, Spacer(1, 10), Image(rep["chart_path"], width=17.2 * cm, height=6.6 * cm), Spacer(1, 8)]

    if rep.get("radar_path") and os.path.exists(rep["radar_path"]):
        flow += [Image(rep["radar_path"], width=9.0 * cm, height=8.2 * cm), Spacer(1, 8)]

    s = rep["aspect_summary"]
    disp_cols = ["aspect_name", "positive", "neutral", "negative", "mentions", "net_score"]
    headers = ["Aspect", "Positive", "Neutral", "Negative", "Mentions", "Net Score"]
    atbl = Table([headers] + s[disp_cols].astype(str).values.tolist(), hAlign="LEFT")
    atbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), indigo),
        ("TEXTCOLOR", (0, 0), (-1, 0), sky),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    flow += [
        P("Aspect-Based Sentiment Breakdown", "DossierH2"),
        atbl,
        P("Strengths", "DossierH2"),
    ] + [P("• " + x) for x in rep["strengths"]]

    flow += [P("Weaknesses", "DossierH2")] + [P("• " + x) for x in rep["weaknesses"]]
    if rep.get("takeaways"):
        flow += [P("Audience Takeaways & NLP Signals", "DossierH2")] + [P("• " + x) for x in rep["takeaways"]]
    flow += [P("RAG-Retrieved Review Evidence", "DossierH2")] + [P("• " + x) for x in _evidence_lines(rep)]
    flow += [P("Final Studio Verdict", "DossierH2"), P(rep["verdict"])]
    doc.build(flow)
    return str(path)


def export_docx(rep: dict, path) -> str:
    from docx import Document
    from docx.shared import Inches

    d = Document()
    pop = rep["popcorn"]
    d.add_heading(f"SlayOrSkip Dossier: {rep['title']}", 0)
    d.add_paragraph(
        f"Popcorn Meter: {pop['score']:.0f}% ({pop['tier']}) | "
        f"{rep['n_reviews']} reviews analyzed | Generator: {rep['generator']}."
    )
    d.add_heading("Executive Consensus", 1)
    d.add_paragraph(rep["overview"])

    t = d.add_table(rows=1, cols=3)
    t.style = "Light Grid Accent 1"
    for c, h in zip(t.rows[0].cells, ["Sentiment", "Reviews", "Share"]):
        c.text = h
    for l in LABELS:
        cells = t.add_row().cells
        cells[0].text, cells[1].text, cells[2].text = l.capitalize(), str(rep["counts"][l]), f"{rep['pct'][l]:.1f}%"
    d.add_picture(rep["chart_path"], width=Inches(6.3))
    if rep.get("radar_path") and os.path.exists(rep["radar_path"]):
        d.add_picture(rep["radar_path"], width=Inches(4.2))

    d.add_heading("Aspect-Based Sentiment", 1)
    s = rep["aspect_summary"]
    disp_cols = ["aspect_name", "positive", "neutral", "negative", "mentions", "net_score"]
    headers = ["Aspect", "Positive", "Neutral", "Negative", "Mentions", "Net Score"]
    at = d.add_table(rows=1, cols=len(headers))
    at.style = "Light Grid Accent 1"
    for c, h in zip(at.rows[0].cells, headers):
        c.text = h
    for row in s[disp_cols].astype(str).values.tolist():
        for c, v in zip(at.add_row().cells, row):
            c.text = v

    d.add_heading("Strengths", 1)
    for x in rep["strengths"]:
        d.add_paragraph(x, style="List Bullet")
    d.add_heading("Weaknesses", 1)
    for x in rep["weaknesses"]:
        d.add_paragraph(x, style="List Bullet")
    if rep.get("takeaways"):
        d.add_heading("Audience Takeaways", 1)
        for x in rep["takeaways"]:
            d.add_paragraph(x, style="List Bullet")
    d.add_heading("Supporting Evidence (RAG)", 1)
    for x in _evidence_lines(rep):
        d.add_paragraph(x, style="List Bullet")
    d.add_heading("Overall Verdict", 1)
    d.add_paragraph(rep["verdict"])
    d.save(str(path))
    return str(path)
