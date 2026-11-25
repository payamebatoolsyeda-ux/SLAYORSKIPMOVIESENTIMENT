"""Basic unit and integration tests for the SlayOrSkip movie sentiment app."""
import pandas as pd

from src import rag
from src.aspects import extract_aspect_sentences
from src.config import DATA_DIR, MOVIE_PROFILES
from src.data import load_reviews_csv
from src.nlp_insights import (
    adjust_sentiment_for_sarcasm,
    compute_popcorn_meter,
    detect_sarcasm,
    detect_spoiler,
)
from src.preprocess import clean_text, compute_corpus_stats, tokenize
from src.report import answer_global_question, build_global_rag_bundle, detect_query_scope


class _DummySentimentModel:
    """Fast deterministic sentiment predictor for offline CI testing."""

    def predict(self, texts):
        results = []
        for t in texts:
            low = str(t).lower()
            if any(w in low for w in ("terrible", "wooden", "disappointing", "unfunny", "bored", "mess", "rushed")):
                results.append({"label": "negative", "score": 0.85})
            elif any(w in low for w in ("okay", "average", "decent", "mediocre", "mixed")):
                results.append({"label": "neutral", "score": 0.65})
            else:
                results.append({"label": "positive", "score": 0.90})
        return results


def test_preprocess_clean_and_stats():
    cleaned = clean_text("<b>Great</b>   film &amp; score!")
    assert cleaned == "Great film & score!"
    toks = tokenize(cleaned, remove_stopwords=True, lemmatize=True)
    assert "great" in toks
    stats = compute_corpus_stats(["The acting was great.", "The plot was slow."])
    assert stats["n_reviews"] == 2
    assert stats["total_tokens"] > 0
    assert stats["vocab_size"] > 0


def test_nlp_insights_sarcasm_spoiler_and_popcorn():
    sarc = detect_sarcasm("A masterpiece if you are trying to cure insomnia.")
    assert sarc["is_sarcastic"] is True
    adj_label, _ = adjust_sentiment_for_sarcasm("positive", 0.9, sarc)
    assert adj_label == "negative"

    spoil = detect_spoiler("Spoiler: in the final scene the villain betrays everyone.")
    assert spoil["has_spoiler"] is True

    pop = compute_popcorn_meter({"positive": 75.0, "neutral": 10.0, "negative": 15.0})
    assert pop["tier"] == "Freshly Popped"
    assert pop["score"] >= 68.0


def test_aspect_extraction():
    df = extract_aspect_sentences([
        "The cinematography is stunning. The acting felt wooden."
    ])
    assert not df.empty
    assert set(df["aspect"]).issuperset({"cinematography", "acting"})


def test_curated_six_genres_dataset():
    df = load_reviews_csv(DATA_DIR / "sample_reviews.csv")
    expected = {
        "Neon Horizon": "Sci-Fi",
        "Midnight Symphony": "Mystery / Drama",
        "Shadow Protocol": "Action",
        "Whispering Hollow": "Horror",
        "Autumn in Lisbon": "Romance",
        "Cafe Chaos": "Comedy",
    }
    actual = df.drop_duplicates("movie").set_index("movie")["genre"].to_dict()
    assert actual == expected
    assert set(MOVIE_PROFILES.keys()) == set(expected.keys())


def test_global_rag_and_chatbot(monkeypatch):
    # Force fast deterministic TF-IDF backend so CI never blocks on model downloads
    monkeypatch.setattr(rag, "_encoder_failed", True)

    df = pd.read_csv(DATA_DIR / "sample_reviews.csv")
    bundle = build_global_rag_bundle(df, _DummySentimentModel(), adjust_sarcasm=True)
    assert len(bundle["movie_genres"]) == 6

    scope_movie = detect_query_scope("How was the acting in Shadow Protocol?", bundle["movie_genres"])
    assert scope_movie["type"] == "movie"
    assert scope_movie["movie"] == "Shadow Protocol"

    scope_genre = detect_query_scope("How is the Comedy movie?", bundle["movie_genres"])
    assert scope_genre["type"] == "genre"
    assert scope_genre["genre"] == "Comedy"

    greeting = answer_global_question("hi", bundle)
    assert greeting["citations"] == []

    acting = answer_global_question("How was the acting in Shadow Protocol?", bundle)
    assert acting["scope"] == "Movie: Shadow Protocol (Action)"
    assert len(acting["citations"]) >= 1
    assert all(c["movie"] == "Shadow Protocol" for c in acting["citations"])

    missing = answer_global_question("What do people think about Inception?", bundle)
    assert "not found in the database" in missing["answer"].lower()
    assert missing["citations"] == []
