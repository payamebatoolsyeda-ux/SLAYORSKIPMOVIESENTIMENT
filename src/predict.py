"""Unified sentiment prediction, 4-model Arena voting, and token-level XAI."""
import re
from typing import Dict, List

import joblib
import numpy as np

from .config import ARENA_MODELS_PATH, BASELINE_PATH, BERT_DIR, ID2LABEL, LABELS

MODEL_DISPLAY = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "svm": "Support Vector Machine",
    "neural_mlp": "Neural Network (MLP)",
    "bert": "DistilBERT (Transformer)",
}


def _try_load_bert():
    if not (BERT_DIR / "config.json").exists():
        return None
    try:
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model=str(BERT_DIR),
            tokenizer=str(BERT_DIR),
            truncation=True,
            max_length=256,
        )
    except Exception:
        return None


def _predict_sklearn(pipe, texts: List[str]) -> List[dict]:
    preds = pipe.predict(texts)
    clf = pipe.named_steps["clf"]
    if hasattr(pipe, "predict_proba"):
        probs_mat = pipe.predict_proba(texts)
        probs = probs_mat.max(axis=1)
        prob_dicts = [{LABELS[i]: round(float(row[i]), 4) for i in range(len(LABELS))} for row in probs_mat]
    elif hasattr(clf, "predict_proba"):
        probs_mat = pipe.predict_proba(texts)
        probs = probs_mat.max(axis=1)
        prob_dicts = [{LABELS[i]: round(float(row[i]), 4) for i in range(len(LABELS))} for row in probs_mat]
    else:
        probs = [None] * len(texts)
        prob_dicts = [None] * len(texts)
    return [
        {
            "label": ID2LABEL[int(p)],
            "score": None if s is None else float(s),
            "probs": pd,
        }
        for p, s, pd in zip(preds, probs, prob_dicts)
    ]


class SentimentModel:
    def __init__(self, prefer: str = "auto"):
        """prefer: 'auto' (BERT if available & working, then baseline), 'bert', or 'baseline'."""
        self.kind = None
        self._pipe = None
        if prefer in ("auto", "bert"):
            bert_pipe = _try_load_bert()
            if bert_pipe is not None:
                self._pipe = bert_pipe
                self.kind = "bert"

        if self._pipe is None and prefer in ("auto", "baseline") and BASELINE_PATH.exists():
            self._pipe = joblib.load(BASELINE_PATH)
            self.kind = "baseline"
        elif self._pipe is None:
            raise FileNotFoundError(
                "No trained model found. Run `python -m src.train_baselines` "
                "(and optionally `python -m src.train_bert`) first."
            )

    def predict(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """Returns [{'label': str, 'score': float | None, 'probs': dict | None}, ...]"""
        texts = [str(t) for t in texts]
        if not texts:
            return []
        if self.kind == "bert":
            outs = self._pipe(texts, batch_size=batch_size)
            return [{"label": o["label"], "score": float(o["score"]), "probs": None} for o in outs]
        return _predict_sklearn(self._pipe, texts)


class ArenaPredictor:
    """Loads all trained ML architectures for side-by-side voting and token-level XAI."""

    def __init__(self):
        self.models: Dict[str, object] = {}
        if ARENA_MODELS_PATH.exists():
            self.models.update(joblib.load(ARENA_MODELS_PATH))
        elif BASELINE_PATH.exists():
            self.models["baseline"] = joblib.load(BASELINE_PATH)

        bert_pipe = _try_load_bert()
        if bert_pipe is not None:
            self.models["bert"] = bert_pipe

        if not self.models:
            raise FileNotFoundError("No Arena models found. Run `python -m src.train_baselines` first.")

    def predict_all(self, text: str) -> dict:
        votes = []
        counts = {l: 0 for l in LABELS}
        for key, model in self.models.items():
            if key == "bert":
                out = model([text])[0]
                pred = {"label": out["label"], "score": float(out["score"]), "probs": None}
            else:
                pred = _predict_sklearn(model, [text])[0]
            counts[pred["label"]] += 1
            votes.append({
                "key": key,
                "name": MODEL_DISPLAY.get(key, key),
                "label": pred["label"],
                "confidence": round(pred["score"] or 0.0, 3),
                "probs": pred.get("probs"),
            })
        consensus = max(LABELS, key=lambda l: (counts[l], l == "neutral"))
        return {"votes": votes, "counts": counts, "consensus": consensus}

    def explain_tokens(self, text: str, model_key: str = "logistic_regression") -> dict:
        """Compute token-level positive vs negative push weights for a review."""
        if model_key not in self.models:
            model_key = next(iter(self.models))
        pipe = self.models[model_key]
        raw_tokens = re.findall(r"\S+", str(text))
        if not raw_tokens:
            return {"model": MODEL_DISPLAY.get(model_key, model_key), "tokens": [], "top_positive": [], "top_negative": []}

        weights = self._token_weights(str(text), raw_tokens, model_key, pipe)
        max_abs = max((abs(w) for w in weights), default=1.0) or 1.0
        norm_tokens = []
        agg: Dict[str, float] = {}

        for tok, w in zip(raw_tokens, weights):
            nw = round(float(w / max_abs), 3)
            direction = "positive" if nw > 0.08 else ("negative" if nw < -0.08 else "neutral")
            norm_tokens.append({"token": tok, "weight": nw, "direction": direction})
            clean = re.sub(r"[^a-z0-9']+", "", tok.lower())
            if len(clean) >= 2 and abs(nw) > 0.08:
                agg[clean] = agg.get(clean, 0.0) + nw

        sorted_terms = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        top_pos = [{"term": k, "weight": round(v, 3)} for k, v in sorted_terms if v > 0][:6]
        top_neg = [{"term": k, "weight": round(v, 3)} for k, v in reversed(sorted_terms) if v < 0][:6]
        return {
            "model": MODEL_DISPLAY.get(model_key, model_key),
            "tokens": norm_tokens,
            "top_positive": top_pos,
            "top_negative": top_neg,
        }

    def _token_weights(self, text: str, raw_tokens: List[str], model_key: str, pipe) -> List[float]:
        # Linear / NB models: exact TF-IDF feature weight attribution (positive class vs negative class)
        if model_key in ("logistic_regression", "svm", "naive_bayes") and hasattr(pipe, "named_steps"):
            vec = pipe.named_steps["tfidf"]
            clf = pipe.named_steps["clf"]
            vocab = vec.vocabulary_
            if hasattr(clf, "coef_"):
                diff = clf.coef_[2] - clf.coef_[0]
            elif hasattr(clf, "calibrated_classifiers_"):
                coefs = [c.estimator.coef_[2] - c.estimator.coef_[0] for c in clf.calibrated_classifiers_]
                diff = np.mean(coefs, axis=0)
            elif hasattr(clf, "feature_log_prob_"):
                diff = clf.feature_log_prob_[2] - clf.feature_log_prob_[0]
            else:
                diff = None

            if diff is not None:
                x = vec.transform([text])
                weights = []
                cleaned = [re.sub(r"[^a-z0-9']+", "", t.lower()) for t in raw_tokens]
                for i, tok in enumerate(cleaned):
                    score = 0.0
                    if tok in vocab:
                        idx = vocab[tok]
                        score += float(x[0, idx] * diff[idx])
                    # Include adjacent bigram contribution split evenly
                    for pair in (
                        f"{cleaned[i - 1]} {tok}" if i > 0 else "",
                        f"{tok} {cleaned[i + 1]}" if i + 1 < len(cleaned) else "",
                    ):
                        if pair and pair in vocab:
                            idx = vocab[pair]
                            score += 0.5 * float(x[0, idx] * diff[idx])
                    weights.append(score)
                return weights

        # Perturbation sensitivity fallback for neural / transformer models
        if model_key == "bert":
            base_out = pipe([text])[0]
            base_score = base_out["score"] if base_out["label"] == "positive" else (-base_out["score"] if base_out["label"] == "negative" else 0.0)
            weights = []
            for i in range(len(raw_tokens)):
                masked = " ".join(t for j, t in enumerate(raw_tokens) if j != i) or " "
                out = pipe([masked])[0]
                m_score = out["score"] if out["label"] == "positive" else (-out["score"] if out["label"] == "negative" else 0.0)
                weights.append(base_score - m_score)
            return weights

        base_probs = pipe.predict_proba([text])[0]
        base_net = float(base_probs[2] - base_probs[0])
        masked_texts = [" ".join(t for j, t in enumerate(raw_tokens) if j != i) or " " for i in range(len(raw_tokens))]
        masked_probs = pipe.predict_proba(masked_texts)
        return [base_net - float(row[2] - row[0]) for row in masked_probs]
