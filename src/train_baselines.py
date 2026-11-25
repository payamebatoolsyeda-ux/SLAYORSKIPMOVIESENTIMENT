"""Train 4 sentiment architectures on SST-5 (3-class) and export Model Arena artifacts.

Models:
1. Naive Bayes (MultinomialNB)
2. Logistic Regression (TF-IDF)
3. Support Vector Machine (Calibrated LinearSVC)
4. Neural Network (MLPClassifier over TF-IDF + SVD)

Run:  python -m src.train_baselines
"""
import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import (ARENA_EVAL_PATH, ARENA_MODELS_PATH, BASELINE_PATH,
                     MODELS_DIR, REPORTS_DIR)
from .data import load_sst5_3class
from .evaluate import (collect_confusion_samples, compute_roc_data,
                       evaluate_and_save)

MODEL_META = {
    "naive_bayes": {"display_name": "Naive Bayes", "family": "Probabilistic Baseline"},
    "logistic_regression": {"display_name": "Logistic Regression", "family": "Linear TF-IDF"},
    "svm": {"display_name": "Support Vector Machine", "family": "Calibrated LinearSVC"},
    "neural_mlp": {"display_name": "Neural Network (MLP)", "family": "Deep MLP on LSA/TF-IDF"},
}


def make_tfidf_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])


def make_mlp_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=8000)),
        ("svd", TruncatedSVD(n_components=128, random_state=42)),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=30,
                              early_stopping=True, random_state=42)),
    ])


def main():
    data = load_sst5_3class()
    train, val, test = data["train"], data["validation"], data["test"]

    pipelines = {
        "naive_bayes": make_tfidf_pipeline(MultinomialNB(alpha=0.5)),
        "logistic_regression": make_tfidf_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", C=3.0)
        ),
        "svm": make_tfidf_pipeline(
            CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=0.5), cv=3)
        ),
        "neural_mlp": make_mlp_pipeline(),
    }

    rows, fitted, arena_eval = [], {}, {}
    test_texts = test["text"].tolist()

    for name, pipe in pipelines.items():
        pipe.fit(train["text"], train["label"])
        fitted[name] = pipe

        val_preds = pipe.predict(val["text"])
        val_f1 = evaluate_and_save(f"{name}_val", val["label"], val_preds)["f1_macro"]

        test_preds = pipe.predict(test["text"])
        test_probs = pipe.predict_proba(test["text"])
        m = evaluate_and_save(name, test["label"], test_preds)
        m["val_f1_macro"] = val_f1

        roc_info = compute_roc_data(test["label"], test_probs)
        m["roc_auc_macro"] = roc_info["macro_auc"]
        rows.append(m)

        cm = confusion_matrix(test["label"], test_preds, labels=[0, 1, 2]).tolist()
        arena_eval[name] = {
            **MODEL_META[name],
            "key": name,
            "accuracy": round(float(m["accuracy"]), 4),
            "precision_macro": round(float(m["precision_macro"]), 4),
            "recall_macro": round(float(m["recall_macro"]), 4),
            "f1_macro": round(float(m["f1_macro"]), 4),
            "val_f1_macro": round(float(val_f1), 4),
            "roc_auc_macro": roc_info["macro_auc"],
            "confusion_matrix": cm,
            "roc_curves": roc_info["per_class"],
            "cell_samples": collect_confusion_samples(test_texts, test["label"], test_preds, limit=8),
        }

    results = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / "baseline_metrics.csv", index=False)
    ARENA_EVAL_PATH.write_text(json.dumps(arena_eval, indent=2), encoding="utf-8")
    print("\nSummary (test set):\n", results.round(3).to_string(index=False))

    best = results.iloc[0]["model"]
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best], BASELINE_PATH)
    joblib.dump(fitted, ARENA_MODELS_PATH)
    print(f"\nSaved best baseline ({best}) -> {BASELINE_PATH}")
    print(f"Saved 4-model arena bundle -> {ARENA_MODELS_PATH}")
    print(f"Saved evaluation telemetry -> {ARENA_EVAL_PATH}")


if __name__ == "__main__":
    main()
