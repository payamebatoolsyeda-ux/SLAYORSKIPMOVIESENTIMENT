from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)
from sklearn.preprocessing import label_binarize

from .config import LABELS, REPORTS_DIR


def evaluate_and_save(name: str, y_true, y_pred) -> dict:
    """Prints + saves report and confusion matrix; returns summary metrics."""
    REPORTS_DIR.mkdir(exist_ok=True)
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABELS,
                                   digits=3, zero_division=0)
    print(f"\n=== {name} ===\n{report}")
    (REPORTS_DIR / f"{name}_classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2],
                                                 average="macro", zero_division=0)
    return {"model": name, "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": p, "recall_macro": r, "f1_macro": f}


def compute_roc_data(y_true, y_prob: np.ndarray, max_points: int = 45) -> dict:
    """Compute one-vs-rest ROC curves and macro AUC for 3-class predictions."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    curves = {}
    aucs = []
    for idx, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        score = float(auc(fpr, tpr))
        aucs.append(score)
        if len(fpr) > max_points:
            sel = np.linspace(0, len(fpr) - 1, max_points, dtype=int)
            fpr, tpr = fpr[sel], tpr[sel]
        curves[label] = {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
            "auc": round(score, 4),
        }
    return {"per_class": curves, "macro_auc": round(float(np.mean(aucs)), 4)}


def collect_confusion_samples(texts: List[str], y_true, y_pred, limit: int = 8) -> Dict[str, List[str]]:
    """Collect sample texts for every (true, pred) cell of the 3x3 confusion matrix."""
    buckets: Dict[str, List[str]] = {f"{i}:{j}": [] for i in range(3) for j in range(3)}
    for text, yt, yp in zip(texts, y_true, y_pred):
        key = f"{int(yt)}:{int(yp)}"
        if len(buckets[key]) < limit:
            buckets[key].append(str(text))
    return buckets
