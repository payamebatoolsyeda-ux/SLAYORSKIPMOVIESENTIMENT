"""Fine-tune DistilBERT (or any HF encoder) for 3-class sentiment.

Run:  python -m src.train_bert --epochs 3
GPU strongly recommended (a free Colab/Kaggle GPU works; copy models/bert back afterwards).
"""
import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from .config import BERT_DIR, ID2LABEL, LABEL2ID, MODELS_DIR, REPORTS_DIR
from .data import load_sst5_3class
from .evaluate import evaluate_and_save


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    data = load_sst5_3class()
    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True)

    train_ds, val_ds, test_ds = (to_ds(data[s]) for s in ("train", "validation", "test"))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)

    targs = TrainingArguments(
        output_dir=str(MODELS_DIR / "bert_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()

    out = trainer.predict(test_ds)
    metrics = evaluate_and_save("bert", out.label_ids, np.argmax(out.predictions, axis=-1))
    REPORTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "bert_metrics.csv", index=False)

    BERT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(BERT_DIR))
    tok.save_pretrained(str(BERT_DIR))
    print(f"Saved fine-tuned model -> {BERT_DIR}")


if __name__ == "__main__":
    main()
