"""
Step 3 - Evaluation on the held-out test set + Naive Bayes baseline.

Reports accuracy, precision, recall, F1, ROC-AUC and the confusion matrix
for the LSTM, and trains a TF-IDF + Multinomial Naive Bayes baseline on the
same train split for comparison.

Run:  python evaluate.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from train_lstm import SMSDataset, SpamLSTM, f1_score_manual, OUT_DIR, MAX_LEN


def main():
    torch.manual_seed(42)
    vocab = json.load(open(os.path.join(OUT_DIR, "vocab.json")))
    test_df = pd.read_csv(os.path.join(OUT_DIR, "test.csv"))
    test_df["clean"] = test_df.clean.fillna("").astype(str)

    # ---------------- LSTM on test set ----------------
    ckpt = torch.load(os.path.join(OUT_DIR, "model_best.pt"), weights_only=True)
    model = SpamLSTM(ckpt["vocab_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = SMSDataset(test_df.clean.tolist(), test_df.label_code.tolist(), vocab)
    probs = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, length, _ = test_ds[i]
            probs.append(float(torch.sigmoid(model(x.unsqueeze(0), length.unsqueeze(0)))))
    preds = [int(p >= 0.5) for p in probs]
    labels = test_df.label_code.tolist()

    acc = float(np.mean(np.array(preds) == np.array(labels)))
    prec, rec, f1 = f1_score_manual(labels, probs)
    auc = float(roc_auc_score(labels, probs))

    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)

    print("=" * 62)
    print(f"LSTM RESULTS ON TEST SET ({len(test_df)} messages)")
    print("=" * 62)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}   (of predicted spam, how much really spam)")
    print(f"Recall   : {rec:.4f}   (of real spam, how much was caught)")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("-" * 62)
    print("Confusion matrix          Predicted")
    print("                        ham      spam")
    print(f"Actual  ham     {tn:6d}   {fp:6d}")
    print(f"Actual  spam    {fn:6d}   {tp:6d}")

    fpr, tpr, _ = roc_curve(labels, probs)
    np.save(os.path.join(OUT_DIR, "roc_points.npy"),
            np.vstack([fpr, tpr]))
    lstm_metrics = {"accuracy": round(acc, 4), "precision": round(prec, 4),
                    "recall": round(rec, 4), "f1": round(f1, 4), "roc_auc": round(auc, 4),
                    "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}

    # ---------------- Naive Bayes baseline ----------------
    train_df = pd.read_csv(os.path.join(OUT_DIR, "train.csv"))
    train_df["clean"] = train_df.clean.fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tr = tfidf.fit_transform(train_df.clean)
    X_te = tfidf.transform(test_df.clean)
    nb = MultinomialNB()
    nb.fit(X_tr, train_df.label_code)
    nb_preds = nb.predict(X_te)
    nb_probs = nb.predict_proba(X_te)[:, 1]
    nb_acc = float(np.mean(nb_preds == np.array(labels)))
    nb_prec, nb_rec, nb_f1 = f1_score_manual(labels, nb_probs.tolist())
    nb_auc = float(roc_auc_score(labels, nb_probs))
    nb_fpr, nb_tpr, _ = roc_curve(labels, nb_probs)
    np.save(os.path.join(OUT_DIR, "roc_points_nb.npy"), np.vstack([nb_fpr, nb_tpr]))

    print("-" * 62)
    print("BASELINE: TF-IDF + Multinomial Naive Bayes (same splits)")
    print(f"Accuracy : {nb_acc:.4f}  Precision: {nb_prec:.4f}  "
          f"Recall: {nb_rec:.4f}  F1: {nb_f1:.4f}  ROC-AUC: {nb_auc:.4f}")
    print("=" * 62)

    metrics = {"lstm": lstm_metrics,
               "naive_bayes": {"accuracy": round(nb_acc, 4), "precision": round(nb_prec, 4),
                               "recall": round(nb_rec, 4), "f1": round(nb_f1, 4),
                               "roc_auc": round(nb_auc, 4)}}
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    test_df = test_df.copy()
    test_df["probability"] = [round(p, 4) for p in probs]
    test_df["predicted"] = preds
    test_df["correct"] = (test_df.predicted == test_df.label_code).astype(int)
    test_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
    print("Saved outputs/: metrics.json, roc_points.npy, test_predictions.csv")


if __name__ == "__main__":
    main()
