"""
Step 2 - LSTM model definition and training for Spam Message Detection.

Architecture:
    Embedding(vocab, 100, padding_idx=0)
    -> LSTM(100, 128, batch_first=True)   [final hidden state of the real
        sequence via pack_padded_sequence, so padding never leaks in]
    -> Dropout(0.5)
    -> Linear(128, 1)                     [logit, trained with BCEWithLogitsLoss,
        pos_weight compensates the 13.4% spam class imbalance]

Run:  python train_lstm.py
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

MAX_LEN = 50          # covers the 99th percentile of message length
EMBED_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3


class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.encoded = [encode(t, vocab) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        seq = self.encoded[i][:MAX_LEN]
        length = max(len(seq), 1)
        seq = seq + [0] * (MAX_LEN - len(seq))     # pad to MAX_LEN
        return (torch.tensor(seq, dtype=torch.long),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.float))


def encode(text, vocab):
    return [vocab.get(tok, 1) for tok in text.split()]


class SpamLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        out = self.dropout(h_n[-1])
        return self.fc(out).squeeze(1)             # logits


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.set_grad_enabled(training):
        for x, lengths, y in loader:
            logits = model(x, lengths)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            probs = torch.sigmoid(logits).detach()
            correct += ((probs >= 0.5) == y.bool()).sum().item()
            n += len(y)
            all_probs.extend(probs.tolist())
            all_labels.extend(y.tolist())
    return total_loss / n, correct / n, all_probs, all_labels


def f1_score_manual(labels, probs, thr=0.5):
    tp = sum(1 for p, l in zip(probs, labels) if p >= thr and l == 1)
    fp = sum(1 for p, l in zip(probs, labels) if p >= thr and l == 0)
    fn = sum(1 for p, l in zip(probs, labels) if p < thr and l == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main():
    vocab = json.load(open(os.path.join(OUT_DIR, "vocab.json")))
    train_df = pd.read_csv(os.path.join(OUT_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(OUT_DIR, "val.csv"))
    # empty cleaned messages come back from CSV as NaN -> restore ""
    train_df["clean"] = train_df.clean.fillna("").astype(str)
    val_df["clean"] = val_df.clean.fillna("").astype(str)

    train_ds = SMSDataset(train_df.clean.tolist(), train_df.label_code.tolist(), vocab)
    val_ds = SMSDataset(val_df.clean.tolist(), val_df.label_code.tolist(), vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SpamLSTM(len(vocab))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Embedding({len(vocab)}, {EMBED_DIM}) -> LSTM({HIDDEN_DIM}) "
          f"-> Dropout(0.5) -> Linear(128, 1)")
    print(f"Trainable parameters: {n_params:,}")

    # pos_weight = n_negative / n_positive rebalances the 13.4% spam minority
    n_pos = float((train_df.label_code == 1).sum())
    n_neg = float((train_df.label_code == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos])
    print(f"class imbalance handled with pos_weight = {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, va_probs, va_labels = run_epoch(model, val_loader, criterion)
        _, _, va_f1 = f1_score_manual(va_labels, va_probs)
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "train_acc": round(tr_acc, 4), "val_loss": round(va_loss, 4),
                        "val_acc": round(va_acc, 4), "val_f1": round(va_f1, 4)})
        marker = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"model_state": model.state_dict(),
                        "vocab_size": len(vocab)}, os.path.join(OUT_DIR, "model_best.pt"))
            marker = "  <- saved (best val F1)"
        print(f"Epoch {epoch:2d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} F1 {va_f1:.4f}{marker}")

    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training finished. Best validation F1 = {best_f1:.4f}. "
          f"Model saved to outputs/model_best.pt")


if __name__ == "__main__":
    main()
