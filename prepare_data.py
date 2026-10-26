"""
Step 1 - Data preparation for Spam Message Detection using LSTM.

Loads the UCI SMS Spam Collection, cleans and tokenises every message,
builds the vocabulary from the TRAINING split only (no data leakage),
and stores train/validation/test splits plus the vocabulary on disk.

Run:  python prepare_data.py
"""

import json
import os
import random
import re
from collections import Counter

import pandas as pd

SEED = 42
random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "SMSSpamCollection")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
TOKEN_RE = re.compile(r"[a-z$!]+")


def clean_text(text: str) -> str:
    """Lowercase, replace URLs/phone numbers with placeholders, keep $ and !
    because they carry strong spam signals."""
    text = text.lower()
    text = URL_RE.sub(" <url> ", text)
    text = re.sub(r"\d{7,}", " <num> ", text)   # long digit runs (phone numbers)
    text = re.sub(r"\d+", " <num> ", text)      # remaining numbers
    tokens = TOKEN_RE.findall(text)
    return " ".join(tokens)


def build_vocab(texts, min_freq=2, max_size=8000):
    """Vocabulary: 0 = <pad>, 1 = <unk>, then words sorted by frequency."""
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


def main():
    # ---- load ------------------------------------------------------------
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    df["label_code"] = df["label"].map({"ham": 0, "spam": 1})
    print(f"Loaded {len(df)} messages  "
          f"(ham = {(df.label == 'ham').sum()}, spam = {(df.label == 'spam').sum()}, "
          f"spam ratio = {(df.label == 'spam').mean():.1%})")

    # ---- clean + tokenise ------------------------------------------------
    df["clean"] = df["message"].apply(clean_text)
    df["n_tokens"] = df["clean"].str.split().str.len()

    stats = {
        "n_messages": int(len(df)),
        "n_ham": int((df.label == "ham").sum()),
        "n_spam": int((df.label == "spam").sum()),
        "spam_ratio": round(float((df.label == "spam").mean()), 4),
        "mean_tokens_ham": round(float(df[df.label == "ham"].n_tokens.mean()), 2),
        "mean_tokens_spam": round(float(df[df.label == "spam"].n_tokens.mean()), 2),
        "p95_tokens": int(df.n_tokens.quantile(0.95)),
        "p99_tokens": int(df.n_tokens.quantile(0.99)),
        "max_tokens": int(df.n_tokens.max()),
    }
    print(f"Message length (tokens): mean ham = {stats['mean_tokens_ham']}, "
          f"mean spam = {stats['mean_tokens_spam']}, "
          f"95th pct = {stats['p95_tokens']}, 99th pct = {stats['p99_tokens']}, "
          f"max = {stats['max_tokens']}")

    # ---- stratified split 70 / 15 / 15 ------------------------------------
    idx_all = list(range(len(df)))
    ham_idx = [i for i in idx_all if df.label_code[i] == 0]
    spam_idx = [i for i in idx_all if df.label_code[i] == 1]
    rng = random.Random(SEED)
    rng.shuffle(ham_idx)
    rng.shuffle(spam_idx)

    def split(indices):
        n = len(indices)
        a = int(0.70 * n)
        b = int(0.85 * n)
        return indices[:a], indices[a:b], indices[b:]

    ham_tr, ham_va, ham_te = split(ham_idx)
    sp_tr, sp_va, sp_te = split(spam_idx)

    train_idx = sorted(ham_tr + sp_tr)
    val_idx = sorted(ham_va + sp_va)
    test_idx = sorted(ham_te + sp_te)

    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sub = df.iloc[idxs]
        sub[["label", "label_code", "clean", "n_tokens"]].to_csv(
            os.path.join(OUT_DIR, f"{name}.csv"), index=False)
        print(f"{name:>5}: {len(sub):5d} messages  "
              f"({int(sub.label_code.sum())} spam, {len(sub) - int(sub.label_code.sum())} ham)")

    # ---- vocabulary from TRAIN only ---------------------------------------
    vocab = build_vocab(df.iloc[train_idx].clean)
    vocab_size = len(vocab)
    train_tokens = [tok for t in df.iloc[train_idx].clean for tok in t.split()]
    known = sum(1 for tok in train_tokens if tok in vocab)
    print(f"Vocabulary size = {vocab_size} (min_freq = 2), "
          f"train-token coverage = {known / len(train_tokens):.1%}")

    with open(os.path.join(OUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    stats["vocab_size"] = vocab_size
    stats["train_size"] = len(train_idx)
    stats["val_size"] = len(val_idx)
    stats["test_size"] = len(test_idx)
    stats["test_spam"] = int(df.iloc[test_idx].label_code.sum())
    with open(os.path.join(OUT_DIR, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved outputs/: train.csv, val.csv, test.csv, vocab.json, data_stats.json")


if __name__ == "__main__":
    main()
