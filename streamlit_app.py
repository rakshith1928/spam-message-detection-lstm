"""
Spam Message Detection using LSTM - Streamlit Demo (Showcase Component)
========================================================================
Loads the trained PyTorch LSTM from outputs/model_best.pt together with the
vocabulary saved by prepare_data.py and classifies any SMS message typed by
the user, live in the browser.

Usage:
    streamlit run streamlit_app.py

Requirements (in addition to the batch pipeline):
    streamlit
"""

import json
import os
import re

import streamlit as st
import torch

from prepare_data import clean_text
from predict_demo import DEMO_MESSAGES
from train_lstm import MAX_LEN, OUT_DIR, SMSDataset, SpamLSTM

SPAM_THRESHOLD = 0.5
# lexical back-off: the UCI corpus is 2007 UK SMS, so contemporary Indian
# money phrasing ("you have won ... lakh rupees") tokenises poorly and scores
# near 0; a tiny cue layer catches wording the corpus never saw.
ACTION_RE = re.compile(
    r"\b(?:won(?!['’]?t\b)|win(?:n(?:er|ing))?\b|prize|claimed?|"
    r"congratulations|congrats|urgent(?:ly)?|free\b|jackpot|lottery|"
    r"guaranteed|click\b|<url>)")
MONEY_RE = re.compile(
    r"\b(?:rupees|rs\b|lakh|crore|paise|cash|gift|prize|jackpot)\b|\$|£")


def spam_policy(text, spam_prob):
    low = text.lower()
    action = {m.group(0) for m in ACTION_RE.finditer(low)}
    money = {m.group(0) for m in MONEY_RE.finditer(low)}
    signals = sorted(action | money)
    is_spam = (spam_prob >= SPAM_THRESHOLD
               or (bool(action) and bool(money))
               or len(signals) >= 4)
    return is_spam, signals

st.set_page_config(
    page_title="Spam Message Detection - LSTM",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Spam Message Detection - PyTorch LSTM")
st.caption(
    "Trained on the UCI SMS Spam Collection (5,572 messages). "
    "Type any message below and the model classifies it - the LSTM has "
    "never seen these exact words in this order before."
)


@st.cache_resource
def load_model():
    vocab = json.load(open(os.path.join(OUT_DIR, "vocab.json"),
                           encoding="utf-8"))
    ckpt = torch.load(os.path.join(OUT_DIR, "model_best.pt"),
                      map_location="cpu", weights_only=True)
    model = SpamLSTM(ckpt["vocab_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return vocab, model


@st.cache_data
def load_metrics():
    return json.load(open(os.path.join(OUT_DIR, "metrics.json"),
                          encoding="utf-8"))


try:
    vocab, model = load_model()
    test_metrics = load_metrics()["lstm"]
except Exception as exc:  # noqa: BLE001
    st.error(
        "Could not load the trained model. Run the pipeline first "
        "(python run_all.py) - it creates outputs/model_best.pt."
    )
    st.exception(exc)
    st.stop()


def predict_probability(text):
    clean = clean_text(text)
    ds = SMSDataset([clean], [0], vocab)
    x, length, _ = ds[0]
    with torch.no_grad():
        return float(torch.sigmoid(model(x.unsqueeze(0), length.unsqueeze(0))))


c1, c2, c3, c4 = st.columns(4)
c1.metric("Test Accuracy", f"{test_metrics['accuracy']:.1%}")
c2.metric("Spam F1 Score", f"{test_metrics['f1']:.3f}")
c3.metric("Precision", f"{test_metrics['precision']:.3f}")
c4.metric("Recall", f"{test_metrics['recall']:.3f}")

st.divider()


# ------------------------------------------------------------------ #
# Live inference
# ------------------------------------------------------------------ #
st.subheader("✍️ Try it yourself")

if "msg" not in st.session_state:
    st.session_state.msg = ""

sample = DEMO_MESSAGES[0][0][:24] + "..."
for row in range(2):
    cols = st.columns(4)
    for j in range(4):
        msg_text = DEMO_MESSAGES[row * 4 + j][0]
        if cols[j].button(msg_text[:24] + ("..." if len(msg_text) > 24 else ""),
                          key=f"demo{row * 4 + j}",
                          help=msg_text):
            st.session_state.msg = msg_text

text = st.text_area(
    "SMS message",
    key="msg",
    height=120,
    placeholder="Paste a message here, or click one of the samples above...",
)

if st.button("Classify message", type="primary"):
    if not text.strip():
        st.warning("Type or paste a message first.")
    else:
        spam_prob = predict_probability(text)
        n_tokens = min(len(clean_text(text).split()), MAX_LEN)
        is_spam, signals = spam_policy(text, spam_prob)
        if is_spam:
            st.error(f"🚨 SPAM  -  P(spam) = {spam_prob:.1%}")
        else:
            st.success(f"✅ HAM (genuine message)  -  P(spam) = {spam_prob:.1%}")
        if signals:
            st.caption(f"Detected signals: {', '.join(signals)} | model "
                       f"P(spam)={spam_prob:.1%} at threshold {SPAM_THRESHOLD}")
        st.progress(spam_prob, text=f"P(spam) = {spam_prob:.1%} "
                                    f"| {n_tokens} of {MAX_LEN} token slots used")

st.divider()


# ------------------------------------------------------------------ #
# How the model works
# ------------------------------------------------------------------ #
st.subheader("📐 How the model works")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        "Preprocessing\n"
        "- URLs and digit runs become `<url>` / `<num>` placeholders\n"
        "- lowercased, tokenised keeping only letters, `$` and `!`\n"
        "- padded to 50 tokens; long messages are truncated accurately"
    )
with col2:
    st.markdown(
        "LSTM architecture\n"
        "- Embedding(vocab, 100) -> LSTM(100, 128) -> Dropout(0.5) -> Linear\n"
        "- only the final hidden state of the true sequence is used\n"
        "- sigmoid > 0.5 verdict; trained with BCE + pos_weight for imbalance"
    )

st.info(
    "The Naive Bayes baseline reaches 97.97% accuracy vs the LSTM's 98.33% - "
    "deep learning beats the classical benchmark on the same split."
)
st.caption(
    "Deployed policy note: sentences using phrasing outside the 2007 UK corpus "
    "(e.g. 'you have won ... lakh rupees') also trigger a lexical spam-signal "
    "back-off on top of the LSTM itself - both signals and model probability "
    "are shown above."
)

st.divider()
st.caption(
    "Spam Message Detection using LSTM | DL Mini Project | Streamlit Demo"
)
