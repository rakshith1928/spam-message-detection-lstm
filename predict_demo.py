"""
Step 4 - Live demo: classify brand-new, hand-written messages with the
trained LSTM. This is the showcase used in the report's Results section.

Run:  python predict_demo.py
"""

import json
import os

import torch

from train_lstm import SMSDataset, SpamLSTM, OUT_DIR


DEMO_MESSAGES = [
    ("Congratulations! You've WON a $1,000 Walmart gift card. "
     "Go to http://bit.ly/claimnow to receive your prize!", None),
    ("URGENT! Your mobile number has won £2,000 in our monthly draw. "
     "Call 09061234567 now to claim. T&C apply", None),
    ("Free entry in the weekly contest to win an iPhone 15. "
     "Text WIN to 81212 now!", None),
    ("Hey, are we still meeting at the library at 5 pm for the project?", None),
    ("I'll call you after class. Mom is making dinner tonight.", None),
    ("Happy birthday! Have an awesome day, see you at the party.", None),
    ("Your lecture timetable for the 6th semester is attached.", None),
    ("You have 1 new voicemail. Click <url> to listen to your message now!", None),
]


def main():
    torch.manual_seed(42)
    vocab = json.load(open(os.path.join(OUT_DIR, "vocab.json")))
    ckpt = torch.load(os.path.join(OUT_DIR, "model_best.pt"), weights_only=True)
    model = SpamLSTM(ckpt["vocab_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # reuse the exact cleaning rule from prepare_data without importing it
    from prepare_data import clean_text

    print("=" * 74)
    print("LIVE DEMO - messages the model has never seen during training")
    print("=" * 74)
    print(f"{'P(spam)':>8}  {'Verdict':>7}   Message")
    print("-" * 74)
    for text, _ in DEMO_MESSAGES:
        clean = clean_text(text)
        ds = SMSDataset([clean], [0], vocab)
        x, length, _ = ds[0]
        with torch.no_grad():
            p = float(torch.sigmoid(model(x.unsqueeze(0), length.unsqueeze(0))))
        verdict = "SPAM" if p >= 0.5 else "HAM"
        short = text if len(text) <= 56 else text[:53] + "..."
        print(f"{p:8.4f}  {verdict:>7}   {short}")
    print("=" * 74)


if __name__ == "__main__":
    main()
