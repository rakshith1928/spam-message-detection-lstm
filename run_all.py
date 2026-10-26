"""
Run the complete Spam Message Detection (LSTM) pipeline in order:
    [1/5] prepare_data.py  - clean, split, build vocabulary
    [2/5] train_lstm.py    - train the LSTM and keep the best model
    [3/5] evaluate.py      - test-set metrics + Naive Bayes baseline
    [4/5] predict_demo.py  - classify brand-new messages
    [5/5] visualise.py     - generate all report charts

Run:  python run_all.py
"""

import subprocess
import sys
import time

STEPS = ["prepare_data.py", "train_lstm.py", "evaluate.py",
         "predict_demo.py", "visualise.py"]


def main():
    total = time.time()
    for i, script in enumerate(STEPS, 1):
        print(flush=True)
        print(f"================ [ {i}/5 ] {script} ================", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"!!! {script} failed with exit code {result.returncode}", flush=True)
            sys.exit(result.returncode)
        print(f"--- {script} finished in {time.time() - t0:.1f}s ---", flush=True)
    print(flush=True)
    print(f"PIPELINE COMPLETE - total time {time.time() - total:.1f}s", flush=True)
    print("All outputs are in the outputs/ folder.", flush=True)


if __name__ == "__main__":
    main()
