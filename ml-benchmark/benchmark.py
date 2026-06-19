#!/usr/bin/env python3
"""LAB 16 - Part 7: LightGBM benchmark on Credit Card Fraud Detection (CPU plan)."""
import os, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score)

print("=" * 64)
print("  LightGBM Benchmark - Credit Card Fraud Detection (r5.xlarge CPU)")
print("=" * 64)

# 1. Locate dataset (download via kagglehub if not cached)
import kagglehub
ds_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv = os.path.join(ds_path, "creditcard.csv")
print(f"Dataset: {csv}")

# 2. Load data
t0 = time.time()
df = pd.read_csv(csv)
load_time = time.time() - t0
print(f"Rows: {len(df):,}  Cols: {df.shape[1]}  "
      f"Fraud: {int(df['Class'].sum())} ({df['Class'].mean()*100:.3f}%)")

X = df.drop(columns=["Class"])
y = df["Class"]
# 3-way split: train (64%) / validation (16%) / test (20%).
# Validation drives early stopping so the test set stays untouched (no leakage).
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.2, random_state=42, stratify=y_tmp)

# 3. Train LightGBM. AUC handles the 0.17% imbalance well; we keep the
# natural class ratio so the 0.5 threshold stays well-calibrated (good precision).
params = {"objective": "binary", "metric": "auc", "boosting_type": "gbdt",
          "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9,
          "bagging_fraction": 0.8, "bagging_freq": 1,
          "min_child_samples": 20, "verbose": -1, "n_jobs": -1}
t0 = time.time()
model = lgb.train(
    params, lgb.Dataset(X_train, label=y_train), num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
train_time = time.time() - t0
best_iter = model.best_iteration

# 4. Evaluate
y_prob = model.predict(X_test, num_iteration=best_iter)
y_pred = (y_prob >= 0.5).astype(int)
auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

# 5. Inference latency / throughput
one = X_test.iloc[:1]
model.predict(one)  # warm-up
N = 200
t0 = time.time()
for _ in range(N):
    model.predict(one)
lat_1 = (time.time() - t0) / N * 1000  # ms per single-row prediction

batch = X_test.iloc[:1000]
t0 = time.time()
model.predict(batch)
thr_time = time.time() - t0
throughput = 1000 / thr_time

# 6. Report
print("\n" + "=" * 64)
print(f"  {'Metric':<30}{'Value'}")
print("-" * 64)
rows = [
    ("Load data time", f"{load_time:.2f} s"),
    ("Training time", f"{train_time:.2f} s"),
    ("Best iteration", f"{best_iter}"),
    ("AUC-ROC", f"{auc:.5f}"),
    ("Accuracy", f"{acc:.5f}"),
    ("F1-Score", f"{f1:.5f}"),
    ("Precision", f"{prec:.5f}"),
    ("Recall", f"{rec:.5f}"),
    ("Inference latency (1 row)", f"{lat_1:.3f} ms"),
    ("Throughput (1000 rows)", f"{throughput:,.0f} rows/s ({thr_time*1000:.1f} ms)"),
]
for k, v in rows:
    print(f"  {k:<30}{v}")
print("=" * 64)

result = {
    "instance_type": "r5.xlarge", "vcpu": os.cpu_count(),
    "rows": int(len(df)), "fraud_rate": round(float(df["Class"].mean()), 6),
    "load_time_s": round(load_time, 3), "train_time_s": round(train_time, 3),
    "best_iteration": int(best_iter), "auc_roc": round(float(auc), 5),
    "accuracy": round(float(acc), 5), "f1": round(float(f1), 5),
    "precision": round(float(prec), 5), "recall": round(float(rec), 5),
    "inference_latency_1row_ms": round(lat_1, 4),
    "throughput_1000rows_per_s": round(throughput, 1),
}
out = os.path.expanduser("~/ml-benchmark/benchmark_result.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved metrics -> {out}")
