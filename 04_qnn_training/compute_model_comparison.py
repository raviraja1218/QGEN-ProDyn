#!/usr/bin/env python3
"""
compute_model_comparison.py

- Loads QNN CV RMSEs and MLP baseline per-fold RMSEs (dataset or original units).
- Computes Welch's t-test, Cohen's d, bootstrap 95% CI for difference in RMSE.
- Saves:
  - stats/significance_summary.csv
  - metrics/table2_performance.csv
  - figures/figure_3b_parity_vs_baseline.png
"""

import os, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MET = ROOT / "metrics"
STATS = ROOT / "stats"
FIGS = ROOT / "figures"
STATS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
MET.mkdir(exist_ok=True)

# 1) Load QNN RMSE original-units if present, else dataset-units
qnn_orig = MET / "qnn_cv_rmse_original_units.txt"
qnn_ds   = MET / "qnn_cv_rmse_dataset_units.txt"
if qnn_orig.exists():
    qnn = np.loadtxt(qnn_orig)
    unit_note = "original"
elif qnn_ds.exists():
    qnn = np.loadtxt(qnn_ds)
    unit_note = "dataset"
else:
    raise FileNotFoundError("QNN RMSE file not found in metrics/ (qnn_cv_rmse_*.txt)")

# 2) Load MLP per-fold RMSEs
# try metrics/performance_summary.csv (fold rows) or metrics/perf_mlp_baseline.csv
mlp_csv1 = MET / "performance_summary.csv"
mlp_csv2 = MET / "perf_mlp_baseline.csv"
mlp = None
if mlp_csv1.exists():
    df = pd.read_csv(mlp_csv1)
    # Try to find MLP rows; expect 'model' column
    if 'model' in df.columns and 'RMSE' in df.columns:
        mlp = df[df['model'].str.contains('MLP', case=False, na=False)]['RMSE'].values
    else:
        # fallback: if csv is simply folds x metrics with model column absent, assume second column RMSE for MLP
        try:
            mlp = df['RMSE'].values
        except Exception:
            mlp = None
if mlp is None and mlp_csv2.exists():
    df = pd.read_csv(mlp_csv2)
    if 'RMSE' in df.columns:
        mlp = df['RMSE'].values

if mlp is None:
    # try to read models/performance_summary.csv from parent dir
    raise FileNotFoundError("Could not find per-fold MLP RMSEs in metrics/. Please provide performance_summary.csv or perf_mlp_baseline.csv with 'RMSE' and 'model' columns.")

# Ensure same length: if mlp and qnn lengths differ, try to match smallest or error
if len(mlp) != len(qnn):
    print("[warn] different fold counts: mlp:", len(mlp), "qnn:", len(qnn), " -> using min length and truncating")
    L = min(len(mlp), len(qnn))
    mlp = mlp[:L]
    qnn = qnn[:L]

# 3) Basic summary
summary = {
    "n_folds": int(len(qnn)),
    "qnn_mean": float(np.mean(qnn)),
    "qnn_se": float(np.std(qnn, ddof=1)/np.sqrt(len(qnn))),
    "mlp_mean": float(np.mean(mlp)),
    "mlp_se": float(np.std(mlp, ddof=1)/np.sqrt(len(mlp))),
    "units": unit_note
}

# 4) Welch t-test
tstat, pval = stats.ttest_ind(qnn, mlp, equal_var=False)
summary.update({"welch_t": float(tstat), "p_value": float(pval)})

# 5) Cohen's d (for unequal ns use pooled sd approx)
def cohens_d(a,b):
    na, nb = len(a), len(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    # pooled sd
    s = np.sqrt(((na-1)*sa + (nb-1)*sb) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / s

d = cohens_d(qnn, mlp)
summary["cohen_d"] = float(d)

# 6) Bootstrap 95% CI for difference (qnn - mlp)
rng = np.random.default_rng(12345)
nboot = 20000
diffs = []
for _ in range(nboot):
    a = rng.choice(qnn, size=len(qnn), replace=True)
    b = rng.choice(mlp, size=len(mlp), replace=True)
    diffs.append(np.mean(a) - np.mean(b))
lo, hi = np.percentile(diffs, [2.5, 97.5])
summary["bootstrap_diff_mean"] = float(np.mean(diffs))
summary["bootstrap_CI_low"] = float(lo)
summary["bootstrap_CI_high"] = float(hi)

# 7) Write stats CSV
stats_df = pd.DataFrame([{
    "metric":"RMSE",
    "mean_QNN": summary["qnn_mean"],
    "se_QNN": summary["qnn_se"],
    "mean_MLP": summary["mlp_mean"],
    "se_MLP": summary["mlp_se"],
    "p_value": summary["p_value"],
    "cohen_d": summary["cohen_d"],
    "bootstrap_CI_low": summary["bootstrap_CI_low"],
    "bootstrap_CI_high": summary["bootstrap_CI_high"],
    "units": summary["units"]
}])
STATS.mkdir(exist_ok=True)
stats_df.to_csv(STATS / "significance_summary.csv", index=False)
print("[info] Wrote", STATS / "significance_summary.csv")

# 8) Create table2 CSV with per-fold RMSEs
folds = np.arange(1, len(qnn)+1)
table = pd.DataFrame({
    "fold": folds,
    "qnn_rmse": qnn,
    "mlp_rmse": mlp
})
(MET / "table2_performance.csv").write_text(table.to_csv(index=False))
print("[info] Wrote", MET / "table2_performance.csv")

# 9) Parity + bar figure
plt.figure(figsize=(8,4))
ax1 = plt.subplot(1,2,1)
ax1.scatter(mlp, qnn, s=60)
mn = min(min(mlp), min(qnn)); mx = max(max(mlp), max(qnn))
ax1.plot([mn,mx],[mn,mx], color='k', lw=0.8, linestyle='--')
ax1.set_xlabel("MLP RMSE ({})".format(unit_note))
ax1.set_ylabel("QNN RMSE ({})".format(unit_note))
ax1.set_title("Per-fold parity (QNN vs MLP)")

ax2 = plt.subplot(1,2,2)
means = [np.mean(mlp), np.mean(qnn)]
errs = [np.std(mlp, ddof=1)/np.sqrt(len(mlp)), np.std(qnn, ddof=1)/np.sqrt(len(qnn))]
ax2.bar([0,1], means, yerr=errs, capsize=5)
ax2.set_xticks([0,1]); ax2.set_xticklabels(["MLP","QNN"])
ax2.set_ylabel("RMSE ({})".format(unit_note))
ax2.set_title("Mean RMSE ± SE")

plt.tight_layout()
fig_path = FIGS / "figure_3b_parity_vs_baseline.png"
plt.savefig(fig_path, dpi=300)
print("[info] Wrote", fig_path)

# 10) human-readable summary
print("\nSUMMARY")
print("QNN mean RMSE ({u}): {qm:.3f} ± {qse:.3f}".format(u=unit_note, qm=summary["qnn_mean"], qse=summary["qnn_se"]))
print("MLP mean RMSE ({u}): {mm:.3f} ± {mse:.3f}".format(u=unit_note, mm=summary["mlp_mean"], mse=summary["mlp_se"]))
print("Welch t-test p-value:", summary["p_value"])
print("Cohen's d:", summary["cohen_d"])
print("Bootstrap mean diff (QNN-MLP): {:.4f}, 95% CI [{:.4f}, {:.4f}]".format(summary["bootstrap_diff_mean"], summary["bootstrap_CI_low"], summary["bootstrap_CI_high"]))

# done
