#!/usr/bin/env python3
"""
compute_qnn_stats.py
Performs Welch t-test, Cohen's d, and bootstrap CI
for QNN vs MLP RMSE/MAE/R2 metrics.
"""
import pandas as pd, numpy as np, os
from scipy import stats

os.makedirs("stats", exist_ok=True)

perf = pd.read_csv("metrics/performance_summary.csv")

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_sd = np.sqrt(((nx-1)*x.var() + (ny-1)*y.var()) / dof)
    return (x.mean() - y.mean()) / pooled_sd

def bootstrap_ci(a, b, func=np.mean, nboot=10000, alpha=0.05):
    diffs = []
    for _ in range(nboot):
        sa = np.random.choice(a, len(a), replace=True)
        sb = np.random.choice(b, len(b), replace=True)
        diffs.append(func(sa) - func(sb))
    lo, hi = np.percentile(diffs, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

metrics = ["RMSE", "MAE", "R2"]
summary = []

for metric in metrics:
    qnn = perf.loc[perf["model"]=="QNN", metric].dropna()
    mlp = perf.loc[perf["model"]=="MLP", metric].dropna()
    t, p = stats.ttest_ind(qnn, mlp, equal_var=False)
    d = cohen_d(qnn, mlp)
    ci_lo, ci_hi = bootstrap_ci(qnn, mlp)
    summary.append({
        "metric": metric,
        "mean_QNN": qnn.mean(),
        "mean_MLP": mlp.mean(),
        "p_value": p,
        "cohen_d": d,
        "bootstrap_CI_low": ci_lo,
        "bootstrap_CI_high": ci_hi
    })

df = pd.DataFrame(summary)
out_path = "stats/significance_summary.csv"
df.to_csv(out_path, index=False)
print("✅ Wrote", out_path)
print(df.round(4))
