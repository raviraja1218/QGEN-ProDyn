#!/usr/bin/env python3
"""
compute_param_efficiency.py  (fixed)

Reads stats/param_counts.csv and computes:
 - mean params per model family (MLP, QNN_CORE, HYBRID_QNN)
 - ratio and compression factor
Writes:
 - stats/param_efficiency_summary.csv
 - appends param summary rows to stats/significance_summary.csv (creates backup)

Usage:
  cd ~/QGEN-ProDyn/04_qnn_training
  conda activate qgen-prodyn
  python compute_param_efficiency.py
"""
import os, sys, shutil
import pandas as pd
import numpy as np

ROOT = os.path.dirname(__file__) or "."
STATS_DIR = os.path.join(ROOT, "stats")
os.makedirs(STATS_DIR, exist_ok=True)

param_csv = os.path.join(STATS_DIR, "param_counts.csv")
if not os.path.exists(param_csv):
    print("Error: expected", param_csv, "to exist.")
    sys.exit(1)

df = pd.read_csv(param_csv)

# Normalise model filenames -> families
def family_from_name(name):
    n = str(name).lower()
    if "mlp" in n and "hybrid" not in n:
        return "MLP"
    if "hybrid" in n or "hybrid_qnn" in n:
        return "HYBRID_QNN"
    # qnn core (avoid catching hybrid)
    if "qnn" in n and "hybrid" not in n:
        return "QNN_CORE"
    # fallback: categorize by known prefixes
    if "mlp" in n:
        return "MLP"
    return "OTHER"

df['family'] = df['model_file'].apply(family_from_name)

# Aggregate stats
summary = df.groupby('family', as_index=False)['params'].agg(['mean','median','count']).reset_index()
# Rename to friendly names (this produces exactly 4 columns)
summary = summary.rename(columns={'mean':'mean_params', 'median':'median_params', 'count':'n_files'})

# Save a human-readable summary table
summary_path = os.path.join(STATS_DIR, "param_counts_summary_by_family.csv")
summary.to_csv(summary_path, index=False)

# Extract means if present
def maybe_get_mean(fam):
    row = summary[summary['family'] == fam]
    if not row.empty:
        return float(row['mean_params'].values[0])
    return np.nan

mlp_mean = maybe_get_mean('MLP')
hybrid_mean = maybe_get_mean('HYBRID_QNN')
qnn_core_mean = maybe_get_mean('QNN_CORE')

out_rows = []
if not np.isnan(mlp_mean) and not np.isnan(hybrid_mean):
    ratio = hybrid_mean / mlp_mean
    pct = ratio * 100.0
    compression = (mlp_mean / hybrid_mean) if hybrid_mean > 0 else np.nan
    out_rows.append({
        'compare':'HYBRID_QNN_vs_MLP',
        'mlp_mean_params': mlp_mean,
        'hybrid_mean_params': hybrid_mean,
        'ratio': ratio,
        'percent_of_mlp': pct,
        'compression_x': compression
    })

if not np.isnan(mlp_mean) and not np.isnan(qnn_core_mean):
    ratio2 = qnn_core_mean / mlp_mean
    pct2 = ratio2 * 100.0
    compression2 = (mlp_mean / qnn_core_mean) if qnn_core_mean > 0 else np.nan
    out_rows.append({
        'compare':'QNN_CORE_vs_MLP',
        'mlp_mean_params': mlp_mean,
        'qnn_core_mean_params': qnn_core_mean,
        'ratio': ratio2,
        'percent_of_mlp': pct2,
        'compression_x': compression2
    })

out_df = pd.DataFrame(out_rows)
out_path = os.path.join(STATS_DIR, "param_efficiency_summary.csv")
out_df.to_csv(out_path, index=False)

# Append to significance_summary.csv safely (create backup)
sig_csv = os.path.join(STATS_DIR, "significance_summary.csv")
if os.path.exists(sig_csv):
    shutil.copy2(sig_csv, sig_csv + ".bak")
else:
    # create a minimal CSV with header
    with open(sig_csv, "w") as f:
        f.write("metric,mean_QNN,mean_MLP,p_value,cohen_d,bootstrap_CI_low,bootstrap_CI_high,units\n")

# Append param rows (CSV-friendly)
with open(sig_csv, "a") as f:
    if not np.isnan(hybrid_mean):
        f.write(f"PARAMS_HYBRID_OVER_MLP,{hybrid_mean:.6f},{mlp_mean:.6f},,,,{hybrid_mean/mlp_mean:.6f},hybrid_params\n")
    if not np.isnan(qnn_core_mean):
        f.write(f"PARAMS_QNNCORE_OVER_MLP,{qnn_core_mean:.6f},{mlp_mean:.6f},,,,{qnn_core_mean/mlp_mean:.6f},qnn_core_params\n")

# Print human readable summary
print("\nWrote:", out_path)
print("Summary by family written to:", summary_path)
print("\nHuman-readable summary:")
if not np.isnan(mlp_mean):
    print(f" - MLP mean params: {mlp_mean:.0f}")
if not np.isnan(hybrid_mean):
    ratio = hybrid_mean / mlp_mean
    pct = ratio * 100.0
    compression = mlp_mean / hybrid_mean
    print(f" - Hybrid QNN mean params: {hybrid_mean:.0f}")
    print(f" - Hybrid/MLP = {ratio:.4f} ({pct:.2f}%), compression = {compression:.1f}× fewer params")
if not np.isnan(qnn_core_mean):
    ratio2 = qnn_core_mean / mlp_mean
    pct2 = ratio2 * 100.0
    compression2 = mlp_mean / qnn_core_mean
    print(f" - QNN core mean params: {qnn_core_mean:.0f}")
    print(f" - QNNcore/MLP = {ratio2:.4f} ({pct2:.2f}%), compression = {compression2:.1f}× fewer params")

print("\nAppended rows to", sig_csv, " (backup at", sig_csv + ".bak", ")")
