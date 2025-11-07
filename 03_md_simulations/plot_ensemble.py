#!/usr/bin/env python3
"""
plot_ensemble.py

Reads:
 - metrics/ensemble_quality.csv  (either per-replicate rows with a 'group' column OR a key,value layout)
 - metrics/ensemble_phi_psi_hists.npz (optional; may contain histograms or precomputed KLs)

Writes:
 - figure_2d_ensemble_quality.png
 - figure_2d_ensemble_quality.svg

This version is robust to several CSV layouts and will compute per-group means ± SE for Rg and Q_native.
It will attempt to compute KL from histogram arrays if available, else fall back to CSV keys KL_phi/KL_psi.
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

ROOT = os.path.dirname(__file__) or "."
MET = os.path.join(ROOT, "metrics")
CSV = os.path.join(MET, "ensemble_quality.csv")
HISTNPZ = os.path.join(MET, "ensemble_phi_psi_hists.npz")

if not os.path.exists(CSV):
    print("Missing CSV:", CSV)
    sys.exit(1)

# Helper: compute sem robustly (ignores NaNs)
def safe_sem(arr):
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    if n <= 1:
        return 0.0
    return float(np.std(a, ddof=1) / math.sqrt(n))

# Load CSV and detect layout
df_raw = pd.read_csv(CSV, header=0)

# Case A: per-replicate rows with a 'group' column (preferred layout)
if 'group' in df_raw.columns:
    df = df_raw.copy()
    # Accept multiple possible column names for Rg and Q_native
    rg_col = None
    q_col = None
    for c in ['Rg_mean','Rg','Rg_nm','Rg_mean_nm']:
        if c in df.columns:
            rg_col = c; break
    for c in ['Q_native','Q','Q_mean']:
        if c in df.columns:
            q_col = c; break
    if rg_col is None or q_col is None:
        # try to infer by numeric columns other than group
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        # remove possible KL columns
        candidates = [c for c in numeric_cols if not c.lower().startswith('kl')]
        if len(candidates) >= 2:
            rg_col = rg_col or candidates[0]
            q_col = q_col or candidates[1]
        elif len(candidates) == 1:
            rg_col = rg_col or candidates[0]
            q_col = q_col or candidates[0]
        else:
            print("Could not find Rg/Q columns in CSV. Columns:", df.columns.tolist())
            sys.exit(1)
    # Group stats
    groups = sorted(df['group'].astype(str).unique().tolist())
    stats = {}
    for g in groups:
        sub = df[df['group'].astype(str) == str(g)]
        rg_vals = pd.to_numeric(sub[rg_col], errors='coerce').values
        q_vals  = pd.to_numeric(sub[q_col], errors='coerce').values
        stats[g] = {
            'Rg_mean': float(np.nanmean(rg_vals)) if rg_vals.size>0 else float('nan'),
            'Rg_se': safe_sem(rg_vals),
            'Q_mean': float(np.nanmean(q_vals)) if q_vals.size>0 else float('nan'),
            'Q_se': safe_sem(q_vals)
        }
    # If CSV contains KL columns at per-row or aggregated level, capture unique values
    kl_phi = None
    kl_psi = None
    if 'KL_phi' in df.columns:
        vals = df['KL_phi'].dropna().unique()
        kl_phi = float(vals[0]) if vals.size>0 else None
    if 'KL_psi' in df.columns:
        vals = df['KL_psi'].dropna().unique()
        kl_psi = float(vals[0]) if vals.size>0 else None

# Case B: two-column key,value CSV (fallback, aggregated single-run output)
else:
    # Try to interpret as key,value pairs
    if df_raw.shape[1] == 2:
        kv = {}
        # ensure strings for keys
        for _, row in df_raw.iterrows():
            k = str(row.iloc[0]).strip()
            v = row.iloc[1]
            # try numeric conversion
            try:
                kv[k] = float(v)
            except Exception:
                kv[k] = v
        # Try to extract expected keys
        groups = ['quantum', 'control']
        stats = {}
        # Rg and Q may be stored as Rg_mean_quantum, etc.
        for g in groups:
            rkey = f"Rg_mean_{g}"
            qkey = f"Q_native_{g}"
            if rkey in kv and qkey in kv:
                stats[g] = {
                    'Rg_mean': float(kv[rkey]),
                    'Rg_se': float(kv.get(f"Rg_se_{g}", 0.0)),
                    'Q_mean': float(kv[qkey]),
                    'Q_se': float(kv.get(f"Q_se_{g}", 0.0))
                }
        # KL keys
        kl_phi = kv.get('KL_phi', None)
        kl_psi = kv.get('KL_psi', None)
        # If stats still empty, try alternate key names
        if not stats:
            # Try quantum_Rg, control_Rg, quantum_Q...
            for g in groups:
                for prefix in [g, g.capitalize(), g.upper()]:
                    rkey = f"{prefix}_Rg"
                    qkey = f"{prefix}_Q_native"
                    if rkey in kv and qkey in kv:
                        stats[g] = {
                            'Rg_mean': float(kv[rkey]),
                            'Rg_se': float(kv.get(f"{prefix}_Rg_se", 0.0)),
                            'Q_mean': float(kv[qkey]),
                            'Q_se': float(kv.get(f"{prefix}_Q_se", 0.0))
                        }
        if not stats:
            print("Unrecognized aggregated CSV layout. Keys found:", list(kv.keys()))
            print("Please provide CSV with either (group, Rg_mean, Q_native) rows or key,value pairs.")
            sys.exit(1)
    else:
        print("CSV does not contain 'group' column and is not a 2-column key,value table. Columns:", df_raw.columns.tolist())
        sys.exit(1)

# At this point we have:
# - groups: list of group names
# - stats: dict[group] -> {Rg_mean, Rg_se, Q_mean, Q_se}
# - kl_phi, kl_psi maybe set (or None)

# Try computing KL from histograms if available
if os.path.exists(HISTNPZ):
    try:
        data = np.load(HISTNPZ, allow_pickle=True)
        # Accept multiple possible naming conventions
        def pick(key_candidates):
            for k in key_candidates:
                if k in data.files:
                    return data[k]
            return None
        # For phi/psi we may have 2D hist or separate 1D hist arrays; handle both:
        q_phi = pick(['quantum_phi', 'quantum_phi_hist', 'q_phi', 'phi_quantum'])
        c_phi = pick(['control_phi', 'control_phi_hist', 'c_phi', 'phi_control'])
        q_psi = pick(['quantum_psi', 'quantum_psi_hist', 'q_psi', 'psi_quantum'])
        c_psi = pick(['control_psi', 'control_psi_hist', 'c_psi', 'psi_control'])
        # If 2D histograms stored as 'quantum_H' and 'control_H' try to marginalize to phi and psi if needed
        # But simplest: if 1D arrays present, compute KL; else if 'KL_phi' stored, use them
        if q_phi is not None and c_phi is not None:
            a = np.asarray(q_phi, dtype=float).ravel()
            b = np.asarray(c_phi, dtype=float).ravel()
            L = min(a.size, b.size)
            if L > 0:
                a = a[:L] + 1e-12
                b = b[:L] + 1e-12
                a = a / a.sum()
                b = b / b.sum()
                kl_phi = float(entropy(a, b))
        if q_psi is not None and c_psi is not None:
            a = np.asarray(q_psi, dtype=float).ravel()
            b = np.asarray(c_psi, dtype=float).ravel()
            L = min(a.size, b.size)
            if L > 0:
                a = a[:L] + 1e-12
                b = b[:L] + 1e-12
                a = a / a.sum()
                b = b / b.sum()
                kl_psi = float(entropy(a, b))
        # final fallback: maybe the npz contains precomputed KL values
        if kl_phi is None and 'KL_phi' in data.files:
            kl_phi = float(data['KL_phi'])
        if kl_psi is None and 'KL_psi' in data.files:
            kl_psi = float(data['KL_psi'])
    except Exception as e:
        print("Warning: error loading/using hist npz:", e)

# Ensure numeric kl values (or set to np.nan)
kl_phi = np.nan if kl_phi is None else float(kl_phi)
kl_psi = np.nan if kl_psi is None else float(kl_psi)

# Prepare plotting arrays
labels = list(stats.keys())
means_rg = [stats[g]['Rg_mean'] for g in labels]
errs_rg =  [stats[g]['Rg_se'] for g in labels]
means_q  = [stats[g]['Q_mean']  for g in labels]
errs_q   = [stats[g]['Q_se']   for g in labels]

# Plot
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))

# Panel 1: KL
kl_vals = [0.0 if np.isnan(kl_phi) else kl_phi, 0.0 if np.isnan(kl_psi) else kl_psi]
axs[0].bar([0,1], kl_vals, color=['#2a6f97','#f28e2b'])
axs[0].set_xticks([0,1]); axs[0].set_xticklabels(['KL_phi','KL_psi'])
axs[0].set_title('KL(φ/ψ) Q||C')
if not np.isnan(kl_phi) or not np.isnan(kl_psi):
    axs[0].set_ylabel('KL (nats)')

# Panel 2: Rg
x = np.arange(len(labels))
axs[1].bar(x, means_rg, yerr=errs_rg, capsize=4)
axs[1].set_xticks(x); axs[1].set_xticklabels(labels)
axs[1].set_title('Rg mean (nm)')
axs[1].set_ylabel('Rg (nm)')

# Panel 3: Q_native
axs[2].bar(x, means_q, yerr=errs_q, capsize=4)
axs[2].set_xticks(x); axs[2].set_xticklabels(labels)
axs[2].set_title('Q_native')

for ax in axs:
    ax.tick_params(labelsize=9)

plt.suptitle("Figure 2D — Ensemble quality (φ/ψ KL, Rg, Q_native)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])

png = os.path.join(ROOT, "figure_2d_ensemble_quality.png")
svg = os.path.join(ROOT, "figure_2d_ensemble_quality.svg")
plt.savefig(png, dpi=300)
plt.savefig(svg)
print("[plot_ensemble] Wrote", png, "and", svg)
