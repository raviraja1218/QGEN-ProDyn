#!/usr/bin/env python3
"""
compute_param_counts.py
Reads saved model .pt files in models/ (qnn_* and mlp_*) and reports parameter counts.
Outputs: stats/param_counts.csv
"""
import os, torch, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUT = ROOT / "stats"
OUT.mkdir(exist_ok=True)

def count_params_state_dict(path):
    st = torch.load(path, map_location="cpu")
    total = 0
    for k,v in st.items():
        total += v.numel()
    return total

rows = []
for p in MODELS.glob("*.pt"):
    try:
        cnt = count_params_state_dict(p)
    except Exception as e:
        cnt = None
    rows.append({"model_file": p.name, "params": cnt})

df = pd.DataFrame(rows)
df.to_csv(OUT / "param_counts.csv", index=False)
print("Wrote", OUT / "param_counts.csv")
print(df)
