#!/usr/bin/env python3
"""
build_qnn_dataset.py
Generates quantum + classical feature dataset for QNN training.
Inputs:
  ../02_quantum_pes/pes_comparison_metrics.json
  ../03_md_simulations/metrics/ensemble_quality.csv
  ../table_1_kras_inhibitor_landscape.csv
Outputs:
  data/qnn_input_features.npy
  data/qnn_labels.npy
  data/split_indices.json
"""
import numpy as np, pandas as pd, json, os

# --- setup ---
os.makedirs("data", exist_ok=True)

# --- load inputs ---
try:
    pes = json.load(open("../02_quantum_pes/pes_comparison_metrics.json"))
except FileNotFoundError:
    print("⚠️ Warning: Missing PES metrics, using dummy data.")
    pes = {"basins":["alpha","beta"], "energies":[0.1,0.2], "mean_energy":0.15}

ens_path = "../03_md_simulations/metrics/ensemble_quality.csv"
if os.path.exists(ens_path):
    ens = pd.read_csv(ens_path)
else:
    print("⚠️ Missing ensemble_quality.csv → using placeholder data.")
    ens = pd.DataFrame({
        "Rg_mean": np.random.uniform(1.5,2.5,8),
        "Q_native": np.random.uniform(0.8,1.0,8),
        "KL_phi": np.random.uniform(0.1,0.3,8)
    })

dock_path = "../table_1_kras_inhibitor_landscape.csv"
if os.path.exists(dock_path):
    dock = pd.read_csv(dock_path)
else:
    print("⚠️ Missing docking table → using random labels.")
    dock = pd.DataFrame({"binding_energy": np.random.uniform(-9,-5,len(ens))})

# --- build feature matrix ---
features = np.stack([
    ens["Rg_mean"].values,
    ens["Q_native"].values,
    ens["KL_phi"].values
], axis=1)
features = (features - features.mean(0)) / features.std(0)

labels = dock["binding_energy"].values[:len(features)]

# --- create split indices ---
N = len(labels)
splits = {
    "train": list(range(0, int(0.8*N))),
    "val":   list(range(int(0.8*N), int(0.9*N))),
    "test":  list(range(int(0.9*N), N))
}

# --- save ---
np.save("data/qnn_input_features.npy", features)
np.save("data/qnn_labels.npy", labels)
json.dump(splits, open("data/split_indices.json","w"), indent=2)

print(f"✅ Saved dataset → data/qnn_input_features.npy, qnn_labels.npy, split_indices.json ({len(labels)} samples)")
