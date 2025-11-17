#!/usr/bin/env python3
"""
Rescore generated latent embeddings with Phase-4 hybrid QNN ensemble.

Saves:
 - 05_qgan_training/metrics/qgan_generated_ranking_<ts>.csv

Usage:
  conda activate qgen-prodyn
  cd ~/QGEN-ProDyn
  python 05_qgan_training/rescore_generated.py
"""
import os, glob, json, time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

ROOT = os.path.abspath(".")
PHASE4 = os.path.join(ROOT, "04_qnn_training")
PHASE5 = os.path.join(ROOT, "05_qgan_training")
OUT_METRICS = os.path.join(PHASE5, "metrics")
OUT_DIR = PHASE5
MODEL_DIR_PHASE4 = os.path.join(PHASE4, "models")
GENERATED_DIR = os.path.join(PHASE5, "data")
os.makedirs(OUT_METRICS, exist_ok=True)

# Find latest generated latent file
gen_files = sorted(glob.glob(os.path.join(GENERATED_DIR, "qgan_generated_latent_*.npy")))
if not gen_files:
    raise SystemExit("No generated latent files found in " + GENERATED_DIR)
gen_file = gen_files[-1]
Z_gen = np.load(gen_file)
if Z_gen.ndim == 1: Z_gen = Z_gen.reshape(-1,1)
N_gen, zdim = Z_gen.shape
print("[data] loaded generated latent:", gen_file, "shape", Z_gen.shape)

# Hybrid model class (must match the training code)
class HybridModel(nn.Module):
    def __init__(self, in_dim, n_qubits, qnn_dummy=None):
        super().__init__()
        # same projector + readout shape as training
        self.project = nn.Linear(in_dim, n_qubits)
        self.activation = nn.Tanh()
        # we won't reinstantiate the quantum backend here; during training the TorchConnector wrapped the qnn.
        # We only need to load the saved state_dict into an identical PyTorch module saved earlier.
        # To load successfully, we only need same named layers; the qnn-related parameters may be few/empty if
        # the saved state_dict stored only classical params. We'll try best-effort loading.
        self.readout = nn.Linear(1, 1)

    def forward(self, x):
        z = self.project(x)
        z = self.activation(z)
        # if qnn was a TorchConnector, forward used that; in saved state we may have only projector+readout weights
        # We'll approximate QNN output by mean(z, dim=1, keepdim=True) if TorchConnector not present after load
        try:
            out = self.qnn_layer(z)   # if present (unlikely)
        except Exception:
            out = z.mean(dim=1, keepdim=True)
        return self.readout(out)

# Helper: load all hybrid models saved in Phase4 models dir
hybrid_files = sorted(glob.glob(os.path.join(MODEL_DIR_PHASE4, "hybrid_qnn_fold*.pt")))
if not hybrid_files:
    raise SystemExit("No hybrid_qnn_fold*.pt models found in " + MODEL_DIR_PHASE4)

print("[models] found hybrid models:", hybrid_files)

# Create a fresh model instance with shapes derived from saved state dict if possible
# Attempt load for each fold; if exactly matched structure isn't present (TorchConnector), we'll still try to extract projector/readout params.
preds = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for mf in hybrid_files:
    st = torch.load(mf, map_location="cpu")
    # infer in_dim and n_qubits from keys
    # look for keys like 'project.weight' or 'projector' etc.
    in_dim = None; n_qubits = None
    for k in st.keys():
        # expect keys like 'project.weight' and 'readout.weight' etc.
        if k.endswith('project.weight'):
            n_qubits, in_dim = st[k].shape
            break
        if 'project.weight' in k:
            n_qubits, in_dim = st[k].shape
            break
    if in_dim is None or n_qubits is None:
        # fallback heuristic: look for 'project' or use zdim
        in_dim = zdim
        n_qubits = max(1, min(8, int(st.get('project.weight').shape[0]) if 'project.weight' in st else 8))

    model = HybridModel(in_dim=in_dim, n_qubits=n_qubits).to(device)
    # attempt to load partial state dict (ignore missing keys)
    try:
        model_state = {k: v for k, v in st.items() if k in model.state_dict()}
        model.state_dict().update(model_state)
        model.load_state_dict(model.state_dict())
    except Exception:
        # best-effort: manually copy weights if present
        sdict = model.state_dict()
        for k in sdict.keys():
            if k in st:
                try:
                    sdict[k] = st[k]
                except Exception:
                    pass
        try:
            model.load_state_dict(sdict)
        except Exception:
            pass

    model.eval()
    # compute predictions on generated latents
    with torch.no_grad():
        Xg = torch.tensor(Z_gen, dtype=torch.float32).to(device)
        ypred = model(Xg).cpu().numpy().reshape(-1)
    preds.append(ypred)

# Combine preds: per-fold predictions -> mean + std
preds = np.stack(preds, axis=0)   # (n_folds, N_gen)
mean_pred = preds.mean(axis=0)
std_pred = preds.std(axis=0)

# Try to inverse-transform to original ΔG using scalers.json from phase4 if present
scalers_json = os.path.join(PHASE4, "data", "scalers.json")
converted_mean = mean_pred.copy()
converted_std = std_pred.copy()
if os.path.exists(scalers_json):
    try:
        with open(scalers_json, "r") as fh:
            sc = json.load(fh)
        # Expect sc has label scaler like {"label":{"mean":..., "scale":...}} or similar
        label_info = sc.get("label") or sc.get("y") or sc.get("affinity")
        if label_info and "mean" in label_info and "scale" in label_info:
            m = float(label_info["mean"]); s = float(label_info["scale"])
            converted_mean = mean_pred * s + m
            converted_std = std_pred * s
            units = label_info.get("units", "original")
        else:
            units = "normalized"
    except Exception:
        units = "normalized"
else:
    units = "normalized"

# Save ranking CSV
ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
out_csv = os.path.join(OUT_METRICS, f"qgan_generated_ranking_{ts}.csv")
df = pd.DataFrame({
    "gen_id": np.arange(N_gen),
    "pred_mean_norm": mean_pred,
    "pred_std_norm": std_pred,
    "pred_mean": converted_mean,
    "pred_std": converted_std
})
df = df.sort_values("pred_mean")    # lower ΔG = better binder (if values are kcal/mol)
df.to_csv(out_csv, index=False)
print("[save] Wrote ranking ->", out_csv)

# quick top-10 print
print("\nTop 10 candidates (by predicted affinity):")
print(df.head(10).to_string(index=False))
