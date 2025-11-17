#!/usr/bin/env python3
"""
load_qnn_latent.py

- Inputs (preferred):
    ../04_qnn_training/data/qnn_input_features_norm.npy
    ../04_qnn_training/data/qnn_labels_norm.npy

- Fallback inputs:
    ../04_qnn_training/data/qnn_input_features.npy
    ../04_qnn_training/data/qnn_labels.npy

- Outputs:
    05_qgan_training/data/qnn_latent_embeddings.npy
    05_qgan_training/data/affinity_labels.npy
    05_qgan_training/data/preprocessing_notes.md

Usage (from project root or anywhere):
    python 05_qgan_training/data/load_qnn_latent.py
"""
import os
import json
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA

# ---------------------------
# Path handling (robust)
# ---------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # .../05_qgan_training/data
PHASE5_DIR = os.path.abspath(os.path.join(HERE, ".."))     # .../05_qgan_training
PROJECT_ROOT = os.path.abspath(os.path.join(PHASE5_DIR, ".."))

SRC_DIR = os.path.join(PROJECT_ROOT, "04_qnn_training", "data")
OUTDIR = os.path.join(PHASE5_DIR, "data")                  # correct target: .../05_qgan_training/data
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------
# Find feature file (prefer normalized)
# ---------------------------
feature_candidates = [
    os.path.join(SRC_DIR, "qnn_input_features_norm.npy"),
    os.path.join(SRC_DIR, "qnn_input_features.npy"),
]
feature_path = None
for p in feature_candidates:
    if os.path.exists(p):
        feature_path = p
        break
if feature_path is None:
    raise SystemExit(f"ERROR: Could not find any qnn input features in {SRC_DIR}. "
                     f"Checked: {feature_candidates}")

# ---------------------------
# Find label file (prefer normalized)
# ---------------------------
label_candidates = [
    os.path.join(SRC_DIR, "qnn_labels_norm.npy"),
    os.path.join(SRC_DIR, "qnn_labels.npy"),
]
label_path = None
for p in label_candidates:
    if os.path.exists(p):
        label_path = p
        break
if label_path is None:
    raise SystemExit(f"ERROR: Could not find any qnn labels in {SRC_DIR}. "
                     f"Checked: {label_candidates}")

# ---------------------------
# Load data
# ---------------------------
X = np.load(feature_path)
y = np.load(label_path)

if X.ndim == 1:
    X = X.reshape(-1, 1)
if y.ndim > 1 and y.shape[1] == 1:
    y = y.ravel()

if X.shape[0] != y.shape[0]:
    raise SystemExit(f"ERROR: Number of samples mismatch: features {X.shape[0]} vs labels {y.shape[0]}")

print(f"[data] Loaded features: {feature_path} -> shape {X.shape}")
print(f"[data] Loaded labels:   {label_path} -> shape {y.shape}")

# ---------------------------
# Create latent embeddings
# ---------------------------
# Choose latent dim: min(16, n_features, n_samples-1)
latent_dim = min(16, X.shape[1], max(1, X.shape[0] - 1))
if latent_dim < 1:
    latent_dim = 1

pca = PCA(n_components=latent_dim, random_state=12345)
Z = pca.fit_transform(X)
print(f"[data] PCA latent shape: {Z.shape}  (n_components={latent_dim})")

# ---------------------------
# Save outputs
# ---------------------------
feat_out = os.path.join(OUTDIR, "qnn_latent_embeddings.npy")
lab_out  = os.path.join(OUTDIR, "affinity_labels.npy")
notes   = os.path.join(OUTDIR, "preprocessing_notes.md")
meta_json = os.path.join(OUTDIR, "preprocessing_meta.json")

np.save(feat_out, Z)
np.save(lab_out, y)

with open(notes, "w") as f:
    f.write(f"- source_features: {feature_path}\n")
    f.write(f"- source_labels: {label_path}\n")
    f.write(f"- method: PCA(n_components={latent_dim})\n")
    f.write(f"- n_samples: {X.shape[0]}\n")
    f.write(f"- feature_shape: {X.shape}\n")
    f.write(f"- latent_shape: {Z.shape}\n")
    f.write(f"- date_utc: {datetime.utcnow().isoformat()}Z\n")

meta = {
    "feature_path": os.path.relpath(feature_path, PROJECT_ROOT),
    "label_path": os.path.relpath(label_path, PROJECT_ROOT),
    "out_embeddings": os.path.relpath(feat_out, PROJECT_ROOT),
    "out_labels": os.path.relpath(lab_out, PROJECT_ROOT),
    "method": "PCA",
    "pca_n_components": latent_dim,
    "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    "created_at_utc": datetime.utcnow().isoformat() + "Z",
    "notes_file": os.path.relpath(notes, PROJECT_ROOT),
}
with open(meta_json, "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"[data] Saved latent embeddings -> {feat_out}")
print(f"[data] Saved labels             -> {lab_out}")
print(f"[meta] Wrote preprocessing meta -> {meta_json}")
print(f"[notes] Wrote preprocessing notes -> {notes}")
