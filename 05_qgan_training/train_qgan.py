#!/usr/bin/env python3
"""
train_qgan.py

Hybrid QGAN training (quantum generator + PyTorch discriminator).

Usage:
  conda activate qgen-prodyn
  cd ~/QGEN-ProDyn
  python 05_qgan_training/train_qgan.py --epochs 2000 --batch_size 16 --lr 1e-4

Notes:
 - The script expects:
     05_qgan_training/data/qnn_latent_embeddings.npy
     05_qgan_training/data/affinity_labels.npy
 - Quantum generator uses pennylane default.qubit (CPU simulator).
 - Discriminator (PyTorch) will use CUDA if available.
"""
import os
import argparse
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pennylane as qml

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser(description="Train hybrid QGAN (quantum generator + PyTorch discriminator)")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--qbits", type=int, default=5, help="Number of qubits for the generator")
parser.add_argument("--seed", type=int, default=12345, help="Random seed")
parser.add_argument("--outdir", type=str, default="05_qgan_training", help="Phase5 working dir")
args = parser.parse_args()

# ---------------------------
# Reproducibility
# ---------------------------
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ---------------------------
# Paths & folders
# ---------------------------
ROOT = os.path.abspath(".")
DATA_DIR = os.path.join(ROOT, args.outdir, "data")
OUT_DIR = os.path.join(ROOT, args.outdir)
MODELS_DIR = os.path.join(OUT_DIR, "models")
FIG_DIR = os.path.join(OUT_DIR, "figures")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device} (CUDA available: {torch.cuda.is_available()})")

# ---------------------------
# Load data
# ---------------------------
feat_path = os.path.join(DATA_DIR, "qnn_latent_embeddings.npy")
lab_path = os.path.join(DATA_DIR, "affinity_labels.npy")
if not os.path.exists(feat_path) or not os.path.exists(lab_path):
    raise SystemExit(f"Missing data. Expected:\n - {feat_path}\n - {lab_path}")

Z = np.load(feat_path)  # (N, zdim)
aff = np.load(lab_path)  # (N,)
if Z.ndim == 1:
    Z = Z.reshape(-1, 1)
if aff.ndim > 1 and aff.shape[1] == 1:
    aff = aff.ravel()

N, zdim = Z.shape
print(f"[data] Loaded Z {Z.shape}, affinities {aff.shape}")

# Normalize affinity to [0,1] for conditioning if needed (simple min-max)
aff_min, aff_max = float(np.min(aff)), float(np.max(aff))
if aff_max - aff_min < 1e-8:
    aff_norm = np.zeros_like(aff, dtype=np.float32)
else:
    aff_norm = (aff - aff_min) / (aff_max - aff_min)
aff_norm = aff_norm.astype(np.float32)

# Torch datasets
X_t = torch.tensor(Z, dtype=torch.float32)
C_t = torch.tensor(aff_norm.reshape(-1, 1), dtype=torch.float32)
dataset = TensorDataset(X_t, C_t)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

# ---------------------------
# PyTorch Discriminator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# PennyLane Quantum Generator (sample-by-sample qnode eval)
# - outputs gen_output_dim values per sample
# - post classical layer maps qnode outputs + condition -> gen_output_dim
# ---------------------------
qbits = max(1, int(args.qbits))
# determine generator output dim based on data latent dim
gen_output_dim = zdim
print(f"[shape] data latent dim={zdim}, gen_output_dim={gen_output_dim}, qbits={qbits}")

# PennyLane device (CPU simulator). If lightning available, you could switch automatically.
try:
    # attempt to use lightning if installed (much faster)
    qml_dev = qml.device("lightning.qubit", wires=qbits)
    print("[qml] Using pennylane-lightning backend")
except Exception:
    qml_dev = qml.device("default.qubit", wires=qbits)
    print("[qml] Using default.qubit backend (CPU simulator)")

# Create qnode that returns expectation values (one per qubit or fewer)
def make_qnode(n_qubits, out_dim, nlayers=2):
    @qml.qnode(qml_dev, interface="torch")
    def circuit(params, cond):
        # params: (nlayers, n_qubits) tensor (torch)
        # cond: shape (cond_dim,) or scalar - we allow cond to modulate angles
        # simple parameterized rotations + entangling CZs
        for layer in range(params.shape[0]):
            for w in range(n_qubits):
                qml.RY(params[layer, w], wires=w)
            for w in range(n_qubits - 1):
                qml.CZ(wires=[w, w+1])
        # readout: expectation of Z on first out_dim qubits
        outs = []
        for w in range(min(n_qubits, out_dim)):
            outs.append(qml.expval(qml.PauliZ(w)))
        return outs
    return circuit

nlayers = 2
init_params = 0.05 * np.random.randn(nlayers, qbits).astype(np.float32)
# wrap these into torch Parameter inside module

# QuantumGenerator torch Module
class QuantumGenerator(nn.Module):
    def __init__(self, qnode, init_params, gen_output_dim, cond_dim=1):
        super().__init__()
        self.qnode = qnode
        # register params as torch Parameter
        self.params = nn.Parameter(torch.tensor(init_params, dtype=torch.float32))
        self.qout_dim = min(qbits, gen_output_dim)
        self.cond_dim = cond_dim
        # small classical post-mapping
        self.post = nn.Sequential(
            nn.Linear(self.qout_dim + cond_dim, gen_output_dim),
            nn.Tanh()
        )

    def forward(self, cond_batch):
        """
        cond_batch: tensor (batch, cond_dim)
        returns: tensor (batch, gen_output_dim)
        """
        batch = cond_batch.shape[0]
        outs = []
        # Evaluate qnode per-sample (unbatched) — fine for small batches and small qbits
        for i in range(batch):
            # ensure cond is numpy float array (not strictly used inside qnode here)
            cond = cond_batch[i].detach().cpu().numpy().astype(np.float32)
            # call qnode with (params, cond) signature
            qout = self.qnode(self.params, cond)
            # qout is a list-like -> convert to torch tensor
            qout_t = torch.tensor(qout, dtype=torch.float32)
            outs.append(qout_t)
        outs = torch.stack(outs).to(device)  # (batch, qout_dim)
        x = torch.cat([outs, cond_batch.to(device)], dim=1)  # (batch, qout_dim+cond_dim)
        return self.post(x)  # (batch, gen_output_dim)

# ---------------------------
# Instantiate models
# ---------------------------
cond_dim = 1
qnode = make_qnode(qbits, gen_output_dim, nlayers=nlayers)
G = QuantumGenerator(qnode, init_params, gen_output_dim, cond_dim=cond_dim).to(device)
D = Discriminator(input_dim=(gen_output_dim + cond_dim)).to(device)

# ---------------------------
# Optimizers / losses
# ---------------------------
# Only train classical parameters: D params + G.post params + G.params are torch params (G.params is trainable)
opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))
opt_G = torch.optim.Adam(list(G.post.parameters()) + [G.params], lr=args.lr, betas=(0.5, 0.9))
bce = nn.BCELoss()

# ---------------------------
# Training loop
# ---------------------------
epochs = args.epochs
batch_size = args.batch_size
print(f"[train] epochs={epochs}, batch_size={batch_size}, lr={args.lr}")

history = {"G_loss": [], "D_loss": []}
start_time = time.time()

# For label smoothing
real_label = 0.9
fake_label = 0.0

for ep in range(1, epochs + 1):
    epoch_G_loss = 0.0
    epoch_D_loss = 0.0
    nsteps = 0
    for real_z_batch, cond_batch in loader:
        nsteps += 1
        real_z_batch = real_z_batch.to(device)                 # (b, zdim)
        cond_batch = cond_batch.to(device)                     # (b,1)

        bsize = real_z_batch.shape[0]
        # ----------------------
        # Update Discriminator: maximize log D(x) + log(1 - D(G(z)))
        # ----------------------
        D.train()
        opt_D.zero_grad()

        # Real pairs -> input = concat(real_z, cond)
        real_input = torch.cat([real_z_batch, cond_batch], dim=1)  # (b, zdim+cond)
        labels_real = torch.full((bsize, 1), real_label, dtype=torch.float32, device=device)
        out_real = D(real_input)
        loss_real = bce(out_real, labels_real)

        # Fake pairs (G generates from condition only; noise can be condition)
        fake_z = G(cond_batch)  # (b, gen_output_dim)
        fake_input = torch.cat([fake_z.detach(), cond_batch], dim=1)
        labels_fake = torch.full((bsize, 1), fake_label, dtype=torch.float32, device=device)
        out_fake = D(fake_input)
        loss_fake = bce(out_fake, labels_fake)

        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        # ----------------------
        # Update Generator: try to maximize D(G(z)) -> minimize BCE(D(G(z)), 1)
        # ----------------------
        opt_G.zero_grad()
        fake_z2 = G(cond_batch)
        out_fake2 = D(torch.cat([fake_z2, cond_batch], dim=1))
        # generator tries to make discriminator output close to 1
        labels_gen = torch.full((bsize, 1), 1.0, dtype=torch.float32, device=device)
        loss_G = bce(out_fake2, labels_gen)
        loss_G.backward()
        opt_G.step()

        epoch_G_loss += float(loss_G.item())
        epoch_D_loss += float(loss_D.item())

    # log per epoch
    if nsteps == 0:
        avg_G = 0.0
        avg_D = 0.0
    else:
        avg_G = epoch_G_loss / nsteps
        avg_D = epoch_D_loss / nsteps
    history["G_loss"].append(avg_G)
    history["D_loss"].append(avg_D)

    if ep == 1 or ep % 10 == 0 or ep == epochs:
        print(f"Epoch {ep:4d} | G {avg_G:.4f} D {avg_D:.4f}")

# ---------------------------
# Save artifacts
# ---------------------------
ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
model_G_path = os.path.join(MODELS_DIR, f"qgan_generator_{ts}.pt")
model_D_path = os.path.join(MODELS_DIR, f"qgan_discriminator_{ts}.pt")
torch.save(G.state_dict(), model_G_path)
torch.save(D.state_dict(), model_D_path)
print(f"[save] Saved G -> {model_G_path}")
print(f"[save] Saved D -> {model_D_path}")

# Save training history
hist_path = os.path.join(METRICS_DIR, f"qgan_train_history_{ts}.json")
with open(hist_path, "w") as fh:
    json.dump(history, fh, indent=2)
print(f"[save] Wrote training history -> {hist_path}")

# ---------------------------
# Generate a batch of samples and t-SNE plot (real vs generated)
# ---------------------------
G.eval()
with torch.no_grad():
    # use whole dataset conditions to produce generated points
    cond_all = torch.tensor(aff_norm.reshape(-1, 1), dtype=torch.float32).to(device)
    gen_all = G(cond_all).cpu().numpy()  # (N, zdim)
    real_all = Z  # numpy (N, zdim)

# t-SNE (works with small N)
tsne = TSNE(n_components=2, random_state=args.seed, perplexity=max(2, min(30, N-1)))
proj_real = tsne.fit_transform(real_all)
proj_gen = tsne.fit_transform(np.vstack([real_all, gen_all]))  # reproject combined
proj_gen = proj_gen[real_all.shape[0]:]  # keep generated part

plt.figure(figsize=(6, 4))
plt.scatter(proj_real[:, 0], proj_real[:, 1], label="real", alpha=0.8, s=40)
plt.scatter(proj_gen[:, 0], proj_gen[:, 1], label="generated", alpha=0.8, s=40, marker='x')
plt.legend()
plt.title("t-SNE: real vs QGAN-generated (latent space)")
png = os.path.join(FIG_DIR, f"qgan_latent_tsne_{ts}.png")
svg = os.path.join(FIG_DIR, f"qgan_latent_tsne_{ts}.svg")
plt.tight_layout()
plt.savefig(png, dpi=300)
plt.savefig(svg)
print(f"[save] Saved t-SNE -> {png} / {svg}")

# Also save generated sequences / embeddings
gen_out_path = os.path.join(OUT_DIR, "data", f"qgan_generated_latent_{ts}.npy")
np.save(gen_out_path, gen_all)
print(f"[save] Wrote generated latent embeddings -> {gen_out_path}")

elapsed = time.time() - start_time
print(f"[done] Training finished in {elapsed:.1f}s")
