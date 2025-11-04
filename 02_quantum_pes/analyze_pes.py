#!/usr/bin/env python3
import os, json, numpy as np
import matplotlib.pyplot as plt
from collections import deque

HERE = os.path.dirname(__file__)

vqe = np.load(os.path.join(HERE, "vqe_energy_grid.npy"))
ref = np.load(os.path.join(HERE, "dft_energy_grid.npy"))
phi = np.load(os.path.join(HERE, "phi_values.npy"))
psi = np.load(os.path.join(HERE, "psi_values.npy"))

# -------- metrics ----------
x = vqe.ravel()
y = ref.ravel()
mae  = float(np.mean(np.abs(x - y)))
rmse = float(np.sqrt(np.mean((x - y)**2)))
# guard divide-by-zero for MARD
den = np.maximum(np.abs(y), 1e-8)
mard = float(np.mean(np.abs((x - y)/den)))
# R^2
ybar = float(np.mean(y))
ss_res = float(np.sum((y - x)**2))
ss_tot = float(np.sum((y - ybar)**2))
r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
metrics = {"R2": r2, "MAE": mae, "RMSE": rmse, "MARD": mard,
           "VQE_min": float(x.min()), "VQE_max": float(x.max()),
           "REF_min": float(y.min()), "REF_max": float(y.max())}
json.dump(metrics, open(os.path.join(HERE, "pes_comparison_metrics.json"), "w"), indent=2)
print("Metrics:", metrics)

# -------- low-energy mask & basins ----------
q = np.quantile(vqe, 0.15)  # lowest 15%
mask = vqe <= q

# connected components on 4-neighbour grid
H, W = mask.shape
labels = -np.ones_like(vqe, dtype=int)
lab = 0
dirs = [(1,0),(-1,0),(0,1),(0,-1)]
for i in range(H):
    for j in range(W):
        if mask[i,j] and labels[i,j] < 0:
            dq = deque([(i,j)])
            labels[i,j] = lab
            while dq:
                r,c = dq.popleft()
                for dr,dc in dirs:
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr,cc] and labels[rr,cc] < 0:
                        labels[rr,cc] = lab
                        dq.append((rr,cc))
            lab += 1

np.save(os.path.join(HERE, "low_energy_mask.npy"), mask)
np.save(os.path.join(HERE, "basin_labels.npy"), labels)

# -------- figures ----------
# 2A: heatmap of VQE PES
plt.figure(figsize=(6,5))
im = plt.imshow(vqe.T, origin="lower",
                extent=[phi[0], phi[-1]+(phi[1]-phi[0]), psi[0], psi[-1]+(psi[1]-psi[0])],
                aspect='auto')
plt.xlabel("φ (deg)")
plt.ylabel("ψ (deg)")
plt.title("Φ/Ψ PES (VQE surrogate)")
plt.colorbar(im, label="Energy (kcal/mol)")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "figure_2a_pes_heatmap.png"), dpi=300)
plt.savefig(os.path.join(HERE, "figure_2a_pes_heatmap.svg"))
plt.close()

# 2B: correlation VQE vs REF
plt.figure(figsize=(5.2,5.2))
plt.scatter(y, x, s=8, alpha=0.6)
mn = float(min(x.min(), y.min()))
mx = float(max(x.max(), y.max()))
plt.plot([mn, mx], [mn, mx], lw=1)
plt.xlabel("Reference energy (kcal/mol)")
plt.ylabel("VQE energy (kcal/mol)")
plt.title(f"VQE vs Reference (R²={r2:.3f}, MAE={mae:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "figure_2b_energy_correlation.png"), dpi=300)
plt.savefig(os.path.join(HERE, "figure_2b_energy_correlation.svg"))
plt.close()

# Supplementary: basin annotation on PES
plt.figure(figsize=(6,5))
im = plt.imshow(vqe.T, origin="lower",
                extent=[phi[0], phi[-1]+(phi[1]-phi[0]), psi[0], psi[-1]+(psi[1]-psi[0])],
                aspect='auto')
plt.contour(mask.T.astype(int),
            levels=[0.5],
            origin="lower",
            extent=[phi[0], phi[-1]+(phi[1]-phi[0]), psi[0], psi[-1]+(psi[1]-psi[0])])
plt.xlabel("φ (deg)")
plt.ylabel("ψ (deg)")
plt.title("Low-energy basins (threshold: 15%)")
plt.colorbar(im, label="Energy (kcal/mol)")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "supplementary_figure_1_basin_annotation.png"), dpi=300)
plt.close()

print("Figures written to 02_quantum_pes/")
