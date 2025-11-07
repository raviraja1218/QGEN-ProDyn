# plot_convergence.py
# Build Figure 2C from the saved group-mean RMSD curves.

import os, numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
METRICS = os.path.join(ROOT, "metrics")

def safe_load(path):
    try:
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 0:  # empty file case
            return None
        return arr
    except Exception:
        return None

q = safe_load(os.path.join(METRICS, "convergence_curves_quantum.csv"))
c = safe_load(os.path.join(METRICS, "convergence_curves_control.csv"))

plt.figure(figsize=(7,4))
if q is not None:
    plt.plot(q, label="Quantum-init (Å)")
if c is not None:
    plt.plot(c, label="Control (Å)")
plt.xlabel("Frame index (arbitrary spacing)")
plt.ylabel("RMSD (Å)")
plt.title("Figure 2C — Convergence curves (mean per group)")
plt.legend()
plt.tight_layout()
plt.savefig("figure_2c_convergence_curves.png", dpi=300)
plt.savefig("figure_2c_convergence_curves.svg")
print("[plot_convergence] Wrote figure_2c_convergence_curves.*")
