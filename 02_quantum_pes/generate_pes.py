#!/usr/bin/env python3
import json, os, numpy as np

HERE = os.path.dirname(__file__)
CFG = json.load(open(os.path.join(HERE, "config_pes.json")))

rng = np.random.default_rng(CFG.get("random_seed", 12345))

# --- grid ---
G = int(CFG["grid_points"])
phi = np.linspace(-180.0, 180.0, G, endpoint=False)
psi = np.linspace(-180.0, 180.0, G, endpoint=False)
PHI, PSI = np.meshgrid(phi, psi, indexing="ij")

def _rad(x): return np.deg2rad(x)

# ------------------------------
# Quantum-inspired surrogate PES
# (small diagonal 2-qubit Hamiltonian H=a ZI + b IZ + c ZZ;
#  ground-state energy is min over z_i ∈ {-1, +1})
# ------------------------------
def vqe_energy(phi_deg, psi_deg):
    x, y = _rad(phi_deg), _rad(psi_deg)
    a = -1.0 + 0.30*np.cos(x) + 0.20*np.sin(y) + 0.06*np.cos(2*(x-y))
    b = -0.7 + 0.25*np.cos(y) - 0.15*np.sin(x) + 0.05*np.cos(x+y)
    c = -0.5 + 0.40*np.cos(x-y) + 0.10*np.cos(2*(x+y))
    # eigenvalues for diagonal H on |z1,z2> with z1,z2 ∈ {-1,+1}
    evals = []
    for z1 in (-1.0, +1.0):
        for z2 in (-1.0, +1.0):
            evals.append(a*z1 + b*z2 + c*z1*z2)
    e = min(evals)
    # convert to kcal/mol-ish scale and add small smooth term to shape landscape
    e = (e + 2.5) * 2.0 + 0.3*(1-np.cos(x+0.3))* (1-np.cos(y-0.2))
    return float(e)

# ------------------------------
# Classical reference (proxy to DFT)
# Ramachandran-like multi-cosine potential with α/β basins
# ------------------------------
def proxy_ref_energy(phi_deg, psi_deg):
    x, y = _rad(phi_deg), _rad(psi_deg)
    # alpha basin near (-60, -45); beta near (-135, 135)
    E  = 1.2*(1 - np.cos(x + np.deg2rad(60))) \
       + 1.0*(1 - np.cos(y + np.deg2rad(45))) \
       + 0.9*(1 - np.cos(x + np.deg2rad(135))) \
       + 0.8*(1 - np.cos(y - np.deg2rad(135)))
    # steric/repulsion shaping
    E += 0.6*(1 + np.cos(x + y)) + 0.3*(1 - np.cos(2*x - y))
    return float(E)

# --- build grids ---
vqe_grid = np.empty((G, G), dtype=np.float64)
ref_grid = np.empty((G, G), dtype=np.float64)

for i, ph in enumerate(phi):
    for j, ps in enumerate(psi):
        vqe_grid[i, j] = vqe_energy(ph, ps)
        if CFG["reference"]["mode"] == "proxy_mm":
            ref_grid[i, j] = proxy_ref_energy(ph, ps)
        else:
            # placeholder: if later switching to PSI4, replace this branch
            ref_grid[i, j] = proxy_ref_energy(ph, ps)

# shift minima to 0.0 for both
vqe_grid -= vqe_grid.min()
ref_grid -= ref_grid.min()

# save arrays
np.save(os.path.join(HERE, "vqe_energy_grid.npy"), vqe_grid)
np.save(os.path.join(HERE, "dft_energy_grid.npy"), ref_grid)
np.save(os.path.join(HERE, "phi_values.npy"), phi)
np.save(os.path.join(HERE, "psi_values.npy"), psi)

print("Saved:", {
    "vqe_energy_grid.npy": vqe_grid.shape,
    "dft_energy_grid.npy": ref_grid.shape,
    "phi_values.npy": phi.shape,
    "psi_values.npy": psi.shape
})
