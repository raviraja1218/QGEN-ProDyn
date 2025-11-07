# analyze_ensemble.py
# Compute ensemble quality metrics on the "equilibrated" window (last 50% of frames):
# - φ/ψ 2D histogram → KL divergence (quantum || control)
# - Radius of gyration (Rg)
# - Native contacts Q against first frame of each traj
# Saves CSV metrics + Figure 2D summary.

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os, json

os.makedirs("metrics", exist_ok=True)

def load_group(group):
    tops = sorted([f"topologies/{t}" for t in os.listdir("topologies") if t.startswith(group)])
    if not tops: return []
    top_ref = tops[0]
    trajs = []
    for dcd in sorted([f"trajectories/{f}" for f in os.listdir("trajectories") if f.startswith(f"{group}_traj_") and f.endswith(".dcd")]):
        try:
            trj = md.load(dcd, top=top_ref).center_coordinates()
            trajs.append(trj)
        except Exception as e:
            print(f"[ensemble] Skip {dcd}: {e}")
    return trajs

def eq_window(trj):
    start = trj.n_frames // 2
    return trj[start:]

def phi_psi_hist(trj, bins=36):
    phi = md.compute_phi(trj)[1]  # radians
    psi = md.compute_psi(trj)[1]
    ang = np.concatenate([phi, psi], axis=1)  # (frames, n-1 + n-1) but we just histogram pairwise mean per frame
    # Simpler: use all phi and psi separately into 2D hist
    phi_vals = phi.flatten()
    psi_vals = psi.flatten()
    H, xedges, yedges = np.histogram2d(
        np.degrees(phi_vals), np.degrees(psi_vals),
        bins=bins, range=[[-180,180],[-180,180]], density=True
    )
    # Add epsilon to avoid zeros (for KL)
    H = H + 1e-12
    H /= H.sum()
    return H, xedges, yedges

def kl_div(P, Q):
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(np.sum(P * (np.log(P) - np.log(Q))))

def rg_mean(trj):
    return float(md.compute_rg(trj).mean())

def q_native(trj):
    # native contacts against first frame of the SAME traj
    pairs = md.geometry.contact._contact_pairs(trj.topology, scheme='ca')
    ref = trj[0]
    q = md.compute_contacts(trj, contacts=pairs, scheme='ca')[0]
    q_ref = md.compute_contacts(ref, contacts=pairs, scheme='ca')[0]
    # fraction of contacts within +0.5 nm of native
    thr = 0.5  # nm
    frac = (np.abs(q - q_ref) < thr).mean()
    return float(frac)

def group_metrics(group):
    trajs = load_group(group)
    if not trajs: return None
    # concat equilibrated windows
    eq_trajs = [eq_window(t) for t in trajs if t.n_frames >= 4]
    if not eq_trajs: return None
    cat = eq_trajs[0]
    for t in eq_trajs[1:]:
        cat = cat.join(t)
    H, _, _ = phi_psi_hist(cat)
    rg = rg_mean(cat)
    qn = q_native(eq_trajs[0])  # approximate per-group with first
    return {"H": H, "Rg": rg, "Q_native": qn}

qm = group_metrics("quantum")
cm = group_metrics("control")

if qm and cm:
    KL = kl_div(qm["H"], cm["H"])
    out = {
        "KL_phi_psi_quantum||control": KL,
        "Rg_mean_quantum": qm["Rg"],
        "Rg_mean_control": cm["Rg"],
        "Q_native_quantum": qm["Q_native"],
        "Q_native_control": cm["Q_native"]
    }
    with open("metrics/ensemble_quality.json","w") as f:
        json.dump(out, f, indent=2)
    print("[ensemble] Wrote metrics/ensemble_quality.json")

    # Simple Figure 2D: bar panel of KL, Rg, Q
    fig, ax = plt.subplots(1,3, figsize=(9,3))
    ax[0].bar([0],[KL]); ax[0].set_title("KL(φ/ψ) Q||C"); ax[0].set_xticks([])
    ax[1].bar([0,1],[qm["Rg"], cm["Rg"]])
    ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["Q","C"]); ax[1].set_title("Rg mean (nm)")
    ax[2].bar([0,1],[qm["Q_native"], cm["Q_native"]])
    ax[2].set_xticks([0,1]); ax[2].set_xticklabels(["Q","C"]); ax[2].set_title("Q_native")
    plt.tight_layout()
    plt.savefig("figure_2d_ensemble_quality.png", dpi=300)
    plt.savefig("figure_2d_ensemble_quality.svg")
    print("[ensemble] Wrote figure_2d_ensemble_quality.*")
else:
    print("[ensemble] Not enough trajectories to compute ensemble metrics.")
