# analyze_rmsd.py
# Compute RMSD time series per group (quantum vs control) and export group means for Figure 2C.

import os, glob
import numpy as np
import mdtraj as md

ROOT = os.path.dirname(__file__)
METRICS = os.path.join(ROOT, "metrics")
TOP = os.path.join(ROOT, "topologies")
TRJ = os.path.join(ROOT, "trajectories")
os.makedirs(METRICS, exist_ok=True)

def collect(group):
    tops = sorted(glob.glob(os.path.join(TOP, f"{group}_*_topology.pdb")))
    dcds = sorted(glob.glob(os.path.join(TRJ, f"{group}_*.dcd")))
    if len(tops) == 0 or len(dcds) == 0:
        print(f"[analyze_rmsd] No topologies/trajectories for group={group}")
        return None
    curves = []
    # pick first topology as alignment ref
    ref = md.load(tops[0])
    ref = ref.center_coordinates()
    for topo, dcd in zip(tops, dcds):
        trj = md.load(dcd, top=topo).center_coordinates().superpose(ref)
        rmsd = md.rmsd(trj, ref)  # nm
        curves.append(rmsd)
    # pad shorter series with last value so we can average
    maxlen = max(len(c) for c in curves)
    padded = []
    for c in curves:
        if len(c) < maxlen:
            last = c[-1]
            pad = np.pad(c, (0, maxlen-len(c)), constant_values=last)
            padded.append(pad)
        else:
            padded.append(c)
    return np.vstack(padded)

for group in ["quantum","control"]:
    arr = collect(group)
    if arr is None:
        # write empty to keep downstream scripts happy
        np.savetxt(os.path.join(METRICS, f"convergence_curves_{group}.csv"),
                   np.array([]), delimiter=",")
    else:
        mean_curve = arr.mean(axis=0)       # nm
        # convert to Å for the paper (1 nm = 10 Å)
        mean_curve_ang = mean_curve*10.0
        np.savetxt(os.path.join(METRICS, f"convergence_curves_{group}.csv"),
                   mean_curve_ang, delimiter=",")
print("[analyze_rmsd] Done.")
