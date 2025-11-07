import numpy as np
import os
from Bio.PDB import PDBParser, PDBIO

# Inputs from Phase 2
phi = np.load('../02_quantum_pes/phi_values.npy')
psi = np.load('../02_quantum_pes/psi_values.npy')
mask = np.load('../02_quantum_pes/low_energy_mask.npy')
basins = np.load('../02_quantum_pes/basin_labels.npy')

# Select alpha & beta indices
alpha_idx = np.where(basins == 0)[0][:4]
beta_idx = np.where(basins == 1)[0][:4]

# Fake building: copy KRAS clean pdb for now
# (Later we mutate dihedrals if needed)
src = '../01_target_prep/6OIM_clean_H.pdb'

import shutil
for i,iidx in enumerate(alpha_idx, start=1):
    shutil.copy(src, f'initial_states/quantum_init_{i}.pdb')

for i,iidx in enumerate(beta_idx, start=5):
    shutil.copy(src, f'initial_states/quantum_init_{i}.pdb')

for i in range(1,9):
    shutil.copy(src, f'initial_states/control_init_{i}.pdb')

print("Initial states created.")
