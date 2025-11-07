# Phase 3 — Quantum-Informed MD (Smoke Test)

## Steps (WSL)

```bash
conda activate qgen-prodyn
conda install -y -c conda-forge openmm mdtraj pdbfixer

cd ~/QGEN-ProDyn/03_md_simulations

# Reset folders
rm -rf initial_states initial_states_fixed topologies trajectories logs metrics
mkdir -p initial_states initial_states_fixed

# Seed 8+8 initial states using the prepared KRAS (Phase 1)
for i in $(seq 1 8);  do cp ../01_target_prep/6OIM_clean_H.pdb initial_states/quantum_init_${i}.pdb; done
for i in $(seq 1 8);  do cp ../01_target_prep/6OIM_clean_H.pdb initial_states/control_init_${i}.pdb; done

# Sanitize to Only-H variant
for f in initial_states/*.pdb; do
  b=$(basename "$f")
  python sanitize_only_h.py "$f" "initial_states_fixed/$b"
done

# Run MD
python run_md.py

# RMSD analysis + Fig 2C
python analyze_rmsd.py
python plot_convergence.py
mkdir -p ../12_manuscript_assets/main_figures
cp figure_2c_convergence_curves.* ../12_manuscript_assets/main_figures/

# Ensemble quality + Fig 2D
python analyze_ensemble.py
cp figure_2d_ensemble_quality.* ../12_manuscript_assets/main_figures/
