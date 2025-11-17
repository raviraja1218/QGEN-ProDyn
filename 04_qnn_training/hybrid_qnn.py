#!/usr/bin/env python3
"""
hybrid_qnn.py (improved)

Usage:
  conda activate qgen-prodyn
  cd ~/QGEN-ProDyn/04_qnn_training
  python hybrid_qnn.py

Notes:
 - Will prefer normalized data files if present:
     data/qnn_input_features_norm.npy
     data/qnn_labels_norm.npy
   Falls back to raw:
     data/qnn_input_features.npy
     data/qnn_labels.npy
 - If data/scalers.json exists and contains keys "y_mean" and "y_scale",
   the script will also compute RMSE in original units and save to metrics/.
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import ZZFeatureMap, TwoLocal

# Prefer modern estimator primitive when available
try:
    from qiskit.primitives import StatevectorEstimator as QiskitEstimator
    _ESTIMATOR_NAME = "StatevectorEstimator"
except Exception:
    from qiskit.primitives import Estimator as QiskitEstimator
    _ESTIMATOR_NAME = "Estimator"

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# -------------------------
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
OUT_MODELS = os.path.join(ROOT, "models")
OUT_METRICS = os.path.join(ROOT, "metrics")
os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_METRICS, exist_ok=True)

# -------------------------
# Hyperparams (tweak here)
n_qubits = 8
ansatz_reps = 1      # reduce to 1 if training is unstable / overparameterized
featuremap_reps = 1
epochs = 50
batch_size = 16
lr = 1e-3
seed = 12345

torch.manual_seed(seed)
np.random.seed(seed)

# -------------------------
# Load dataset (prefer normalized versions)
X_norm_path = os.path.join(DATA_DIR, "qnn_input_features_norm.npy")
y_norm_path = os.path.join(DATA_DIR, "qnn_labels_norm.npy")
X_raw_path  = os.path.join(DATA_DIR, "qnn_input_features.npy")
y_raw_path  = os.path.join(DATA_DIR, "qnn_labels.npy")

if os.path.exists(X_norm_path) and os.path.exists(y_norm_path):
    print("[data] Loading NORMALIZED features/labels")
    X = np.load(X_norm_path)
    y = np.load(y_norm_path)
    data_normalized = True
else:
    print("[data] Normalized files not found; loading RAW features/labels")
    X = np.load(X_raw_path)
    y = np.load(y_raw_path)
    data_normalized = False

X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float).reshape(-1)

N, input_dim = X.shape
assert N == y.shape[0], "Mismatch: X and y number of samples"

print(f"[data] samples={N}, input_dim={input_dim}, labels_shape={y.shape}, normalized={data_normalized}")

# Optionally load scaler for labels to un-normalize later (for physical RMSE)
scalers_path = os.path.join(DATA_DIR, "scalers.json")
scalers = None
if os.path.exists(scalers_path):
    try:
        with open(scalers_path, "r") as fh:
            scalers = json.load(fh)
        print("[data] loaded scalers.json (will compute RMSE in original units if possible)")
    except Exception as e:
        print("[data] failed to load scalers.json:", e)
        scalers = None

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[device]", device)

# -------------------------
# Build feature map + ansatz
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=featuremap_reps)
ansatz = TwoLocal(num_qubits=n_qubits, rotation_blocks='ry',
                  entanglement_blocks='cz', reps=ansatz_reps)

# explicit QRegister + circuit append
qr = QuantumRegister(n_qubits, 'q')
qc = QuantumCircuit(qr)
qc.append(feature_map.to_instruction(), qr)
qc.append(ansatz.to_instruction(), qr)

# Parameter lists
input_params = list(feature_map.parameters)
weight_params = list(ansatz.parameters)

print("[qnn] using estimator primitive:", _ESTIMATOR_NAME)
print("[qnn] qubits:", n_qubits)
print("[qnn] featuremap reps:", featuremap_reps, "ansatz reps:", ansatz_reps)
print("[qnn] input_params:", len(input_params))
print("[qnn] weight_params:", len(weight_params))
print("[qnn] total circuit params (qc):", len(qc.parameters))

# -------------------------
# Hybrid model definition (qnn will be attached per-fold)
class HybridModel(nn.Module):
    def __init__(self, in_dim, n_qubits, qnn_connector):
        super().__init__()
        # simple projector: input_dim -> n_qubits
        self.project = nn.Linear(in_dim, n_qubits)
        self.activation = nn.Tanh()
        # qnn_connector is expected to be a TorchConnector wrapping EstimatorQNN
        self.qnn = qnn_connector
        # readout: qnn returns shape (batch, 1) -> linear -> (batch,1)
        self.readout = nn.Linear(1, 1)

    def forward(self, x):
        # x: (batch, in_dim)
        z = self.project(x)          # (batch, n_qubits)
        z = self.activation(z)
        q_out = self.qnn(z)          # (batch, 1)
        out = self.readout(q_out)    # (batch, 1)
        return out

# -------------------------
# Utility training helpers
loss_fn = nn.MSELoss()

def train_batch(model, opt, Xb, yb):
    model.train()
    opt.zero_grad()
    pred = model(Xb)
    loss = loss_fn(pred, yb)
    loss.backward()
    opt.step()
    return float(loss.item())

def eval_preds(model, Xv):
    model.eval()
    with torch.no_grad():
        out = model(Xv).cpu().numpy().reshape(-1)
    return out

# -------------------------
# CV loop - build fresh EstimatorQNN & TorchConnector per fold
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
fold = 0
fold_results_norm = []
fold_results_orig = []

for train_idx, test_idx in kf.split(X):
    fold += 1
    print(f"\n[Fold {fold}] Starting...")

    # instantiate primitive + EstimatorQNN per-fold to avoid cross-fold state
    estimator = QiskitEstimator()
    qnn = EstimatorQNN(
        estimator=estimator,
        circuit=qc,
        input_params=input_params,
        weight_params=weight_params,
        input_gradients=True
    )

    # TorchConnector wrapper
    torch_qnn = TorchConnector(qnn).to(device)

    # build model with this connector
    model = HybridModel(input_dim, n_qubits, torch_qnn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # prepare tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    X_train = X_t[train_idx].to(device)
    y_train = y_t[train_idx].to(device)
    X_test  = X_t[test_idx].to(device)
    y_test  = y_t[test_idx].to(device)

    ntrain = len(train_idx)
    if ntrain == 0:
        print(f"[Fold {fold}] no training samples, skipping")
        continue

    # epoch loop
    for ep in range(1, epochs + 1):
        perm = torch.randperm(ntrain)
        epoch_loss = 0.0
        for i in range(0, ntrain, batch_size):
            batch_idx = perm[i:i+batch_size]
            Xb = X_train[batch_idx]
            yb = y_train[batch_idx]
            l = train_batch(model, opt, Xb, yb)
            epoch_loss += l
        if ep == 1 or ep % 10 == 0:
            avg_loss = epoch_loss / max(1, int((ntrain + batch_size - 1)//batch_size))
            print(f" Fold {fold} | Epoch {ep:03d} train_loss={avg_loss:.4f}")

    # Evaluate on test set (normalized space)
    y_pred_norm = eval_preds(model, X_test)                # normalized (or raw) predictions
    y_true_norm = y[test_idx]                              # numpy array

    # explicit RMSE (safe across sklearn versions)
    rmse_norm = float(np.sqrt(np.mean((y_true_norm - y_pred_norm) ** 2)))
    print(f" Fold {fold} RMSE (dataset units): {rmse_norm:.6f}")

    # If scalers available and data was normalized, compute RMSE in original units
    rmse_orig = None
    if scalers is not None:
        try:
            # expect scalers to contain at least "y_mean" and "y_scale" (scalar or single-element list)
            y_mean = float(np.array(scalers.get("y_mean")).flatten()[0])
            y_scale = float(np.array(scalers.get("y_scale")).flatten()[0])
            # If labels were normalized as (y - mean)/scale
            y_pred_orig = y_pred_norm * y_scale + y_mean
            y_true_orig = y_true_norm * y_scale + y_mean
            rmse_orig = float(np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2)))
            print(f" Fold {fold} RMSE (original units): {rmse_orig:.6f}")
        except Exception as e:
            print("[warn] failed to un-normalize using scalers.json:", e)
            rmse_orig = None

    # Save model state (state_dict)
    save_path = os.path.join(OUT_MODELS, f"hybrid_qnn_fold{fold}.pt")
    torch.save(model.state_dict(), save_path)
    print(f" Saved model -> {save_path}")

    # record results
    fold_results_norm.append(rmse_norm)
    fold_results_orig.append(rmse_orig if rmse_orig is not None else np.nan)

# -------------------------
# Summary & save
fold_results_norm = np.array(fold_results_norm)
fold_results_orig = np.array(fold_results_orig)

print("\nCV RMSEs (dataset units):", fold_results_norm.tolist())
print("Mean RMSE (dataset units):", float(np.mean(fold_results_norm)))
np.savetxt(os.path.join(OUT_METRICS, "qnn_cv_rmse_dataset_units.txt"), fold_results_norm, fmt="%.6f")

if not np.all(np.isnan(fold_results_orig)):
    print("CV RMSEs (original units):", np.nan_to_num(fold_results_orig).tolist())
    print("Mean RMSE (original units):", float(np.nanmean(fold_results_orig)))
    np.savetxt(os.path.join(OUT_METRICS, "qnn_cv_rmse_original_units.txt"), fold_results_orig, fmt="%.6f")
else:
    print("No original-units RMSEs (scalers.json missing or incomplete).")

# also write a small JSON summary
summary = {
    "n_qubits": n_qubits,
    "featuremap_reps": featuremap_reps,
    "ansatz_reps": ansatz_reps,
    "epochs": epochs,
    "batch_size": batch_size,
    "lr": lr,
    "seed": seed,
    "fold_results_dataset_units": fold_results_norm.tolist(),
    "fold_results_original_units": [None if np.isnan(x) else float(x) for x in fold_results_orig.tolist()]
}
with open(os.path.join(OUT_METRICS, "qnn_cv_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

print("\n✅ Done. Results saved to metrics/ (qnn_cv_rmse_*.txt and qnn_cv_summary.json)")
