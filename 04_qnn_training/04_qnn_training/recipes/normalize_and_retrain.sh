#!/usr/bin/env bash
set -euo pipefail
cd ~/QGEN-ProDyn/04_qnn_training

python - <<'PY'
import numpy as np, json, os
from sklearn.preprocessing import StandardScaler
X = np.load("data/qnn_input_features.npy")
y = np.load("data/qnn_labels.npy").reshape(-1,1)
scX = StandardScaler().fit(X)
scY = StandardScaler().fit(y)
np.save("data/scaler_X_mean.npy", scX.mean_)
np.save("data/scaler_X_scale.npy", scX.scale_)
np.save("data/scaler_y_mean.npy", scY.mean_)
np.save("data/scaler_y_scale.npy", scY.scale_)
# write JSON for re-use
with open("data/scalers.json","w") as f:
    json.dump({"X_mean":scX.mean_.tolist(),"X_scale":scX.scale_.tolist(),
               "y_mean":scY.mean_.tolist(),"y_scale":scY.scale_.tolist()},f)
print("Saved scalers -> data/scalers.json")
# overwrite normalized dataset used by training
Xn = scX.transform(X)
yn = scY.transform(y).reshape(-1)
np.save("data/qnn_input_features_norm.npy", Xn)
np.save("data/qnn_labels_norm.npy", yn)
print("Wrote normalized data: data/qnn_input_features_norm.npy, qnn_labels_norm.npy")
PY

# quick retrain: modify hybrid_qnn to use normalized data and fewer epochs for smoke test
python hybrid_qnn.py
