#!/usr/bin/env python3
"""
define_qnn_models.py
Defines architecture metadata for QNN and MLP baseline.
"""
import json, os
os.makedirs("models", exist_ok=True)

qnn = {
    "qubits": 8,
    "encoding": {"type": "ZZFeatureMap", "depth": 2},
    "ansatz": {"type": "TwoLocal", "entanglement": "linear", "reps": 2},
    "trainable_params": 112,
    "optimizer": "Adam",
    "lr": 1e-3,
    "loss": "MSE"
}

mlp = {
    "layers": [32, 64, 32, 1],
    "activation": "relu",
    "optimizer": "Adam",
    "lr": 1e-3,
    "loss": "MSE",
    "params": 760
}

json.dump(qnn, open("models/qnn_architecture.json", "w"), indent=2)
json.dump(mlp, open("models/mlp_architecture.json", "w"), indent=2)
print("✅ Model JSONs saved: qnn_architecture.json, mlp_architecture.json")
