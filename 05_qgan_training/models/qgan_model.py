#!/usr/bin/env python3
"""
qgan_model.py
Defines:
 - ClassicalGenerator (MLP)
 - HybridQuantumGenerator (classical projector -> small QNN readout -> expand)
 - Discriminator (MLP)

This is a minimal, resilient implementation suitable for running locally.
"""
import os, math
import torch
import torch.nn as nn
import numpy as np

# Try to import qiskit connector; if unavailable, we will fallback to classical generator
_qiskit_ok = False
try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import ZZFeatureMap, TwoLocal
    from qiskit.primitives import Estimator as QiskitEstimator
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    _qiskit_ok = True
except Exception as e:
    # print("Qiskit not available:", e)
    _qiskit_ok = False

class ClassicalGenerator(nn.Module):
    def __init__(self, noise_dim, cond_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        return self.net(x)

class HybridQuantumGenerator(nn.Module):
    """
    Hybrid generator: classical projector -> small QNN scalar -> expand to embedding
    If Qiskit not present, falls back to ClassicalGenerator.
    """
    def __init__(self, noise_dim, cond_dim, out_dim, n_qubits=5):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.n_qubits = n_qubits

        # classical projector
        self.project = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, n_qubits),
            nn.Tanh()
        )

        # post-quantum expand
        self.expand = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

        if _qiskit_ok:
            # Build simple QNN (used as a differentiable layer that maps n_qubits->1)
            feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
            ansatz = TwoLocal(num_qubits=n_qubits, rotation_blocks='ry', entanglement_blocks='cz', reps=1)
            qr = QuantumRegister(n_qubits, 'q')
            qc = QuantumCircuit(qr)
            qc.append(feature_map.to_instruction(), qr)
            qc.append(ansatz.to_instruction(), qr)
            estimator = QiskitEstimator()
            input_params = list(feature_map.parameters)
            weight_params = list(ansatz.parameters)
            qnn = EstimatorQNN(estimator=estimator, circuit=qc,
                               input_params=input_params,
                               weight_params=weight_params,
                               input_gradients=True)
            self.qnn = TorchConnector(qnn)
        else:
            self.qnn = None

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        proj = self.project(x)  # -> (batch, n_qubits)
        if self.qnn is not None:
            # QNN expects shape (batch, n_qubits) for inputs; returns (batch, 1)
            qout = self.qnn(proj)  # tensor (batch,1)
            # tile qout channel across n_qubits and combine with proj
            qtile = qout.repeat(1, max(1, proj.shape[1]//qout.shape[1]))
            combined = proj + 0.3 * qtile[:, :proj.shape[1]]
        else:
            combined = proj
        return self.expand(combined)

class Discriminator(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x, c):
        h = torch.cat([x, c], dim=1)
        return self.net(h)
