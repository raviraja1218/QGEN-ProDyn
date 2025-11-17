#!/usr/bin/env python3
"""
train_qnn.py
Trains both Quantum Neural Network (QNN) and MLP baselines on Phase-4 dataset.
Handles k-fold CV, early stopping, and metric logging.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

# --- setup ---
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# --- load data ---
X = np.load("data/qnn_input_features.npy")
y = np.load("data/qnn_labels.npy")
y = y.reshape(-1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- simple baseline MLP (shared architecture) ---
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[64,32], out_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], out_dim)
        )
    def forward(self, x): return self.model(x)

# --- mock QNN (same interface, fewer params to simulate quantum compression) ---
class QNNMock(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 8)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(8, out_dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

# --- training loop ---
def train_model(model, X_train, y_train, X_val, y_val, epochs=200, patience=15):
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_val = 1e9
    stop_counter = 0
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        history.append([epoch, loss.item(), val_loss])
        if val_loss < best_val:
            best_val = val_loss
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter >= patience:
            break
    return history, best_val

# --- 5-fold cross validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
fold = 0
records = []

for train_idx, test_idx in kf.split(X):
    fold += 1
    print(f"\n[Fold {fold}]")

    # split
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_tr, X_val = X_train[:-2], X_train[-2:]
    y_tr, y_val = y_train[:-2], y_train[-2:]

    for model_name, Net in [("QNN", QNNMock), ("MLP", MLP)]:
        print(f" → Training {model_name} ...")
        model = Net(in_dim=X.shape[1]).to(device)
        t0 = time.time()
        hist, best_val = train_model(model, X_tr, y_tr, X_val, y_val)
        dur = time.time()-t0

        # evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        params = sum(p.numel() for p in model.parameters())

        records.append([fold, model_name, rmse, mae, r2, len(hist), params, dur])
        torch.save(model.state_dict(), f"models/{model_name.lower()}_fold{fold}.pt")

        pd.DataFrame(hist, columns=["epoch","train_loss","val_loss"]).to_csv(f"metrics/training_history_{model_name}_fold{fold}.csv", index=False)

# --- summary ---
df = pd.DataFrame(records, columns=["fold","model","RMSE","MAE","R2","epochs","params","train_time_s"])
df.to_csv("metrics/performance_summary.csv", index=False)
print("✅ Saved metrics/performance_summary.csv")

print("\nMean Results:\n", df.groupby("model")[["RMSE","R2","params"]].mean())
