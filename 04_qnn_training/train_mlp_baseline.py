#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import json, os
X = np.load("data/qnn_input_features_norm.npy")
y = np.load("data/qnn_labels_norm.npy")
kf = KFold(5, shuffle=True, random_state=12345)
rows=[]
fold=0
for tr,te in kf.split(X):
    fold+=1
    mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=fold)
    mlp.fit(X[tr], y[tr])
    yp = mlp.predict(X[te])
    rmse = mean_squared_error(y[te], yp, squared=False) if 'squared' in mean_squared_error.__code__.co_varnames else ( ( (y[te]-yp)**2 ).mean()**0.5 )
    rows.append((fold,rmse, sum(p.size for p in mlp.coefs_)+sum(p.size for p in mlp.intercepts_)))
import csv
with open("metrics/perf_mlp_baseline.csv","w") as f:
    w=csv.writer(f); w.writerow(["fold","rmse","params"]); w.writerows(rows)
print("Saved metrics/perf_mlp_baseline.csv")
