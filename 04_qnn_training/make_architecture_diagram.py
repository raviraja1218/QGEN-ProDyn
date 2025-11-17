#!/usr/bin/env python3
"""
make_architecture_diagram.py — improved layout & spacing

Creates:
  figures/figure_3a_architecture_comparison.png
  figures/figure_3a_architecture_comparison.svg
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(__file__) or "."
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Use normalized axes (0..1) for robust scaling
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def box_norm(cx, cy, w, h, text, fontsize=10, pad=0.02):
    """Draw a rounded box in normalized coordinates (0..1)"""
    left = cx - w/2
    bottom = cy - h/2
    rect = FancyBboxPatch(
        (left, bottom), w, h,
        boxstyle="round,pad=0.02", ec="black", fc="#f4f4f4", linewidth=1.2,
        transform=ax.transAxes, zorder=2
    )
    ax.add_patch(rect)
    # text in axes coords
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, wrap=True, transform=ax.transAxes, zorder=3)

def arrow_norm(x0, y0, x1, y1, dashed=False, color="black", lw=1.1):
    style = "-|>" if not dashed else "->"
    patch = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color,
        linestyle="--" if dashed else "-", transform=ax.transAxes, zorder=1
    )
    ax.add_patch(patch)

# Left column positions (normalized)
xL = 0.18
wL = 0.26
h_box = 0.16
y_top = 0.84

box_norm(xL, y_top, wL, h_box, "Input features\n(φ, ψ, descriptors)", fontsize=10)
box_norm(xL, y_top - 0.17, wL, h_box, "Linear projector\n(in_dim → 8)", fontsize=10)
box_norm(xL, y_top - 0.34, wL, h_box, "8-qubit QNN\nZZFeatureMap + TwoLocal", fontsize=10)
box_norm(xL, y_top - 0.51, wL, h_box, "Readout + Linear\n→ ΔG prediction", fontsize=10)

# arrows down the left column (small vertical offsets)
arrow_norm(xL, y_top - 0.05, xL, y_top - 0.12)
arrow_norm(xL, y_top - 0.22, xL, y_top - 0.29)
arrow_norm(xL, y_top - 0.39, xL, y_top - 0.46)

# Right column (MLP)
xR = 0.76
wR = 0.28
box_norm(xR, 0.58, wR, 0.34,
         "MLP baseline\nDense(64) → ReLU\nDense(32) → ReLU\nLinear → ΔG", fontsize=10)

# dashed comparison arrow between columns (placed between boxes, not overlapping)
arrow_norm(xL + wL/2 + 0.03, 0.50, xR - wR/2 - 0.03, 0.58, dashed=True, color="#6b6b6b", lw=1.0)
ax.text(0.5, 0.52, "Compare\nRMSE, parameter efficiency",
        ha="center", va="center", fontsize=9, color="#6b6b6b", transform=ax.transAxes, zorder=3)

# Title and caption with safe margins
ax.text(0.5, 0.96,
        "Figure 3A — Model architecture: Hybrid QNN (left) vs MLP baseline (right)",
        ha="center", va="center", fontsize=13, weight="bold", transform=ax.transAxes)

ax.text(0.5, 0.06,
        "Hybrid QNN: projector → n-qubit variational circuit (ZZFeatureMap + TwoLocal) → readout",
        ha="center", va="center", fontsize=9, transform=ax.transAxes)

# Save: tight but with small pad so nothing clipped
png = os.path.join(FIGDIR, "figure_3a_architecture_comparison.png")
svg = os.path.join(FIGDIR, "figure_3a_architecture_comparison.svg")
fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.12)
fig.savefig(svg, bbox_inches="tight", pad_inches=0.12)
plt.close(fig)
print("Wrote:", png, svg)
