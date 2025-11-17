#!/usr/bin/env python3
"""
combine_figure3.py — improved layout
Creates a composite where:
 - Figure 3A spans full width
 - Figures 3B and 3C share the middle row
 - Figure 3D spans full width

Outputs:
 - figures/figure_3_combined.png
 - figures/figure_3_combined.svg
 - figures/figure_3_combined.pdf
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(__file__) or "."
FIGDIR = os.path.join(ROOT, "figures")

# Input images
F3A = os.path.join(FIGDIR, "figure_3a_architecture_comparison.png")
F3B = os.path.join(FIGDIR, "figure_3b_parity_plot.png")
F3C = os.path.join(FIGDIR, "figure_3c_training_convergence.png")
F3D = os.path.join(FIGDIR, "figure_3d_parameter_efficiency.png")

imgs = {
    "A": mpimg.imread(F3A),
    "B": mpimg.imread(F3B),
    "C": mpimg.imread(F3C),
    "D": mpimg.imread(F3D),
}

plt.rcParams["font.family"] = "Arial"
plt.figure(figsize=(12, 14))

# -----------------------------
# FULL WIDTH — FIGURE 3A
# -----------------------------
axA = plt.subplot2grid((3, 2), (0, 0), colspan=2)
axA.imshow(imgs["A"])
axA.set_title("Figure 3A — Model Architecture", fontsize=16, weight="bold", loc='left')
axA.axis("off")

# -----------------------------
# MIDDLE ROW — 3B and 3C
# -----------------------------
axB = plt.subplot2grid((3, 2), (1, 0))
axB.imshow(imgs["B"])
axB.set_title("Figure 3B — Parity Plot", fontsize=14, weight="bold", loc='left')
axB.axis("off")

axC = plt.subplot2grid((3, 2), (1, 1))
axC.imshow(imgs["C"])
axC.set_title("Figure 3C — Training Convergence", fontsize=14, weight="bold", loc='left')
axC.axis("off")

# -----------------------------
# FULL WIDTH — FIGURE 3D
# -----------------------------
axD = plt.subplot2grid((3, 2), (2, 0), colspan=2)
axD.imshow(imgs["D"])
axD.set_title("Figure 3D — Parameter Efficiency", fontsize=16, weight="bold", loc='left')
axD.axis("off")

plt.tight_layout()

# SAVE OUTPUTS
out_png = os.path.join(FIGDIR, "figure_3_combined.png")
out_svg = os.path.join(FIGDIR, "figure_3_combined.svg")
out_pdf = os.path.join(FIGDIR, "figure_3_combined.pdf")

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_svg, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")

print("DONE — wrote:")
print(" ", out_png)
print(" ", out_svg)
print(" ", out_pdf)
