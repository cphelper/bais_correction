"""Generate the complete proposed methodology workflow diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 20))
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis("off")

c_data = "#4C72B0"
c_preproc = "#DD8452"
c_model = "#55A868"
c_eval = "#C44E52"
c_output = "#8172B3"
c_arrow = "#333333"
c_bg = "#F7F7F7"

def box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.85)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="white",
            wrap=True, linespacing=1.4)

def arrow(ax, x1, y1, x2, y2, text="", color=c_arrow):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, text, fontsize=7.5, color="#555555", style="italic")

def section_bg(ax, x, y, w, h, title, color):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                          facecolor=color, edgecolor="gray", linewidth=0.8, alpha=0.12)
    ax.add_patch(rect)
    ax.text(x + 0.3, y + h - 0.35, title, fontsize=10, fontweight="bold",
            color=color, alpha=0.9)

# ── Title ──
ax.text(8, 21.5, "Proposed Methodology: Complete Workflow", ha="center",
        fontsize=15, fontweight="bold", color="#222222")

# ══════════════════════════════════════════════════════════
# STAGE 1: DATA ACQUISITION
# ══════════════════════════════════════════════════════════
section_bg(ax, 0.3, 18.3, 15.4, 2.8, "STAGE 1: Data Acquisition", c_data)

box(ax, 1, 19.3, 4.5, 1.2,
    "Satellite Altimeter SLA\n(CMEMS L4 gridded)\n0.25° × 0.25°, monthly\n72 files: 2015–2020",
    c_data, fontsize=8.5, bold=True)

box(ax, 6, 19.3, 4.5, 1.2,
    "CMIP6 Model ZOS\n(MPI-ESM1-2-LR, SSP2-4.5)\nCurvilinear grid (13×23)\nMonthly: 2015–2020",
    c_data, fontsize=8.5, bold=True)

box(ax, 11, 19.3, 4, 1.2,
    "Study Domain\n60°–80°N, 50°–110°E\n(Arctic / sub-Arctic)\nHigh-latitude ocean",
    "#7A7A7A", fontsize=8.5, bold=True)

# ── Stage 1 label ──
ax.text(1.5, 18.7, "SARAL/AltiKa, Sentinel-3,\nJason-3, CryoSat-2", fontsize=7,
        color="#666", style="italic")

# ══════════════════════════════════════════════════════════
# STAGE 2: PREPROCESSING
# ══════════════════════════════════════════════════════════
section_bg(ax, 0.3, 15.0, 15.4, 3.0, "STAGE 2: Preprocessing & Feature Engineering", c_preproc)

arrow(ax, 3.25, 19.3, 3.25, 17.8)
arrow(ax, 8.25, 19.3, 8.25, 17.8)

box(ax, 1, 16.6, 4.5, 1.0,
    "Spatial Subsetting\nSlice to ROI (60–80°N, 50–110°E)\nMonthly resampling (1MS mean)",
    c_preproc, fontsize=8.5)

box(ax, 6, 16.6, 4.5, 1.0,
    "ZOS → SLA Conversion\nSubtract 2015–2018 mean\nSLA(t) = ZOS(t) – ZOS̄₂₀₁₅₋₂₀₁₈",
    c_preproc, fontsize=8.5)

box(ax, 11, 16.6, 4, 1.0,
    "Regridding\ncKDTree nearest-neighbor\nCurvilinear → Rectilinear\n(13×23) → (80×240)",
    c_preproc, fontsize=8.5)

arrow(ax, 5.5, 17.1, 6.0, 17.1)
arrow(ax, 10.5, 17.1, 11.0, 17.1)

# Feature table
arrow(ax, 8.25, 16.6, 8.25, 15.8)
box(ax, 4.5, 15.3, 7.5, 0.5,
    "Feature Table:  [model_sla, lat, lon, month] → obs_sla  |  23,524 valid samples  |  Drop NaN",
    c_preproc, fontsize=8.5, bold=True)

# ══════════════════════════════════════════════════════════
# STAGE 3: DATA SPLITTING
# ══════════════════════════════════════════════════════════
section_bg(ax, 0.3, 12.8, 15.4, 2.0, "STAGE 3: Temporal Data Splitting", "#7A7A7A")

arrow(ax, 8.25, 15.3, 8.25, 14.6)

box(ax, 1, 13.2, 4.0, 1.0,
    "Training Set\n2015–2017\n11,808 samples\n(50.2%)",
    "#4C72B0", fontsize=9, bold=True)

box(ax, 5.5, 13.2, 4.0, 1.0,
    "Calibration Set\n2018\n3,734 samples\n(15.9%)",
    "#DD8452", fontsize=9, bold=True)

box(ax, 10.5, 13.2, 4.5, 1.0,
    "Test Set\n2019–2020\n7,982 samples\n(33.9%)",
    "#C44E52", fontsize=9, bold=True)

# ══════════════════════════════════════════════════════════
# STAGE 4: MODEL TRAINING (two branches)
# ══════════════════════════════════════════════════════════
section_bg(ax, 0.3, 8.5, 7.2, 4.0, "STAGE 4A: Deterministic Correction", c_model)
section_bg(ax, 8.0, 8.5, 7.7, 4.0, "STAGE 4B: Probabilistic PI Estimation", c_model)

arrow(ax, 3.0, 13.2, 3.0, 12.3)
arrow(ax, 12.75, 13.2, 12.75, 12.3)

box(ax, 0.8, 11.2, 6.4, 1.0,
    "Random Forest Regressor\n50 trees, max_depth=10\nFeatures → obs_sla (point prediction)",
    c_model, fontsize=8.5, bold=True)

box(ax, 8.3, 11.2, 7.1, 1.0,
    "Tube Loss Neural Network\nTrunk(128→64→32) + 3 heads (base, half-width, asymmetry)\nOutputs: [upper, lower] bounds",
    c_green := "#2E8B57", fontsize=8.5, bold=True)

arrow(ax, 3.0, 11.2, 3.0, 10.5)
arrow(ax, 12.75, 11.2, 12.75, 10.5)

box(ax, 0.8, 9.5, 6.4, 0.9,
    "Train on 2015–2017\nPredict corrected SLA\nSave model (.pkl)",
    c_model, fontsize=8.5)

box(ax, 8.3, 10.0, 3.3, 0.8,
    "Tube Loss\nq=0.90/0.95\nr=0.5, δ=0.0005",
    c_green, fontsize=8)
box(ax, 12.0, 10.0, 3.3, 0.8,
    "QD Loss (baseline)\nPinball τ_u, τ_l\nSame architecture",
    "#8172B3", fontsize=8)

arrow(ax, 10.0, 10.0, 10.0, 9.3)
arrow(ax, 13.6, 10.0, 13.6, 9.3)

box(ax, 8.3, 8.8, 7.1, 0.45,
    "Adam(lr=1e-3) + ReduceLROnPlateau + Early Stop (patience 60)",
    "#666666", fontsize=7.5)

# ══════════════════════════════════════════════════════════
# STAGE 5: CONFORMAL CALIBRATION
# ══════════════════════════════════════════════════════════
section_bg(ax, 0.3, 5.8, 15.4, 2.4, "STAGE 5: Conformal Calibration & Evaluation", c_eval)

arrow(ax, 3.0, 9.5, 3.0, 8.0)
arrow(ax, 12.75, 8.8, 12.75, 8.0)

box(ax, 0.8, 6.7, 6.4, 1.0,
    "Evaluate on Test (2019–2020)\nRMSE, MAE, Bias, R²\nSpatial bias maps",
    c_eval, fontsize=8.5, bold=True)

box(ax, 8.3, 7.15, 7.1, 0.55,
    "Conformal Calibration on 2018 held-out set\nsᵢ = max(f̂₂(xᵢ) − yᵢ,  yᵢ − f̂₁(xᵢ))  →  margin q̂",
    c_eval, fontsize=8, bold=True)

arrow(ax, 12.75, 7.15, 12.75, 6.7)

box(ax, 8.3, 6.15, 7.1, 0.5,
    "Calibrated PI:  [f̂₂(x) − q̂,  f̂₁(x) + q̂]  →  PICP, MPIW evaluation",
    c_eval, fontsize=8.5)

# ══════════════════════════════════════════════════════════
# STAGE 6: OUTPUTS
# ══════════════════════════════════════════════════════════
section_bg(ax, 0.3, 2.5, 15.4, 3.0, "STAGE 6: Results & Outputs", c_output)

arrow(ax, 3.0, 6.15, 3.0, 5.3)
arrow(ax, 12.75, 6.15, 12.75, 5.3)

box(ax, 0.5, 4.2, 3.2, 1.0,
    "Corrected SLA\nRMSE: 0.058m\n(44.2% ↓)",
    c_output, fontsize=9, bold=True)

box(ax, 4.0, 4.2, 3.2, 1.0,
    "Bias Maps\nBias: +0.005m\n(94.7% eliminated)",
    c_output, fontsize=9, bold=True)

box(ax, 7.8, 4.2, 3.5, 1.0,
    "90% PI\nPICP: 89.3%\nMPIW: 0.169m",
    c_output, fontsize=9, bold=True)

box(ax, 11.8, 4.2, 3.5, 1.0,
    "95% PI\nPICP: 92.3%\nMPIW: 0.211m",
    c_output, fontsize=9, bold=True)

# Bottom results
box(ax, 0.5, 2.8, 7.0, 1.0,
    "Time series plots • Scatter analysis\nPer-month demo • Feature importance",
    "#666666", fontsize=9)

box(ax, 8.0, 2.8, 7.2, 1.0,
    "QD vs Tube Loss comparison\nMonthly coverage analysis\nPICP/MPIW bar charts",
    "#666666", fontsize=9)

# ── Title annotations ──
ax.text(8, 0.8, "Fig. 1: Complete workflow of the proposed satellite-guided ML framework for\n"
        "CMIP6 SLA bias correction and Tube Loss based probabilistic prediction interval estimation.",
        ha="center", fontsize=10, style="italic", color="#444444")

plt.tight_layout()
plt.savefig("sla_plots/workflow_diagram.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved → sla_plots/workflow_diagram.png")
