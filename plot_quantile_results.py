import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Tube Loss Quantile Model — Performance Summary", fontsize=15, fontweight="bold", y=1.02)

levels = ["90%", "95%"]
x = np.arange(len(levels))
bar_w = 0.25

# ── Panel 1: Coverage Comparison ──
ax = axes[0]
raw_cov = [51.8, 64.2]
cal_cov = [89.3, 92.3]
target = [90, 95]

b1 = ax.bar(x - bar_w, raw_cov, bar_w, label="Raw Tube Loss", color="#4C72B0", edgecolor="black", linewidth=0.6)
b2 = ax.bar(x, cal_cov, bar_w, label="Calibrated", color="#55A868", edgecolor="black", linewidth=0.6)
b3 = ax.bar(x + bar_w, target, bar_w, label="Target", color="#C44E52", edgecolor="black", linewidth=0.6, alpha=0.5)

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel("Coverage (%)")
ax.set_title("Coverage: Raw vs Calibrated vs Target")
ax.set_ylim(0, 110)
ax.legend(loc="upper left", fontsize=9)
ax.axhline(90, color="#C44E52", ls="--", alpha=0.3, lw=1)
ax.axhline(95, color="#C44E52", ls="--", alpha=0.3, lw=1)
ax.grid(axis="y", alpha=0.2)

# ── Panel 2: Avg Width ──
ax = axes[1]
widths = [0.169, 0.211]
colors = ["#4C72B0", "#55A868"]
bars = ax.bar(x, widths, 0.45, color=colors, edgecolor="black", linewidth=0.6)
for bar, w in zip(bars, widths):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{w:.3f} m", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel("Interval Width (m)")
ax.set_title("Average Prediction Interval Width")
ax.set_ylim(0, 0.28)
ax.grid(axis="y", alpha=0.2)

# ── Panel 3: RMSE Comparison ──
ax = axes[2]
rmse_median = [0.052, 0.052]
rmse_raw = [0.104, 0.104]

b1 = ax.bar(x - 0.15, rmse_raw, 0.28, label="Raw CMIP6", color="#C44E52", edgecolor="black", linewidth=0.6)
b2 = ax.bar(x + 0.15, rmse_median, 0.28, label="Tube Loss Median", color="#55A868", edgecolor="black", linewidth=0.6)

for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f} m", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel("RMSE (m)")
ax.set_title("RMSE: Raw CMIP6 vs Tube Loss Median")
ax.set_ylim(0, 0.14)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.2)

pct = (0.104 - 0.052) / 0.104 * 100
ax.annotate(f"↓ {pct:.0f}% reduction", xy=(0.5, 0.07), fontsize=11,
            ha="center", color="#55A868", fontweight="bold")

plt.tight_layout()
plt.savefig("sla_plots/quantile_results_summary.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved → sla_plots/quantile_results_summary.png")
