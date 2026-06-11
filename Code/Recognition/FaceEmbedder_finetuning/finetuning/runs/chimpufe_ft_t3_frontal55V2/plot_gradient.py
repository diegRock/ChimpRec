import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Consistent thesis-friendly style

plt.rcParams.update({
    "font.family": "STIXGeneral",      # main text
    "mathtext.fontset": "stix",        # math symbols (e.g., $t$)
    "font.size": 12,                  # base size
})

# ------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------
df = pd.read_csv("train_history.csv")

# ------------------------------------------------------------
# Continuous epoch index
# ------------------------------------------------------------
df = df.copy()
df["global_epoch"] = np.arange(1, len(df) + 1)

# ------------------------------------------------------------
# Fix infinities (CAUSE OF YOUR GAPS)
# ------------------------------------------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Optional: interpolate missing values for smooth curves
df["grad_norm_mean"] = df["grad_norm_mean"].interpolate()
df["grad_norm_max"]  = df["grad_norm_max"].interpolate()

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    df["global_epoch"],
    df["grad_norm_mean"],
    label="mean",
    linewidth=2,
)

ax.plot(
    df["global_epoch"],
    df["grad_norm_max"],
    label="max",
    linewidth=2,
)

# ------------------------------------------------------------
# ✅ KEEP: vertical line between stages
# ------------------------------------------------------------
stage1_len = (df["stage"] == "stage1_linear_probe").sum()

ax.axvline(
    stage1_len,
    color="black",
    linestyle="--",
    alpha=0.5,
    label="Stage transition",
)

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------
ax.set_xlabel("Epoch")
ax.set_ylabel("Gradient norm")
ax.set_title("Gradient norm per epoch", fontsize=14)

ax.legend()
fig.tight_layout()

# Save instead of show (no GUI)
plt.savefig(
    "gradient_norm_continuous_fixed.png",
    dpi=300,
    bbox_inches="tight",
)

print("Saved: gradient_norm_continuous_fixed.png")