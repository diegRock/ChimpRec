#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# ------------------------------------------------------------
# Style (same as your main script)
# ------------------------------------------------------------

plt.rcParams.update({
    "font.family": "STIXGeneral",      # main text
    "mathtext.fontset": "stix",        # math symbols (e.g., $t$)
    "font.size": 12,                  # base size
})

# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def plot_per_class_heatmap_with_stage(csv_path: Path, out_path: Path):
    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Extract per-class accuracy columns
    # --------------------------------------------------------
    cls_cols = [c for c in df.columns if c.startswith("val_acc__")]
    if not cls_cols:
        print("No per-class columns found.")
        return

    heatmap_df = df[cls_cols].copy()
    heatmap_df.columns = [c.replace("val_acc__", "") for c in cls_cols]

    # rows = chimps, columns = epochs
    heatmap_df = heatmap_df.T

    # --------------------------------------------------------
    # Create plot
    # --------------------------------------------------------
    fig_width = max(8, heatmap_df.shape[1] * 0.2)
    fig_height = max(5, heatmap_df.shape[0] * 0.3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        heatmap_df,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Validation Accuracy"},
    )

    # --------------------------------------------------------
    # ✅ Compute stage boundary
    # --------------------------------------------------------
    stage1_len = (df["stage"] == "stage1_linear_probe").sum()+0.5

    # Add vertical dashed line
    ax.axvline(
        stage1_len,
        color="white",
        linestyle="--",
        linewidth=2,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Chimpanzee Names")
    ax.set_title("Per-class validation accuracy over epochs", fontsize=14)
    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))

    plt.close(fig)
    print(f"Saved: {out_path}.png / .pdf")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-name", type=str, default="per_class_heatmap_with_stage")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = run_dir / "train_history.csv"
    out_path = run_dir / args.out_name

    plot_per_class_heatmap_with_stage(csv_path, out_path)