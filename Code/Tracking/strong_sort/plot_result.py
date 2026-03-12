import json
import pandas as pd
import matplotlib.pyplot as plt

# Replace with your actual path to history.json or the pasted JSON data file
HISTORY_PATH = "/home/ucl/ingi/trixen/ChimpRec/Code/Tracking/strong_sort/results/history.json"

# If you saved the JSON above to a file, set HISTORY_PATH to that file instead.

with open(HISTORY_PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame([{
    "run": d["model_version"],
    "HOTA": d["metrics"]["HOTA"],
    "DetA": d["metrics"]["DetA"],
    "AssA": d["metrics"]["AssA"],
    "HOTA_std": d["metrics"]["HOTA_std"],
    "DetA_std": d["metrics"]["DetA_std"],
    "AssA_std": d["metrics"]["AssA_std"],
    "nn_budget": d["metrics"]["params"]["nn_budget"],
    "min_conf": d["metrics"]["params"]["min_confidence"],
    "ema_alpha": d["metrics"]["params"]["ema_alpha"],
} for d in data])

print(df[["run","HOTA","DetA","AssA","nn_budget","min_conf","ema_alpha"]])

# Plot HOTA per run with error bars
plt.figure(figsize=(8,4))
plt.errorbar(df["run"], df["HOTA"], yerr=df["HOTA_std"], fmt="o", capsize=4, label="HOTA")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Score")
plt.title("HOTA by run")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("/home/ucl/ingi/trixen/ChimpRec/Code/Tracking/strong_sort/results/plots/hota_by_run.png")
plt.close()

# Scatter: HOTA vs ema_alpha colored by nn_budget
plt.figure(figsize=(6,5))
scatter = plt.scatter(df["ema_alpha"], df["HOTA"], c=df["nn_budget"], cmap="viridis", s=80)
plt.xlabel("ema_alpha")
plt.ylabel("HOTA")
plt.title("HOTA vs ema_alpha (color=nn_budget)")
plt.colorbar(scatter, label="nn_budget")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/ucl/ingi/trixen/ChimpRec/Code/results/plots/hota_vs_ema_alpha.png")
plt.close()