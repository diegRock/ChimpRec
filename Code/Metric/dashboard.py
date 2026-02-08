import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

# Configuration
HISTORY_FILE = "results/history.json"
PLOTS_DIR = "results/plots"

os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

class ExperimentLogger:
    def __init__(self):
        self.history = self._load_history()

    def _load_history(self):
        if not os.path.exists(HISTORY_FILE):
            return []
        with open(HISTORY_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def log_run(self, video_name, model_version, metrics):
        """
        Saves a run to the history file.
        metrics: dict containing 'HOTA', 'DetA', 'AssA', etc.
        """
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_name": video_name,
            "model_version": model_version,  # e.g., "v1_baseline", "v2_tuned_deepsort"
            "metrics": metrics
        }
        self.history.append(entry)
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✅ Run logged for {video_name} ({model_version})")

    def get_dataframe(self):
        """Returns history as a Pandas DataFrame for easier filtering"""
        flat_data = []
        for entry in self.history:
            row = entry.copy()
            # Flatten metrics dict into columns
            for k, v in entry['metrics'].items():
                row[k] = v
            del row['metrics']
            flat_data.append(row)
        return pd.DataFrame(flat_data)

class Plotter:
    def __init__(self):
        self.logger = ExperimentLogger()
        self.df = self.logger.get_dataframe()

    def plot_progress_over_time(self, video_name):
        """
        Plot 1: Line chart showing HOTA score improvement for a SPECIFIC video.
        """
        if self.df.empty: return
        
        # Filter for specific video
        video_df = self.df[self.df['video_name'] == video_name]
        if video_df.empty:
            print(f"No history found for {video_name}")
            return

        # Sort by time
        video_df = video_df.sort_values('timestamp')

        plt.figure(figsize=(10, 6))
        
        # Plot HOTA, DetA, AssA lines
        plt.plot(video_df['model_version'], video_df['HOTA'], marker='o', linewidth=3, label='HOTA')
        plt.plot(video_df['model_version'], video_df['DetA'], marker='s', linestyle='--', label='DetA')
        plt.plot(video_df['model_version'], video_df['AssA'], marker='^', linestyle='--', label='AssA')

        plt.title(f"Model Improvement Over Time: {video_name}")
        plt.xlabel("Model Version")
        plt.ylabel("Score")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        out_path = f"{PLOTS_DIR}/progress_{video_name}.png"
        plt.savefig(out_path)
        print(f"📈 Progress plot saved to {out_path}")
        plt.close()

    def plot_radar_comparison(self, model_version):
        """
        Plot 2: Radar/Spider Chart comparing multiple ground truths (videos) 
        for a specific model version.
        """
        if self.df.empty: return

        # Get data for this model version
        subset = self.df[self.df['model_version'] == model_version]
        if subset.empty: return

        # Metrics to display on radar
        categories = ['HOTA', 'DetA', 'AssA', 'DetRe', 'AssPr']
        N = len(categories)

        # Setup Radar angles
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1] # Close the loop

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for _, row in subset.iterrows():
            values = [row[cat] for cat in categories]
            values += values[:1] # Close loop
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=row['video_name'])
            ax.fill(angles, values, alpha=0.1)

        plt.xticks(angles[:-1], categories)
        plt.title(f"Performance across Videos ({model_version})")
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        out_path = f"{PLOTS_DIR}/radar_compare_{model_version}.png"
        plt.savefig(out_path)
        print(f"🕸️ Radar comparison saved to {out_path}")
        plt.close()

# Example Usage Block
if __name__ == "__main__":
    # 1. Initialize
    logger = ExperimentLogger()
    plotter = Plotter()

    # --- How to log a NEW run manually (usually done inside HOTA.py) ---
    # logger.log_run(
    #     video_name="20241019", 
    #     model_version="v1_baseline", 
    #     metrics={'HOTA': 0.205, 'DetA': 0.706, 'AssA': 0.059, 'DetRe': 0.99, 'AssPr': 0.90}
    # )

    # --- Generate Plots ---
    # Plot history for one video
    plotter.plot_progress_over_time("20241019 - 14h29")
    
    # Compare multiple videos for the latest model
    # plotter.plot_radar_comparison("v1_baseline")