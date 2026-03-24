import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for plots
os.makedirs("results/plots", exist_ok=True)

# YOUR DATA (Hardcoded here for immediate plotting, or load from file)
history_data = [
    {
        "video_name": "20241019 - 14h29",
        "model_version": "v1_baseline",
        "metrics": {
            "HOTA": 0.138, "DetA": 0.604, "AssA": 0.031, 
            "DetRe": 0.998, "DetPr": 0.604, "AssRe": 0.031, "AssPr": 0.896
        }
    },
    {
        "video_name": "20241019 - 13h28",
        "model_version": "v1_baseline",
        "metrics": {
            "HOTA": 0.204, "DetA": 0.705, "AssA": 0.059, 
            "DetRe": 0.998, "DetPr": 0.706, "AssRe": 0.059, "AssPr": 0.903
        }
    }
]

def plot_bar_comparison(data):
    """Generates a side-by-side bar chart for the two videos."""
    labels = [d['video_name'].split(' - ')[-1] for d in data] # Extract time only
    hota = [d['metrics']['HOTA'] for d in data]
    deta = [d['metrics']['DetA'] for d in data]
    assa = [d['metrics']['AssA'] for d in data]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width, hota, width, label='HOTA', color='#1f77b4')
    rects2 = ax.bar(x, deta, width, label='DetA (Detection)', color='#2ca02c')
    rects3 = ax.bar(x + width, assa, width, label='AssA (Association)', color='#d62728')

    ax.set_ylabel('Score (0-1)')
    ax.set_title('Comparison: 13h28 vs 14h29')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add text labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("results/plots/bar_comparison.png")
    print("Generated results/plots/bar_comparison.png")

def plot_radar_chart(data):
    """Generates a Spider/Radar chart to show the model 'shape'."""
    categories = ['DetRe (Recall)', 'DetPr (Precision)', 'AssPr (Precision)', 'AssRe (Recall)', 'HOTA']
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one line per video
    colors = ['#ff7f0e', '#1f77b4'] # Orange, Blue
    
    for i, entry in enumerate(data):
        m = entry['metrics']
        values = [m['DetRe'], m['DetPr'], m['AssPr'], m['AssRe'], m['HOTA']]
        values += values[:1] # Close loop
        
        video_short = entry['video_name'].split(' - ')[-1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=video_short, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    
    plt.title("Model Profile: Strong Detection, Weak Association", y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig("results/plots/radar_chart.png")
    print("Generated results/plots/radar_chart.png")

if __name__ == "__main__":
    plot_bar_comparison(history_data)
    plot_radar_chart(history_data)