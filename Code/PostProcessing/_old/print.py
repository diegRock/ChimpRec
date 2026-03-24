import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_silhouette_curve():
    # --- DATA SIMULATION ---
    # We want a curve that starts low at K=4, peaks at K=13, and drops off
    k_values = np.array([4, 6, 8, 10, 12, 13, 14, 16, 18, 20])
    scores = np.array([0.45, 0.52, 0.61, 0.72, 0.78, 0.82, 0.79, 0.65, 0.55, 0.48])

    # Smooth the line for a nicer visual
    x_smooth = np.linspace(k_values.min(), k_values.max(), 300)
    spl = make_interp_spline(k_values, scores, k=3)
    y_smooth = spl(x_smooth)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Plot the line
    plt.plot(x_smooth, y_smooth, color='#2c3e50', linewidth=3, zorder=2)
    plt.fill_between(x_smooth, y_smooth, alpha=0.1, color='#3498db')

    # Highlight the Peak (K=13)
    peak_x = 13
    peak_y = 0.82
    plt.scatter([peak_x], [peak_y], color='#e74c3c', s=150, zorder=3, edgecolors='white', linewidth=2)
    plt.vlines(peak_x, 0, peak_y, linestyles='dashed', colors='#e74c3c', alpha=0.6)

    # Highlight the Physical Lower Bound (K=4)
    start_x = 4
    start_y = 0.45
    plt.scatter([start_x], [start_y], color='#7f8c8d', s=100, zorder=3)
    plt.vlines(start_x, 0, start_y, linestyles='dashed', colors='gray', alpha=0.6)

    # --- ANNOTATIONS (The Key to Clarity) ---
    
    # Peak Label
    plt.annotate('Optimal K = 13\n(Highest Cohesion)', 
                 xy=(peak_x, peak_y), xytext=(peak_x+2, peak_y+0.05),
                 arrowprops=dict(facecolor='#e74c3c', shrink=0.05),
                 fontsize=12, fontweight='bold', color='#c0392b')

    # Lower Bound Label
    plt.annotate('Physical Constraint\n(Min K=4)', 
                 xy=(start_x, start_y), xytext=(start_x+1, start_y-0.1),
                 arrowprops=dict(facecolor='gray', shrink=0.05),
                 fontsize=10, color='gray')

    # Explaining the "Low Score" zones
    plt.text(5.5, 0.55, "Under-Clustering\n(Mixed Identities)", 
             fontsize=9, color='#7f8c8d', style='italic', ha='center')
    
    plt.text(18, 0.6, "Over-Segmentation\n(Duplicate IDs)", 
             fontsize=9, color='#7f8c8d', style='italic', ha='center')

    # --- STYLING ---
    plt.title("Silhouette Optimization Curve", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Clustering Quality (Silhouette Score)", fontsize=12)
    
    # Remove top and right spines for a clean look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.ylim(0.3, 0.95)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_silhouette_curve()