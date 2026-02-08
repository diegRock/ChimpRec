# Chimpanzee Tracking Evaluation Documentation

**Project:** Chimp Tracker AI Evaluation  
**Methods:** YOLO (Detection), DeepSORT (Association)  
**Metric:** HOTA (Higher Order Tracking Accuracy)

---

## 1. Introduction

### The Pipeline
Our AI pipeline is designed to automate the monitoring of chimpanzees in video footage. It utilizes a two-stage approach:
1.  **Detection (YOLO):** Identifies the location of every chimpanzee in a video frame and draws a bounding box around them.
2.  **Association (DeepSORT):** Assigns a unique ID to each detected bounding box and tracks that ID across time.

### The Problem
Visual confirmation isn't enough to scale our model. We need a standardized, mathematical way to measure performance. Specifically, we need to answer two distinct questions simultaneously:
*   *Is the model finding all the chimps?* (Detection)
*   *Is the model keeping the IDs consistent?* (Association)

---

## 2. Why HOTA?

We have chosen **HOTA (Higher Order Tracking Accuracy)** as our primary evaluation metric. While other metrics exist (such as MOTA or IDF1), they often have biases that make them less suitable for our specific needs.

| Metric | Primary Focus | Potential Downside for Us |
| :--- | :--- | :--- |
| **MOTA** | Detection | Heavily biased towards finding objects. It doesn't penalize ID switches enough. A model that finds chimps but swaps their names constantly would still get a good MOTA score. |
| **IDF1** | Association | Heavily biased towards identity consistency. It can sometimes penalize a model for finding *new* chimps if the ID confidence is low. |
| **HOTA** | **Balanced** | **The Geometric Mean of Detection and Association.** |

**Why it works for Chimps:**
HOTA explicitly balances the score. To get a high HOTA score, our pipeline must be good at finding the chimps **AND** good at tracking them. If YOLO fails (misses a chimp) or DeepSORT fails (confuses Ivan for Djiku), the HOTA score drops proportionally.

---

## 3. Data Transformation & Implementation

To calculate HOTA, we compare our **Tracker Output** (raw model predictions) against our **Ground Truth** (manually corrected annotations).

### 3.1 Custom File Format
Both the prediction files and ground truth files share a custom text-based structure. 
*   **Delimiter:** The `#` symbol indicates the end of a frame.
*   **Data Line:** `ID x1 y1 x2 y2` (Space-separated).

**Example:**
```text
#
5251 372.7 166.7 855.4 627.9   <-- Frame 1, Chimp A
5258 801.7 689.6 1285.3 1078.6 <-- Frame 1, Chimp B
#
5251 356.9 174.2 836.4 631.0   <-- Frame 2, Chimp A (tracked)
#
```

### 3.2 The Evaluation Pipeline
We implemented a custom Python script (`eval_chimps.py`) to bridge the gap between our text files and the mathematical HOTA implementation.

1.  **Parsing:** The script reads the files line-by-line. It accumulates detections into a list until it encounters a `#`, at which point it packages those detections as a single "Frame" and resets the accumulator. This allows it to handle frames with 0, 1, or multiple chimps seamlessly.
2.  **ID Mapping:** The HOTA algorithm requires integer IDs (0, 1, 2...). Our Ground Truth uses names ("Ivan") and our Tracker uses strings ("5251"). The script dynamically maps these to unique integers at runtime (e.g., `Ivan` -> `0`, `Djiku` -> `1`).
3.  **IoU Calculation:** For every frame, the script calculates an **Intersection over Union (IoU)** Similarity Matrix. This measures the spatial overlap between every Ground Truth box and every Predicted box.
4.  **Score Generation:** The script runs the Hungarian Algorithm (optimization) to determine the best matches between Ground Truth and Predictions based on the IoU matrix, then calculates the HOTA sub-scores.

---

## 4. Interpreting the Results

When you run the evaluation, you will receive a console output. Use this "Cheat Sheet" to interpret the health of your model.

### Summary Score
*   **HOTA:** **The headline number.** Ranges from 0 to 1 (0% to 100%).
    *   *Goal:* > 0.5 is generally considered a decent starting point for custom tracking tasks. > 0.7 is excellent.

### Diagnostic Scores (Debugging)
If your HOTA score is low, look at these two to find the root cause:

*   **DetA (Detection Accuracy):** **"Is YOLO working?"**
    *   *High:* You are finding the chimps correctly.
    *   *Low:* Check your bounding boxes. Are you missing chimps (False Negatives)? Are you detecting rocks/bushes as chimps (False Positives)?
    
*   **AssA (Association Accuracy):** **"Is DeepSORT working?"**
    *   *High:* IDs are stable. "Ivan" stays "Ivan".
    *   *Low:* ID Switching is occurring. The model loses track of a chimp and re-assigns a new ID, or swaps IDs between two chimps passing each other.

### Fine-Grained Metrics
*   **DetRe (Detection Recall):** The % of ground-truth chimps successfully found.
*   **DetPr (Detection Precision):** The % of predicted boxes that were actually chimps.
*   **AssRe / AssPr:** Similar concepts applied to the tracking IDs.

---

## 5. Visualization (Plotting)

HOTA is calculated by averaging performance across 19 different overlap thresholds (IoU 0.05 to 0.95). Plotting this curve helps us understand how "tight" our bounding boxes are.

Add this function to your evaluation script to generate a performance graph:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_hota_curve(results, save_path="hota_plot.png"):
    """
    Plots the HOTA, DetA, and AssA scores across different alpha (IoU) thresholds.
    """
    alpha_range = np.arange(0.05, 0.99, 0.05)
    
    plt.figure(figsize=(10, 6))
    
    # Plot main metrics
    plt.plot(alpha_range, results['HOTA'], label='HOTA', color='b', linewidth=3)
    plt.plot(alpha_range, results['DetA'], label='DetA (Detection)', color='g', linestyle='--')
    plt.plot(alpha_range, results['AssA'], label='AssA (Association)', color='r', linestyle='--')
    
    plt.xlabel('Localization Threshold (IoU Alpha)')
    plt.ylabel('Score')
    plt.title('Tracking Performance vs. IoU Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()
```

**How to read the plot:**
*   A curve that drops off sharply as Alpha increases means your boxes are loose (they cover the chimp but aren't perfectly tight).
*   A flat line means your localization is highly accurate.

---

## 6. Future Ameliorations

To further improve the pipeline and evaluation, we can implement the following:

1.  **Tune IoU Thresholds:** Currently, standard metrics often use 0.5 as a cutoff. If chimps are often occluded by foliage, we might want to analyze performance specifically at lower IoU thresholds to see if we are detecting them at all, even if the box isn't perfect.
2.  **Handling "Empty" Frames:** Currently, frames with no chimps are processed. If the model predicts nothing in an empty frame, that is good behavior (True Negative), but tracking metrics primarily measure Positive detection. We should verify if long sequences of empty frames are skewing average scores.
3.  **Per-Chimp Analysis:** We can extend the script to calculate HOTA for specific Class IDs.
    *   *Example:* Calculate metrics *only* for "Ivan". This would tell us if the model has specific trouble tracking distinct individuals (e.g., perhaps dark fur against dark background makes one chimp harder to track than another).
