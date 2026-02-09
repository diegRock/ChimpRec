import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt
from dashboard import ExperimentLogger  # Import the logger

# ==============================================================================
# 1. THE HOTA CLASS
# ==============================================================================
class HOTA:
    """Class which implements the HOTA metrics."""
    def __init__(self):
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
        self.float_array_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA']
        self.float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields

    def eval_sequence(self, data):
        """Calculates the HOTA metrics for one sequence"""
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=np.float64)
        for field in self.float_fields:
            res[field] = 0

        # Short circuit for empty files
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = data['num_gt_dets'] * np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA(0)'] = 1.0
            return self._compute_final_fields(res)
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = data['num_tracker_dets'] * np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA(0)'] = 1.0
            return self._compute_final_fields(res)

        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros((data['num_gt_ids'], 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']))

        # Global Association Loop
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            similarity = data['similarity_scores'][t]
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.array_labels]

        # Timestep Loop
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(gt_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FN'][a] += len(gt_ids_t)
                continue

            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            for a, alpha in enumerate(self.array_labels):
                actually_matched_mask = similarity[match_rows, match_cols] >= alpha - np.finfo('float').eps
                alpha_match_rows = match_rows[actually_matched_mask]
                alpha_match_cols = match_cols[actually_matched_mask]
                num_matches = len(alpha_match_rows)
                res['HOTA_TP'][a] += num_matches
                res['HOTA_FN'][a] += len(gt_ids_t) - num_matches
                res['HOTA_FP'][a] += len(tracker_ids_t) - num_matches
                if num_matches > 0:
                    res['LocA'][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                    matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

        # Calculate Scores
        for a, alpha in enumerate(self.array_labels):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
            res['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(1, res['HOTA_TP'][a])
            ass_re = matches_count / np.maximum(1, gt_id_count)
            res['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(1, res['HOTA_TP'][a])
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res['HOTA_TP'][a])

        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def _compute_final_fields(self, res):
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        res['HOTA(0)'] = res['HOTA'][0]
        return res

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0: return 0
    return intersection_area / union_area

def parse_chimp_file(filepath):
    """Parses custom format. Robust to empty lines and headers."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    frames = []
    current_frame_dets = []
    
    lines = [l.strip() for l in lines if l.strip()]
    if not lines: return []
    if lines[0] == '#': lines = lines[1:]

    for line in lines:
        if line == '#':
            frames.append(current_frame_dets)
            current_frame_dets = []
        else:
            parts = line.split()
            if len(parts) >= 5:
                obj_id = parts[0]
                coords = [float(x) for x in parts[1:5]]
                current_frame_dets.append({'id': obj_id, 'box': coords})
    
    if current_frame_dets:
        frames.append(current_frame_dets)

    return frames

def plot_single_run(results, save_path):
    """Visualizes HOTA metrics across Alpha thresholds for a single run."""
    alpha_range = np.arange(0.05, 0.99, 0.05)
    
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_range, results['HOTA'], label='HOTA (Combined)', color='b', linewidth=3)
    plt.plot(alpha_range, results['DetA'], label='DetA (Detection)', color='g', linestyle='--')
    plt.plot(alpha_range, results['AssA'], label='AssA (Association)', color='r', linestyle='--')
    
    plt.xlabel('IoU Threshold (Alpha)')
    plt.ylabel('Score (0-1)')
    plt.title('Performance vs. Overlap Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"🖼️  Plot saved to {save_path}")
    plt.close()

# ==============================================================================
# 3. MAIN EVALUATION LOGIC
# ==============================================================================

def run_evaluation(tracker_file, gt_file, video_name, model_version, plot=True):
    print(f"📊 Processing {video_name}...")
    
    # 1. Parse
    pred_frames = parse_chimp_file(tracker_file)
    gt_frames = parse_chimp_file(gt_file)

    num_frames = min(len(pred_frames), len(gt_frames))
    if num_frames == 0:
        print("❌ Error: No valid frames found in one or both files.")
        return

    pred_frames = pred_frames[:num_frames]
    gt_frames = gt_frames[:num_frames]

    # 2. Map IDs
    all_gt_ids = set(d['id'] for f in gt_frames for d in f)
    all_pred_ids = set(d['id'] for f in pred_frames for d in f)
    gt_id_map = {name: i for i, name in enumerate(sorted(list(all_gt_ids)))}
    pred_id_map = {name: i for i, name in enumerate(sorted(list(all_pred_ids)))}

    data = {
        'num_tracker_dets': 0,
        'num_gt_dets': 0,
        'num_tracker_ids': len(all_pred_ids),
        'num_gt_ids': len(all_gt_ids),
        'gt_ids': [], 'tracker_ids': [], 'similarity_scores': []
    }

    # 3. Calculate IoU
    for t in tqdm(range(num_frames), desc="Calculating IoU"):
        gts = gt_frames[t]
        preds = pred_frames[t]

        data['num_gt_dets'] += len(gts)
        data['num_tracker_dets'] += len(preds)

        gt_ids_arr = np.array([gt_id_map[d['id']] for d in gts], dtype=np.int32)
        pred_ids_arr = np.array([pred_id_map[d['id']] for d in preds], dtype=np.int32)
        
        data['gt_ids'].append(gt_ids_arr)
        data['tracker_ids'].append(pred_ids_arr)

        iou_matrix = np.zeros((len(gts), len(preds)))
        for i, gt in enumerate(gts):
            for j, pred in enumerate(preds):
                iou_matrix[i, j] = calculate_iou(gt['box'], pred['box'])
        
        data['similarity_scores'].append(iou_matrix)

    # 4. Run HOTA
    hota_metric = HOTA()
    results = hota_metric.eval_sequence(data)

    # 5. Extract Mean Scores
    final_metrics = {
        'HOTA': float(np.mean(results['HOTA'])),
        'DetA': float(np.mean(results['DetA'])),
        'AssA': float(np.mean(results['AssA'])),
        'DetRe': float(np.mean(results['DetRe'])),
        'DetPr': float(np.mean(results['DetPr'])),
        'AssRe': float(np.mean(results['AssRe'])),
        'AssPr': float(np.mean(results['AssPr'])),
    }

    print("\n--- Evaluation Results ---")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.3f}")

    # 6. Log to History
    logger = ExperimentLogger()
    logger.log_run(video_name, model_version, final_metrics)

    # 7. Plot Single Run
    if plot:
        plot_path = f"results/plots/{video_name}_{model_version}.png"
        plot_single_run(results, plot_path)

# ==============================================================================
# RUN
# ==============================================================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    VIDEO_NAME = "13h28-strongsort"  # Change this to your video name (without extension)
    MODEL_VERSION = "strongsort_v1"  # Change this when you improve the model!
    
    # Paths based on your new directory structure
    TRACKER_PATH = f"data/predictions/{VIDEO_NAME}.txt"
    GT_PATH = f"data/ground_truth/{VIDEO_NAME}.txt"
    
    run_evaluation(TRACKER_PATH, GT_PATH, VIDEO_NAME, MODEL_VERSION, plot=True)