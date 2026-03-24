import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ---------------------------------------------------------------------
# Optional: adjust these paths if you want logs/plots elsewhere
# ---------------------------------------------------------------------
HISTORY_FILE = os.path.join("results", "history.json")
PLOTS_DIR = os.path.join("results", "plots")
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# Logger (self-contained; remove if you already import from elsewhere)
# =============================================================================
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
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_name": video_name,
            "model_version": model_version,
            "metrics": metrics
        }
        self.history.append(entry)
        with open(HISTORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✅ Run logged for {video_name} ({model_version})")

    def get_dataframe(self):
        flat_data = []
        for entry in self.history:
            row = entry.copy()
            for k, v in entry['metrics'].items():
                row[k] = v
            del row['metrics']
            flat_data.append(row)
        return pd.DataFrame(flat_data)

# =============================================================================
# HOTA core
# =============================================================================
class HOTA:
    def __init__(self):
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
        self.float_array_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA']
        self.float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields

    def eval_sequence(self, data):
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=np.float64)
        for field in self.float_fields:
            res[field] = 0

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

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(gt_ids_t) == 0:
                for a, _ in enumerate(self.array_labels):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a, _ in enumerate(self.array_labels):
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

        for a, _ in enumerate(self.array_labels):
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

# =============================================================================
# Helpers
# =============================================================================
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def parse_chimp_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    frames = []
    current_frame_dets = []
    lines = [l.strip() for l in lines if l.strip()]
    if not lines:
        return []
    if lines[0] == '#':
        lines = lines[1:]
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
    alpha_range = np.arange(0.05, 0.99, 0.05)
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_range, results['HOTA'], label='HOTA', color='b', linewidth=3)
    plt.plot(alpha_range, results['DetA'], label='DetA', color='g', linestyle='--')
    plt.plot(alpha_range, results['AssA'], label='AssA', color='r', linestyle='--')
    plt.xlabel('IoU Threshold (Alpha)')
    plt.ylabel('Score (0-1)')
    plt.title('Performance vs. Overlap Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"🖼️  Plot saved to {save_path}")
    plt.close()

# =============================================================================
# Chunked evaluation
# =============================================================================
def _slice_frames(frames, start_idx, end_idx):
    return frames[start_idx:end_idx]

def run_evaluation_chunked(tracker_file, gt_file, video_name, model_version,
                           chunks=(0.0, 0.33, 0.66, 1.0), plot_single=False):
    pred_frames = parse_chimp_file(tracker_file)
    gt_frames = parse_chimp_file(gt_file)
    num_frames = min(len(pred_frames), len(gt_frames))
    if num_frames == 0:
        raise ValueError("No valid frames in pred/gt")
    pred_frames = pred_frames[:num_frames]
    gt_frames = gt_frames[:num_frames]

    chunk_indices = []
    for i in range(len(chunks) - 1):
        start = int(chunks[i] * num_frames)
        end = int(chunks[i + 1] * num_frames)
        if end > start:
            chunk_indices.append((start, end))

    chunk_metrics = []
    for (start, end) in chunk_indices:
        cf_pred = _slice_frames(pred_frames, start, end)
        cf_gt = _slice_frames(gt_frames, start, end)

        all_gt_ids = set(d['id'] for f in cf_gt for d in f)
        all_pred_ids = set(d['id'] for f in cf_pred for d in f)
        gt_id_map = {name: i for i, name in enumerate(sorted(list(all_gt_ids)))}
        pred_id_map = {name: i for i, name in enumerate(sorted(list(all_pred_ids)))}

        data = {
            'num_tracker_dets': 0,
            'num_gt_dets': 0,
            'num_tracker_ids': len(all_pred_ids),
            'num_gt_ids': len(all_gt_ids),
            'gt_ids': [], 'tracker_ids': [], 'similarity_scores': []
        }

        for t in tqdm(range(len(cf_pred)), desc=f"IoU chunk {start}-{end}", leave=False):
            gts = cf_gt[t]
            preds = cf_pred[t]
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

        hota_metric = HOTA()
        results = hota_metric.eval_sequence(data)
        final_metrics = {
            'HOTA': float(np.mean(results['HOTA'])),
            'DetA': float(np.mean(results['DetA'])),
            'AssA': float(np.mean(results['AssA'])),
            'DetRe': float(np.mean(results['DetRe'])),
            'DetPr': float(np.mean(results['DetPr'])),
            'AssRe': float(np.mean(results['AssRe'])),
            'AssPr': float(np.mean(results['AssPr'])),
        }
        chunk_metrics.append(final_metrics)

        if plot_single:
            plot_path = f"{PLOTS_DIR}/{video_name}_{model_version}_chunk_{start}_{end}.png"
            plot_single_run(results, plot_path)

    keys = chunk_metrics[0].keys()
    mean_metrics = {k: float(np.mean([cm[k] for cm in chunk_metrics])) for k in keys}
    std_metrics = {k + "_std": float(np.std([cm[k] for cm in chunk_metrics])) for k in keys}

    logger = ExperimentLogger()
    logger.log_run(video_name, model_version, {**mean_metrics, **std_metrics})
    return mean_metrics, std_metrics, chunk_metrics