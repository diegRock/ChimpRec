import os, itertools, tempfile, json, sys
from copy import deepcopy
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import torch

from ui_lib_strongSort import YOLO, build_strongsort, perform_tracking

# Speed-ups for GPU
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# Add Metric folder to import hota_eval
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /home/ucl/ingi/trixen/ChimpRec/Code
METRIC_DIR = PROJECT_ROOT / "Metric"
sys.path.append(str(METRIC_DIR))

from hota_eval import run_evaluation_chunked  # noqa: E402
from hota_eval import ExperimentLogger        # reuse same logger

# -----------------------
# USER CONFIG
# -----------------------
BASE_CFG_PATH = PROJECT_ROOT / "Tracking" / "strong_sort" / "config" / "strongsort_config.yaml"
BODY_MODEL_PATH = PROJECT_ROOT / "Tracking" / "strong_sort" / "Body_detection_models" / "Body_detection_model.pt"
REID_WEIGHTS_PATH = PROJECT_ROOT / "Tracking" / "strong_sort" / "Re-ID_models" / "osnet_ain_x1_0_imagenet.pth"
DEVICE = "cuda:0"   # GPU
USE_HALF_INFER = False  # safer default: keep FP32 for fusion; try True only if stable
DETECTION_CONF = 0.5    # must be >= tracker min_confidence

PRED_DIR = METRIC_DIR / "data" / "predictions"
GT_DIR = METRIC_DIR / "data" / "ground_truth"
INPUT_VIDEO_DIR = Path("/home/ucl/ingi/trixen/ChimpRec/ChimpVideo/input")

VALIDATION_VIDEOS = [
    "20241019 - 13h28",
    "20241019 - 14h29",
]

# Temporal folds (reduce eval cost if needed)
CHUNKS = (0.0, 0.5, 1.0)  # halves; change to thirds (0, 0.33, 0.66, 1) if you prefer

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(PROJECT_ROOT / "results" / "plots", exist_ok=True)

# -----------------------
# Search space (trimmed to 96 combos; expand if needed)
# -----------------------
SEARCH_SPACE = {
    "max_dist":         [0.12, 0.15],
    "max_iou_distance": [0.50, 0.60],
    "max_age":          [90, 140],
    "n_init":           [3, 5],
    "mc_lambda":        [0.995, 0.999],
    "ema_alpha":        [0.90, 0.94],
    "min_confidence":   [0.45, 0.50],  # must be <= DETECTION_CONF
    "nn_budget":        [200, 400],
}

# -----------------------
# Helpers
# -----------------------
def load_base_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_temp_cfg(base_cfg, overrides):
    cfg = deepcopy(base_cfg)
    for k, v in overrides.items():
        if k not in cfg:
            raise KeyError(f"Param {k} not in base cfg")
        cfg[k]["default"] = v
    fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="strongsort_tune_")
    os.close(fd)
    with open(temp_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return temp_path

def _find_video_path(video_name):
    for ext in (".mp4", ".MP4"):
        p = INPUT_VIDEO_DIR / f"{video_name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Video not found for {video_name} with .mp4 or .MP4")

def _validate_inputs(videos):
    missing = []
    valid = []
    for vid in videos:
        gt_path = GT_DIR / f"{vid}.txt"
        if not gt_path.exists():
            missing.append(str(gt_path))
        try:
            _find_video_path(vid)
        except FileNotFoundError:
            missing.append(f"{INPUT_VIDEO_DIR}/{vid}.(mp4|MP4)")
        if not missing or missing[-1].endswith(".txt") is False:  # crude but avoids double-adding
            valid.append(vid)
    if missing:
        print("Missing required files:\n" + "\n".join(missing))
    return valid

def run_tracking_for_video(video_name, tracker_config_path, run_id, yolo_model):
    input_video_path = _find_video_path(video_name)
    pred_out = PRED_DIR / f"{video_name}_{run_id}.txt"

    tracker = build_strongsort(
        reid_weights=str(REID_WEIGHTS_PATH),
        device=DEVICE,
        fp16=False,  # keep tracker in FP32 for stability
        tracker_config_path=tracker_config_path,
    )

    perform_tracking(
        input_video_path=str(input_video_path),
        output_text_file_path=str(pred_out),
        detection_model=yolo_model,
        tracker=tracker,
        confidence_threshold=DETECTION_CONF,
        device=DEVICE,
        use_half=USE_HALF_INFER,
    )
    return pred_out

def plot_run_progress(history, out_path):
    if not history:
        return
    xs = list(range(1, len(history) + 1))
    hota = [h["metrics"]["HOTA"] for h in history]
    assa = [h["metrics"]["AssA"] for h in history]
    deta = [h["metrics"]["DetA"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, hota, label="HOTA", linewidth=3)
    plt.plot(xs, assa, label="AssA", linestyle="--")
    plt.plot(xs, deta, label="DetA", linestyle="--")
    plt.xlabel("Run #")
    plt.ylabel("Score")
    plt.title("Hyperparam Sweep Progress (chunked)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.savefig(out_path)
    print(f"📈 Progress plot saved to {out_path}")
    plt.close()

# -----------------------
# Main sweep
# -----------------------
def main():
    valid_videos = _validate_inputs(VALIDATION_VIDEOS)
    if not valid_videos:
        raise FileNotFoundError("No valid videos found; fix GT/video paths.")
    print(f"Valid videos: {valid_videos}")

    base_cfg = load_base_cfg(BASE_CFG_PATH)
    logger = ExperimentLogger()

    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    combos = list(itertools.product(*values))
    print(f"Total runs: {len(combos)}")

    # Load YOLO once for speed, keep in FP32 to avoid fusion dtype issues
    yolo_model = YOLO(str(BODY_MODEL_PATH))
    yolo_model.to(DEVICE)

    run_history = []
    pbar = tqdm(range(len(combos)), desc="Hyperparam sweep")

    for idx, combo in enumerate(combos):
        overrides = {k: v for k, v in zip(keys, combo)}
        run_id = f"run{idx+1:03d}"
        model_version = f"strongsort_tune_{run_id}"

        # Ensure tracker gate not stricter than detector
        if overrides["min_confidence"] > DETECTION_CONF:
            pbar.update(1)
            continue

        temp_cfg_path = write_temp_cfg(base_cfg, overrides)

        per_video_metrics = []
        for vid in valid_videos:
            pred_path = run_tracking_for_video(vid, temp_cfg_path, run_id, yolo_model)
            gt_path = GT_DIR / f"{vid}.txt"
            mean_metrics, std_metrics, _ = run_evaluation_chunked(
                str(pred_path), str(gt_path), vid, model_version,
                chunks=CHUNKS, plot_single=False
            )
            per_video_metrics.append((mean_metrics, std_metrics))

        keys_m = per_video_metrics[0][0].keys()
        avg_mean = {k: float(np.mean([m[0][k] for m in per_video_metrics])) for k in keys_m}
        avg_std  = {k + "_std": float(np.mean([m[1][k + "_std"] for m in per_video_metrics])) for k in keys_m}

        logger.log_run(
            video_name=";".join(valid_videos),
            model_version=model_version,
            metrics={**avg_mean, **avg_std, "params": overrides},
        )

        run_history.append({"metrics": avg_mean, "params": overrides, "run_id": run_id})
        pbar.set_postfix({"HOTA": f"{avg_mean['HOTA']:.3f}", "AssA": f"{avg_mean['AssA']:.3f}"})
        pbar.update(1)

        os.remove(temp_cfg_path)

    pbar.close()
    plot_run_progress(run_history, out_path=PROJECT_ROOT / "results" / "plots" / "hparam_progress.png")

    if run_history:
        best = max(run_history, key=lambda x: x["metrics"]["HOTA"])
        print("\nBest run:", best["run_id"], "HOTA", best["metrics"]["HOTA"])
        print("Params:", json.dumps(best["params"], indent=2))

if __name__ == "__main__":
    main()