#!/usr/bin/env python3
"""Filter already-extracted face crops by frontal score and eye visibility.

Reads face crops from --input-root, scores each for frontal orientation,
and moves high-quality crops to --output-root. Generates CSVs with scores.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


@dataclass
class QualityMetrics:
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float


def frontal_score(face_bgr) -> float:
    """Estimate how likely a face crop is frontal (looking at camera).

    Combines left/right symmetry with a chimp-specific cue: a frontal
    chimp face shows a BRIGHTER central skin patch (the pinkish/tan
    muzzle and brow region) surrounded by DARKER fur.
    """
    h, w = face_bgr.shape[:2]
    if h < 20 or w < 20:
        return 0.0

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    half = gray.shape[1] // 2
    left = gray[:, :half].astype(np.float32)
    right = cv2.flip(gray[:, half:], 1).astype(np.float32)

    # 1) Left/right symmetry (normalised cross-correlation)
    left_c = left - float(left.mean())
    right_c = right - float(right.mean())
    denom = (np.linalg.norm(left_c) * np.linalg.norm(right_c)) + 1e-6
    symmetry = float(np.clip(np.sum(left_c * right_c) / denom, -1.0, 1.0))
    symmetry_01 = 0.5 * (symmetry + 1.0)

    # 2) Sobel-x gradient balance
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_left = float(np.mean(np.abs(gx[:, :half])))
    grad_right = float(np.mean(np.abs(gx[:, half:])))
    if max(grad_left, grad_right) < 1e-6:
        grad_balance = 0.0
    else:
        grad_balance = 1.0 - min(1.0, abs(grad_left - grad_right) / max(grad_left, grad_right))

    # 3) Horizontal centre of dark mass
    inv = (255 - gray).astype(np.float32)
    mom = cv2.moments(inv)
    if mom["m00"] > 1e-6:
        cx = float(mom["m10"] / mom["m00"]) / gray.shape[1]
    else:
        cx = 0.5
    center_score = 1.0 - min(1.0, abs(cx - 0.5) / 0.5)

    # 4) Brightness balance L vs R
    mean_left = float(np.mean(gray[:, :half]))
    mean_right = float(np.mean(gray[:, half:]))
    bright_balance = 1.0 - min(1.0, abs(mean_left - mean_right) / 255.0)

    # 5) Bright-skin patch (chimp-specific)
    H, W = gray.shape
    yy, xx = np.ogrid[:H, :W]
    cx_f, cy_f = W / 2.0, H / 2.0
    rx_inner, ry_inner = W * 0.30, H * 0.32
    rx_outer, ry_outer = W * 0.48, H * 0.48
    inner = ((xx - cx_f) / rx_inner) ** 2 + ((yy - cy_f) / ry_inner) ** 2 <= 1.0
    outer = ((xx - cx_f) / rx_outer) ** 2 + ((yy - cy_f) / ry_outer) ** 2 <= 1.0
    ring = outer & ~inner

    if inner.any() and ring.any():
        inner_mean = float(gray[inner].mean())
        ring_mean = float(gray[ring].mean())
        skin_patch_score = float(np.clip((inner_mean - ring_mean) / 40.0, 0.0, 1.0))
    else:
        skin_patch_score = 0.0

    score = (
        0.40 * symmetry_01
        + 0.25 * skin_patch_score
        + 0.15 * grad_balance
        + 0.12 * center_score
        + 0.08 * bright_balance
    )
    return float(np.clip(score, 0.0, 1.0))


def detect_eye_count(face_bgr, min_rel_size=0.04):
    """Estimate visible eyes in a chimp face crop (same as extraction)."""
    h, w = face_bgr.shape[:2]
    if h < 32 or w < 32:
        return 0

    y0 = int(h * 0.12)
    y1 = int(h * 0.62)
    x_margin = int(w * 0.10)
    x0 = x_margin
    x1 = w - x_margin
    if y1 - y0 < 12 or x1 - x0 < 12:
        return 0
    roi = face_bgr[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh_val = float(np.percentile(gray, 6))
    thresh_val = float(np.clip(thresh_val, 5.0, 120.0))
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return 0

    min_side = max(3, int(min(rh, rw) * min_rel_size))
    min_area = min_side * min_side
    max_area = int(0.06 * rh * rw)

    candidates = []
    for i in range(1, num):
        bx, by, bw_, bh_, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if bw_ < min_side or bh_ < min_side:
            continue
        aspect = bw_ / max(bh_, 1)
        if aspect < 0.4 or aspect > 3.0:
            continue
        cx, cy = centroids[i]
        if cy > 0.70 * rh:
            continue
        compactness = area / float(bw_ * bh_)
        if compactness < 0.40:
            continue
        candidates.append((float(cx), float(cy), int(area)))

    if not candidates:
        return 0

    mid_x = rw / 2.0
    left = [c for c in candidates if c[0] < mid_x]
    right = [c for c in candidates if c[0] >= mid_x]

    if not left and not right:
        return 0
    if not left or not right:
        return 1

    left.sort(key=lambda c: c[2], reverse=True)
    right.sort(key=lambda c: c[2], reverse=True)
    lcx, lcy, _ = left[0]
    rcx, rcy, _ = right[0]

    dy = abs(lcy - rcy)
    dx = abs(rcx - lcx)

    if dy > 0.18 * rh:
        return 1
    if dx < 0.25 * rw:
        return 1
    if dx > 0.90 * rw:
        return 1
    return 2


def assess_quality(
    face_bgr,
    min_w,
    min_h,
    min_sharpness,
    min_contrast,
    min_brightness,
    max_brightness,
):
    h, w = face_bgr.shape[:2]
    if w < min_w or h < min_h:
        return False, "too_small", QualityMetrics(w, h, 0.0, 0.0, 0.0)

    aspect = w / max(1.0, float(h))
    if aspect < 0.55 or aspect > 1.9:
        return False, "bad_aspect", QualityMetrics(w, h, 0.0, 0.0, 0.0)

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    metrics = QualityMetrics(
        width=w,
        height=h,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
    )

    if sharpness < min_sharpness:
        return False, "blurry", metrics
    if contrast < min_contrast:
        return False, "low_contrast", metrics
    if brightness < min_brightness:
        return False, "too_dark", metrics
    if brightness > max_brightness:
        return False, "too_bright", metrics

    return True, "ok", metrics


def filter_crops(input_root: Path, output_root: Path, min_frontal_score: float, min_eye_count: int, args):
    """Scan all chimp subdirectories, re-score crops, and move high-quality ones."""
    output_root.mkdir(parents=True, exist_ok=True)

    kept_rows = []
    rejected_rows = []
    kept_by_chimp = defaultdict(int)
    rejected_by_chimp = defaultdict(int)

    # Find all chimp subdirs
    chimp_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    if not chimp_dirs:
        print(f"No chimp subdirectories found in {input_root}")
        return

    all_crops = []
    for chimp_dir in chimp_dirs:
        chimp_name = chimp_dir.name
        crops = sorted(chimp_dir.glob("*.jpg"))
        for crop_path in crops:
            all_crops.append((chimp_name, crop_path))

    progress = None
    if args.show_progress and tqdm is not None:
        progress = tqdm(total=len(all_crops), desc="Filtering crops", unit="crop")

    for chimp_name, crop_path in all_crops:
        img = cv2.imread(str(crop_path))
        if img is None or img.size == 0:
            rejected_rows.append({
                "chimp_name": chimp_name,
                "file_name": crop_path.name,
                "reason": "could_not_read_image",
            })
            rejected_by_chimp[chimp_name] += 1
            if progress:
                progress.update(1)
            continue

        # Quality check
        good, reason, metrics = assess_quality(
            img,
            min_w=args.min_face_width,
            min_h=args.min_face_height,
            min_sharpness=args.min_sharpness,
            min_contrast=args.min_contrast,
            min_brightness=args.min_brightness,
            max_brightness=args.max_brightness,
        )
        if not good:
            rejected_rows.append({
                "chimp_name": chimp_name,
                "file_name": crop_path.name,
                "reason": reason,
                "sharpness": f"{metrics.sharpness:.3f}",
                "brightness": f"{metrics.brightness:.3f}",
                "contrast": f"{metrics.contrast:.3f}",
            })
            rejected_by_chimp[chimp_name] += 1
            if progress:
                progress.update(1)
            continue

        # Frontal score
        front_score = frontal_score(img)
        if front_score < min_frontal_score:
            rejected_rows.append({
                "chimp_name": chimp_name,
                "file_name": crop_path.name,
                "reason": "low_frontal_score",
                "frontal_score": f"{front_score:.3f}",
            })
            rejected_by_chimp[chimp_name] += 1
            if progress:
                progress.update(1)
            continue

        # Eye count
        eye_count = detect_eye_count(img, min_rel_size=0.05)
        if eye_count < min_eye_count:
            rejected_rows.append({
                "chimp_name": chimp_name,
                "file_name": crop_path.name,
                "reason": "low_eye_count",
                "eye_count": eye_count,
                "frontal_score": f"{front_score:.3f}",
            })
            rejected_by_chimp[chimp_name] += 1
            if progress:
                progress.update(1)
            continue

        # Passed all checks — copy to output
        out_chimp_dir = output_root / chimp_name
        out_chimp_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_chimp_dir / crop_path.name

        shutil.copy2(str(crop_path), str(out_path))
        kept_rows.append({
            "chimp_name": chimp_name,
            "file_name": crop_path.name,
            "frontal_score": f"{front_score:.3f}",
            "eye_count": eye_count,
            "sharpness": f"{metrics.sharpness:.3f}",
            "brightness": f"{metrics.brightness:.3f}",
            "contrast": f"{metrics.contrast:.3f}",
            "face_w": metrics.width,
            "face_h": metrics.height,
        })
        kept_by_chimp[chimp_name] += 1

        if progress:
            progress.update(1)

    if progress:
        progress.close()

    # Write CSVs
    kept_csv = output_root / "kept_frontal_crops.csv"
    rejected_csv = output_root / "rejected_crops.csv"

    if kept_rows:
        fieldnames = sorted(set(key for row in kept_rows for key in row.keys()))
        with kept_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
    else:
        kept_csv.write_text("")

    if rejected_rows:
        fieldnames = sorted(set(key for row in rejected_rows for key in row.keys()))
        with rejected_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rejected_rows)
    else:
        rejected_csv.write_text("")

    print("\n===== Filtering Summary =====")
    print(f"Kept:     {len(kept_rows)}")
    print(f"Rejected: {len(rejected_rows)}")
    print(f"\nPer chimp kept:")
    for chimp_name in sorted(set(kept_by_chimp.keys()) | set(rejected_by_chimp.keys())):
        kept = kept_by_chimp[chimp_name]
        rejected = rejected_by_chimp[chimp_name]
        total = kept + rejected
        pct = 100.0 * kept / total if total > 0 else 0.0
        print(f"  {chimp_name:12s}  kept: {kept:4d}  rejected: {rejected:4d}  ({pct:.1f}%)")

    print(f"\nKept crops CSV:     {kept_csv}")
    print(f"Rejected crops CSV: {rejected_csv}")
    print(f"Output directory:   {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter face crops by frontal score and eye visibility"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Input directory with chimp subdirs (from extraction script)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output directory for filtered crops (will create chimp subdirs)",
    )
    parser.add_argument(
        "--min-frontal-score",
        type=float,
        default=0.55,
        help="Frontal score threshold [0, 1]",
    )
    parser.add_argument(
        "--min-eye-count",
        type=int,
        default=1,
        help="Minimum visible eyes required",
    )
    parser.add_argument(
        "--min-face-width",
        type=int,
        default=90,
        help="Min face crop width",
    )
    parser.add_argument(
        "--min-face-height",
        type=int,
        default=90,
        help="Min face crop height",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=120.0,
        help="Min Laplacian variance",
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=28.0,
        help="Min grayscale std",
    )
    parser.add_argument(
        "--min-brightness",
        type=float,
        default=35.0,
        help="Min grayscale mean",
    )
    parser.add_argument(
        "--max-brightness",
        type=float,
        default=220.0,
        help="Max grayscale mean",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bar",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not input_root.exists():
        print(f"Input directory not found: {input_root}")
        return

    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Min frontal score: {args.min_frontal_score}")
    print(f"Min eye count:     {args.min_eye_count}")

    filter_crops(input_root, output_root, args.min_frontal_score, args.min_eye_count, args)


if __name__ == "__main__":
    main()
