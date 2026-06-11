#!/usr/bin/env python3
"""Extract high-quality chimp face crops from named tracking files and videos.

Input tracking format:
- One frame block starts with a line containing only '#'.
- Following lines in that block look like:
  <name> <x1> <y1> <x2> <y2>

The script:
1) reads each frame block,
2) loads the matching video frame,
3) crops the tracked chimp region,
4) runs YOLOX face detection inside that region,
5) keeps only quality face crops (size/sharpness/brightness/contrast),
6) optionally keeps only likely camera-facing (frontal) faces,
7) saves crops in output/<chimp_name>/.
"""

from __future__ import annotations

import argparse
import csv
import queue
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


DEFAULT_CHIMP_NAMES = [
    "amadi",
    "banalia",
    "binasera",
    "djiku",
    "ivan",
    "jeje",
    "kalimi",
    "kassongo",
    "kira",
    "lwama",
    "malago",
    "maniema",
    "mazingara",
    "muke",
    "nganja",
    "nzuri",
    "penda",
    "talissa",
    "tanganica",
    "tingitingi",
]

# Common alias seen in annotations.
DEFAULT_ALIASES = {
    "kalemi": "kalimi",
}


@dataclass
class NamedBBox:
    raw_name: str
    canonical_name: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class QualityMetrics:
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.strip().lower())


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "Code").exists():
            return p
    return start


def parse_aliases(alias_items: list[str]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for item in alias_items:
        if "=" not in item:
            continue
        left, right = item.split("=", 1)
        left_n = normalize_token(left)
        right_n = normalize_token(right)
        if left_n and right_n:
            alias_map[left_n] = right_n
    return alias_map


def build_name_resolver(chimp_names: list[str], alias_map: dict[str, str]):
    canonical_map = {normalize_token(name): normalize_token(name) for name in chimp_names}

    def resolve(raw_name: str) -> str | None:
        key = normalize_token(raw_name)
        if not key:
            return None
        key = alias_map.get(key, key)
        return canonical_map.get(key)

    return resolve


def parse_track_file(track_path: Path, resolve_name) -> list[list[NamedBBox]]:
    frames: list[list[NamedBBox]] = []
    current: list[NamedBBox] | None = None

    with track_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line == "#":
                if current is None:
                    current = []
                else:
                    frames.append(current)
                    current = []
                continue

            if current is None:
                continue

            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            raw_name = parts[0]
            canonical_name = resolve_name(raw_name)
            if canonical_name is None:
                continue

            try:
                x1, y1, x2, y2 = map(float, parts[1:5])
            except ValueError:
                continue

            if not np.isfinite([x1, y1, x2, y2]).all():
                continue

            current.append(
                NamedBBox(
                    raw_name=raw_name,
                    canonical_name=canonical_name,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    if current is not None:
        frames.append(current)

    return frames


def _extract_state_dict_for_yolox(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict"):
            val = ckpt.get(key)
            if isinstance(val, dict) and len(val) > 0:
                return val
        if len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    keys = list(ckpt.keys())[:20] if isinstance(ckpt, dict) else []
    raise KeyError(f"Invalid YOLOX checkpoint format. Found keys: {keys}")


def _strip_prefixes(state_dict):
    prefixes = ("model.", "module.")
    cleaned = state_dict
    changed = True
    while changed and len(cleaned) > 0:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in cleaned.keys()):
                cleaned = {k[len(prefix):]: v for k, v in cleaned.items()}
                changed = True
    return cleaned


def load_yolox(chimpufe_src: Path, weights_path: Path, device):
    if str(chimpufe_src) not in sys.path:
        sys.path.append(str(chimpufe_src))

    from tracker.yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
    from tracker.yolox.utils import fuse_model

    depth, width = 1.33, 1.25
    in_channels = [256, 512, 1024]
    num_classes = 1

    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    model = YOLOX(backbone, head)
    model.head.initialize_biases(1e-2)

    ckpt = torch.load(str(weights_path), map_location="cpu")
    state_dict = _strip_prefixes(_extract_state_dict_for_yolox(ckpt))
    msg = model.load_state_dict(state_dict, strict=False)
    print("YOLOX loaded from:", weights_path)
    print("Missing keys:", len(msg.missing_keys), "| Unexpected keys:", len(msg.unexpected_keys))

    model = model.to(device).eval()
    model = fuse_model(model)
    return model


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _nms_xyxy(detections, iou_thres=0.60):
    if len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    kept = []
    for det in detections:
        x1, y1, x2, y2, _ = det
        overlap = False
        for kept_det in kept:
            kx1, ky1, kx2, ky2, _ = kept_det
            if _iou_xyxy((x1, y1, x2, y2), (kx1, ky1, kx2, ky2)) >= iou_thres:
                overlap = True
                break
        if not overlap:
            kept.append(det)
    return kept


def detect_face_yolox(img, model, device, input_size=(800, 1440), conf_thres=0.30, nms_iou_thres=0.60):
    results = detect_face_yolox_batch(
        [img],
        model=model,
        device=device,
        input_size=input_size,
        conf_thres=conf_thres,
        nms_iou_thres=nms_iou_thres,
    )
    return results[0] if results else []


def detect_face_yolox_batch(imgs, model, device, input_size=(800, 1440), conf_thres=0.30, nms_iou_thres=0.60):
    """Run YOLOX on a list of body crops in a single GPU forward pass.

    Returns a list with one detections list per input image. Per-image
    letterbox ratios are tracked so coordinates are unpadded back to
    the original body-crop space.
    """
    if len(imgs) == 0:
        return []

    batch_padded = []
    ratios = []
    valid_mask = []
    for img in imgs:
        if img is None or img.size == 0:
            ratios.append(1.0)
            valid_mask.append(False)
            batch_padded.append(np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8))
            continue
        h0, w0 = img.shape[:2]
        if h0 <= 0 or w0 <= 0:
            ratios.append(1.0)
            valid_mask.append(False)
            batch_padded.append(np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8))
            continue
        ratio = min(input_size[0] / h0, input_size[1] / w0)
        new_w, new_h = int(w0 * ratio), int(h0 * ratio)
        resized = cv2.resize(img, (new_w, new_h))
        padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized
        batch_padded.append(padded)
        ratios.append(ratio)
        valid_mask.append(True)

    arr = np.stack(batch_padded, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
    tensor = torch.from_numpy(arr).to(device, non_blocking=True).float()

    with torch.no_grad():
        preds = model(tensor)

    preds_cpu = preds.detach().cpu().numpy()

    results: list[list[tuple[int, int, int, int, float]]] = []
    for i in range(len(imgs)):
        if not valid_mask[i]:
            results.append([])
            continue
        ratio = ratios[i]
        pred = preds_cpu[i]
        output = []
        for row in pred:
            x_c = float(row[0])
            y_c = float(row[1])
            w = float(row[2])
            h = float(row[3])
            obj_conf = float(row[4])
            class_conf = float(row[5])
            conf = obj_conf * class_conf
            if conf < conf_thres:
                continue
            x1 = int((x_c - w / 2) / ratio)
            y1 = int((y_c - h / 2) / ratio)
            x2 = int((x_c + w / 2) / ratio)
            y2 = int((y_c + h / 2) / ratio)
            output.append((x1, y1, x2, y2, conf))
        output.sort(key=lambda item: item[4], reverse=True)
        output = _nms_xyxy(output, iou_thres=nms_iou_thres)
        results.append(output)
    return results


def select_best_face_detection(dets, body_shape, args):
    """Pick the best face detection using geometry constraints.

    This prevents selecting full-body false positives by enforcing that the
    candidate box has plausible face area and vertical position inside the
    tracked chimp body box.
    """
    if len(dets) == 0:
        return None, "no_face_detected", None

    h_body, w_body = body_shape[:2]
    body_area = max(1.0, float(h_body * w_body))

    candidates = []
    for det in dets:
        x1, y1, x2, y2, conf = det
        w = max(1, int(x2 - x1))
        h = max(1, int(y2 - y1))
        area_ratio = (w * h) / body_area
        center_y = ((y1 + y2) / 2.0) / max(1.0, float(h_body))
        aspect = w / max(1.0, float(h))

        if area_ratio < args.min_face_area_ratio:
            continue
        if area_ratio > args.max_face_area_ratio:
            continue
        if center_y < args.min_face_center_y or center_y > args.max_face_center_y:
            continue
        if aspect < 0.45 or aspect > 2.4:
            continue

        candidates.append((det, area_ratio, center_y, aspect, conf))

    if len(candidates) == 0:
        if args.allow_face_geometry_fallback:
            best = dets[0]
            x1, y1, x2, y2, _ = best
            w = max(1, int(x2 - x1))
            h = max(1, int(y2 - y1))
            area_ratio = (w * h) / body_area
            center_y = ((y1 + y2) / 2.0) / max(1.0, float(h_body))
            return best, "geometry_fallback", (area_ratio, center_y)
        return None, "no_face_geometry_match", None

    candidates.sort(key=lambda t: t[4], reverse=True)
    best_det, area_ratio, center_y, _, _ = candidates[0]
    return best_det, "ok", (area_ratio, center_y)


def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def crop_from_box(img, det_box, margin_ratio=0.05):
    x1, y1, x2, y2, _ = det_box
    h_img, w_img = img.shape[:2]

    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))
    dx = int(margin_ratio * bw)
    dy = int(margin_ratio * bh)

    x1 = max(0, min(int(x1) - dx, w_img - 1))
    y1 = max(0, min(int(y1) - dy, h_img - 1))
    x2 = max(0, min(int(x2) + dx, w_img))
    y2 = max(0, min(int(y2) + dy, h_img))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


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


def frontal_score(face_bgr) -> float:
    """Estimate how likely a face crop is frontal (looking at camera).

    Combines left/right symmetry with a chimp-specific cue: a frontal
    chimp face shows a BRIGHTER central skin patch (the pinkish/tan
    muzzle and brow region) surrounded by DARKER fur. We measure the
    mean-brightness contrast between an inner ellipse (face skin) and
    the outer ring (surrounding fur). Higher inner-vs-outer brightness
    ratio == more likely a frontal face.
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
    # Build an inner-ellipse mask covering the central face area, and a
    # surrounding ring covering the outer fur. The face-skin patch is
    # usually 15-40 grey levels brighter than the dark fur on a frontal
    # chimp; in profile/back views the patch shrinks or disappears.
    H, W = gray.shape
    yy, xx = np.ogrid[:H, :W]
    cx_f, cy_f = W / 2.0, H / 2.0
    rx_inner, ry_inner = W * 0.30, H * 0.32  # central skin ellipse
    rx_outer, ry_outer = W * 0.48, H * 0.48  # full face boundary
    inner = ((xx - cx_f) / rx_inner) ** 2 + ((yy - cy_f) / ry_inner) ** 2 <= 1.0
    outer = ((xx - cx_f) / rx_outer) ** 2 + ((yy - cy_f) / ry_outer) ** 2 <= 1.0
    ring = outer & ~inner

    if inner.any() and ring.any():
        inner_mean = float(gray[inner].mean())
        ring_mean = float(gray[ring].mean())
        # Positive when inner is brighter than ring (frontal chimp face).
        # Map a 0..40 grey-level difference to 0..1.
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


def detect_eye_count(face_bgr, eye_cascade, min_rel_size=0.04):
    """Estimate visible eyes in a chimp face crop.

    Chimp eyes are the darkest spots on their (already dark) face when
    facing the camera. We:
      1) Enhance contrast with CLAHE on the upper face region.
      2) Threshold the darkest ~6% of pixels (percentile-based, robust
         to global brightness).
      3) Find small compact blobs that look like eye-sized dark regions.
      4) Require a well-aligned pair in the LEFT vs RIGHT half of the
         ROI, both above the mid-face line, to count as 2 eyes.

    `eye_cascade` is kept for backward compatibility but unused.
    """
    del eye_cascade  # unused; kept for API stability
    h, w = face_bgr.shape[:2]
    if h < 32 or w < 32:
        return 0

    # Upper-face ROI, trimmed on the sides to focus on eye region.
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
    # CLAHE boosts local contrast between dark eye and slightly less dark
    # surrounding skin/fur — essential for chimps.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Percentile-based dark threshold: only the darkest ~6% of the ROI.
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
        # Must be in upper 70% of ROI (not in nose/mouth zone).
        if cy > 0.70 * rh:
            continue
        # Compactness: area vs bbox area (eyes are roughly elliptical/round)
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
        # Only one side has candidates -> at most one eye.
        return 1

    # Pick the largest blob on each side and check pairing geometry.
    left.sort(key=lambda c: c[2], reverse=True)
    right.sort(key=lambda c: c[2], reverse=True)
    lcx, lcy, _ = left[0]
    rcx, rcy, _ = right[0]

    dy = abs(lcy - rcy)
    dx = abs(rcx - lcx)

    # Vertical alignment: eyes should be roughly level.
    if dy > 0.18 * rh:
        return 1
    # Horizontal separation: meaningful distance between the two eyes.
    if dx < 0.25 * rw:
        return 1
    # Not too far apart either (i.e. blobs on extreme edges).
    if dx > 0.90 * rw:
        return 1
    return 2


def gather_files(root: Path, patterns: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(root.rglob(pattern))
        else:
            files.extend(root.glob(pattern))
    return sorted([p for p in files if p.is_file()])


def stem_key(path: Path) -> str:
    return normalize_token(path.stem)


def build_pairs(tracks: list[Path], videos: list[Path]):
    exact_map = {p.stem.lower(): p for p in videos}
    norm_map: dict[str, list[Path]] = defaultdict(list)
    for video in videos:
        norm_map[stem_key(video)].append(video)

    pairs: list[tuple[Path, Path]] = []
    unmatched_tracks: list[Path] = []
    ambiguous_tracks: list[tuple[Path, list[Path]]] = []

    for track in tracks:
        exact = exact_map.get(track.stem.lower())
        if exact is not None:
            pairs.append((track, exact))
            continue

        matches = norm_map.get(stem_key(track), [])
        if len(matches) == 1:
            pairs.append((track, matches[0]))
        elif len(matches) > 1:
            ambiguous_tracks.append((track, matches))
        else:
            unmatched_tracks.append(track)

    return pairs, unmatched_tracks, ambiguous_tracks


def dedupe_by_name_keep_largest(frame_items: list[NamedBBox]) -> list[NamedBBox]:
    by_name: dict[str, NamedBBox] = {}
    for item in frame_items:
        area = abs((item.x2 - item.x1) * (item.y2 - item.y1))
        prev = by_name.get(item.canonical_name)
        if prev is None:
            by_name[item.canonical_name] = item
            continue
        prev_area = abs((prev.x2 - prev.x1) * (prev.y2 - prev.y1))
        if area > prev_area:
            by_name[item.canonical_name] = item
    return list(by_name.values())


def process_pair(
    track_path: Path,
    video_path: Path,
    output_root: Path,
    face_model,
    device,
    resolve_name,
    eye_cascade,
    args,
    accepted_counter: Counter,
):
    """Pipeline:
      [reader thread, CPU]  read frames + body crops -> work_queue
      [main thread,  GPU]   batch body crops -> YOLOX forward -> dispatch
      [worker pool,  CPU]   select/quality/frontal/eyes/save
    """
    frames = parse_track_file(track_path, resolve_name)
    if len(frames) == 0:
        print(f"[WARN] No valid frame blocks found in {track_path}")
        return [], []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return [], []

    video_stem = video_path.stem
    track_path_str = str(track_path)
    video_path_str = str(video_path)

    max_frame = len(frames)
    if args.max_frames > 0:
        max_frame = min(max_frame, args.max_frames)

    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    rows_lock = threading.Lock()
    counter_lock = threading.Lock()
    counters = {"accepted": 0, "rejected": 0}

    def add_rejected(row: dict) -> None:
        with rows_lock:
            rejected_rows.append(row)
            counters["rejected"] += 1

    def add_accepted(row: dict) -> None:
        with rows_lock:
            accepted_rows.append(row)
            counters["accepted"] += 1

    def post_process(frame_idx: int, item: NamedBBox, body_crop, face_dets) -> None:
        face_det, face_det_reason, geom_metrics = select_best_face_detection(
            face_dets, body_crop.shape[:2], args
        )
        if face_det is None:
            add_rejected({
                "video_path": video_path_str,
                "track_path": track_path_str,
                "frame_idx": frame_idx,
                "raw_name": item.raw_name,
                "canonical_name": item.canonical_name,
                "reason": face_det_reason,
            })
            return

        face_area_ratio = None
        face_center_y = None
        if geom_metrics is not None:
            face_area_ratio, face_center_y = geom_metrics

        face_crop = crop_from_box(body_crop, face_det, margin_ratio=args.margin_ratio)
        if face_crop is None:
            add_rejected({
                "video_path": video_path_str,
                "track_path": track_path_str,
                "frame_idx": frame_idx,
                "raw_name": item.raw_name,
                "canonical_name": item.canonical_name,
                "reason": "invalid_face_crop",
            })
            return

        body_h, body_w = body_crop.shape[:2]
        face_h, face_w = face_crop.shape[:2]
        face_area_ratio_crop = (face_w * face_h) / max(1.0, float(body_w * body_h))
        if face_area_ratio_crop < args.min_face_area_ratio:
            add_rejected({
                "video_path": video_path_str,
                "track_path": track_path_str,
                "frame_idx": frame_idx,
                "raw_name": item.raw_name,
                "canonical_name": item.canonical_name,
                "reason": "face_too_small_vs_body",
                "face_area_ratio": f"{face_area_ratio_crop:.4f}",
            })
            return

        good, reason, metrics = assess_quality(
            face_crop,
            min_w=args.min_face_width,
            min_h=args.min_face_height,
            min_sharpness=args.min_sharpness,
            min_contrast=args.min_contrast,
            min_brightness=args.min_brightness,
            max_brightness=args.max_brightness,
        )
        if not good:
            add_rejected({
                "video_path": video_path_str,
                "track_path": track_path_str,
                "frame_idx": frame_idx,
                "raw_name": item.raw_name,
                "canonical_name": item.canonical_name,
                "reason": reason,
                "sharpness": f"{metrics.sharpness:.3f}",
                "brightness": f"{metrics.brightness:.3f}",
                "contrast": f"{metrics.contrast:.3f}",
                "face_w": metrics.width,
                "face_h": metrics.height,
            })
            return

        face_front_score = frontal_score(face_crop)
        if args.front_only and face_front_score < args.min_front_score:
            add_rejected({
                "video_path": video_path_str,
                "track_path": track_path_str,
                "frame_idx": frame_idx,
                "raw_name": item.raw_name,
                "canonical_name": item.canonical_name,
                "reason": "not_frontal",
                "frontal_score": f"{face_front_score:.3f}",
                "face_w": metrics.width,
                "face_h": metrics.height,
            })
            return

        eye_count = detect_eye_count(face_crop, eye_cascade, min_rel_size=args.min_eye_rel_size)
        if args.require_eyes and eye_count < args.min_eye_detections:
            add_rejected({
                "video_path": video_path_str,
                "track_path": track_path_str,
                "frame_idx": frame_idx,
                "raw_name": item.raw_name,
                "canonical_name": item.canonical_name,
                "reason": "no_visible_eyes",
                "eye_count": eye_count,
                "frontal_score": f"{face_front_score:.3f}",
                "face_w": metrics.width,
                "face_h": metrics.height,
            })
            return

        out_dir = output_root / item.canonical_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with counter_lock:
            if args.max_per_chimp > 0 and accepted_counter[item.canonical_name] >= args.max_per_chimp:
                # Race: another worker filled the quota while we were processing.
                return
            accepted_counter[item.canonical_name] += 1
            index = accepted_counter[item.canonical_name]

        out_name = f"{video_stem}_f{frame_idx:08d}_{item.canonical_name}_{index:06d}.jpg"
        out_path = out_dir / out_name

        write_ok = cv2.imwrite(
            str(out_path),
            face_crop,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)],
        )
        if not write_ok:
            with counter_lock:
                accepted_counter[item.canonical_name] -= 1
            add_rejected({
                "video_path": video_path_str,
                "track_path": track_path_str,
                "frame_idx": frame_idx,
                "raw_name": item.raw_name,
                "canonical_name": item.canonical_name,
                "reason": "save_failed",
            })
            return

        add_accepted({
            "video_path": video_path_str,
            "track_path": track_path_str,
            "frame_idx": frame_idx,
            "raw_name": item.raw_name,
            "canonical_name": item.canonical_name,
            "face_conf": f"{face_det[4]:.5f}",
            "sharpness": f"{metrics.sharpness:.3f}",
            "brightness": f"{metrics.brightness:.3f}",
            "contrast": f"{metrics.contrast:.3f}",
            "frontal_score": f"{face_front_score:.3f}",
            "eye_count": eye_count,
            "face_area_ratio": f"{(face_area_ratio if face_area_ratio is not None else face_area_ratio_crop):.4f}",
            "face_center_y": f"{(face_center_y if face_center_y is not None else ((face_det[1] + face_det[3]) / 2.0 / max(1.0, float(body_h)))):.4f}",
            "face_selection": face_det_reason,
            "face_w": metrics.width,
            "face_h": metrics.height,
            "out_path": str(out_path),
        })

    # ---- Reader thread (CPU): pull frames + build body crops ----
    prefetch = max(2, int(args.prefetch_frames))
    work_queue: "queue.Queue" = queue.Queue(maxsize=prefetch)
    reader_error: list[BaseException] = []

    def reader() -> None:
        try:
            for frame_idx in range(max_frame):
                ok, frame = cap.read()
                if not ok:
                    break
                if args.frame_step > 1 and (frame_idx % args.frame_step != 0):
                    work_queue.put(("tick", frame_idx, None))
                    continue
                frame_items = dedupe_by_name_keep_largest(frames[frame_idx])
                body_items: list[tuple[NamedBBox, "np.ndarray"]] = []
                for item in frame_items:
                    if args.max_per_chimp > 0:
                        with counter_lock:
                            if accepted_counter[item.canonical_name] >= args.max_per_chimp:
                                continue
                    body_crop = safe_crop(frame, item.x1, item.y1, item.x2, item.y2)
                    if body_crop is None:
                        add_rejected({
                            "video_path": video_path_str,
                            "track_path": track_path_str,
                            "frame_idx": frame_idx,
                            "raw_name": item.raw_name,
                            "canonical_name": item.canonical_name,
                            "reason": "invalid_body_bbox",
                        })
                        continue
                    body_items.append((item, body_crop))
                work_queue.put(("frame", frame_idx, body_items))
        except BaseException as exc:  # noqa: BLE001 - propagate to main
            reader_error.append(exc)
        finally:
            work_queue.put(("done", -1, None))

    reader_thread = threading.Thread(target=reader, name=f"reader-{video_stem}", daemon=True)
    reader_thread.start()

    # ---- CPU worker pool for post-processing ----
    cpu_workers = max(1, int(args.num_cpu_workers))
    executor = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix=f"post-{video_stem}")
    futures: list = []

    progress = None
    if args.show_progress and tqdm is not None:
        progress = tqdm(
            total=max_frame,
            desc=f"Frames {video_stem}",
            unit="frame",
            dynamic_ncols=True,
        )

    pending_imgs: list = []
    pending_meta: list[tuple[int, NamedBBox, "np.ndarray"]] = []

    def flush_batch() -> None:
        if not pending_imgs:
            return
        batch_results = detect_face_yolox_batch(
            pending_imgs,
            face_model,
            device,
            conf_thres=args.face_conf_thres,
            nms_iou_thres=args.nms_iou_thres,
        )
        for (frame_idx, item, body_crop), dets in zip(pending_meta, batch_results):
            futures.append(executor.submit(post_process, frame_idx, item, body_crop, dets))
        pending_imgs.clear()
        pending_meta.clear()

    batch_size = max(1, int(args.batch_size))
    last_progress_update = 0

    while True:
        kind, frame_idx, body_items = work_queue.get()
        if kind == "done":
            break

        if kind == "frame" and body_items:
            for item, body_crop in body_items:
                pending_imgs.append(body_crop)
                pending_meta.append((frame_idx, item, body_crop))
                if len(pending_imgs) >= batch_size:
                    flush_batch()

        if progress is not None:
            target = frame_idx + 1
            if target > last_progress_update:
                progress.update(target - last_progress_update)
                last_progress_update = target
                if frame_idx % 20 == 0:
                    progress.set_postfix(
                        accepted=counters["accepted"],
                        rejected=counters["rejected"],
                        queue=work_queue.qsize(),
                        refresh=False,
                    )

    flush_batch()
    reader_thread.join()

    for future in futures:
        future.result()
    executor.shutdown(wait=True)

    cap.release()
    if progress is not None:
        progress.set_postfix(
            accepted=counters["accepted"],
            rejected=counters["rejected"],
            refresh=False,
        )
        progress.close()

    if reader_error:
        raise reader_error[0]

    return accepted_rows, rejected_rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    repo_root = find_repo_root(Path(__file__).resolve())
    default_chimpufe_src = repo_root / "Code" / "chimplib" / "ChimpUFE" / "src"
    default_yolox_weights = repo_root / "Models" / "Face Recognition" / "yolox_best_only_model.pth"

    parser = argparse.ArgumentParser(description="Extract high-quality chimp face crops from named tracking text + videos")

    parser.add_argument("--output-root", type=str, required=True, help="Directory where per-chimp folders are saved")

    parser.add_argument("--track-path", type=str, default=None, help="Single tracking txt file")
    parser.add_argument("--video-path", type=str, default=None, help="Single video file matching --track-path")

    parser.add_argument("--tracks-dir", type=str, default=None, help="Directory containing tracking txt files")
    parser.add_argument("--videos-dir", type=str, default=None, help="Directory containing video files")
    parser.add_argument("--recursive", action="store_true", help="Recursively search in tracks/videos directories")
    parser.add_argument("--video-exts", type=str, default="mp4,mov,avi,mkv,m4v", help="Comma-separated video extensions")

    parser.add_argument(
        "--chimp-names",
        nargs="+",
        default=DEFAULT_CHIMP_NAMES,
        help="Allowed chimp names used as output folder names",
    )
    parser.add_argument(
        "--name-alias",
        nargs="*",
        default=[f"{k}={v}" for k, v in DEFAULT_ALIASES.items()],
        help="Optional aliases, format old=new (example: kalemi=kalimi)",
    )

    parser.add_argument("--chimpufe-src", type=str, default=str(default_chimpufe_src), help="Path to ChimpUFE src (contains tracker.yolox)")
    parser.add_argument("--yolox-weights", type=str, default=str(default_yolox_weights), help="YOLOX checkpoint path for face detection")

    parser.add_argument("--face-conf-thres", type=float, default=0.40, help="Min face detector confidence")
    parser.add_argument("--margin-ratio", type=float, default=0.05, help="Extra margin ratio around detected face")
    parser.add_argument("--min-face-width", type=int, default=90, help="Min accepted face crop width")
    parser.add_argument("--min-face-height", type=int, default=90, help="Min accepted face crop height")
    parser.add_argument("--min-face-area-ratio", type=float, default=0.04, help="Min face area / body area ratio")
    parser.add_argument("--max-face-area-ratio", type=float, default=0.35, help="Max face area / body area ratio")
    parser.add_argument("--min-face-center-y", type=float, default=0.02, help="Min vertical face center in body crop (0=top, 1=bottom)")
    parser.add_argument("--max-face-center-y", type=float, default=0.78, help="Max vertical face center in body crop (0=top, 1=bottom)")
    parser.add_argument("--nms-iou-thres", type=float, default=0.60, help="NMS IoU threshold for face detections")
    parser.add_argument("--allow-face-geometry-fallback", action="store_true", help="If no geometry-valid face is found, fallback to best-confidence detection")
    parser.add_argument("--min-sharpness", type=float, default=120.0, help="Min variance of Laplacian")
    parser.add_argument("--min-contrast", type=float, default=28.0, help="Min grayscale std")
    parser.add_argument("--min-brightness", type=float, default=35.0, help="Min grayscale mean")
    parser.add_argument("--max-brightness", type=float, default=220.0, help="Max grayscale mean")
    parser.add_argument("--front-only", action="store_true", help="Keep only faces likely looking at the camera")
    parser.add_argument("--min-front-score", type=float, default=0.0, help="Frontal score threshold in [0,1] (0 disables frontal thresholding by default)")
    parser.add_argument("--require-eyes", action="store_true", help="Require visible eyes in the face crop (stricter) using OpenCV cascade")
    parser.add_argument("--min-eye-detections", type=int, default=0, help="Minimum number of detected eyes when --require-eyes is enabled (0 disables by default)")
    parser.add_argument("--min-eye-rel-size", type=float, default=0.05, help="Minimum eye blob size relative to face crop (chimp-eye heuristic)")

    parser.add_argument("--frame-step", type=int, default=1, help="Keep one frame every N frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug limit per video, 0 = all")
    parser.add_argument("--max-per-chimp", type=int, default=0, help="Max saved crops per chimp, 0 = unlimited")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality for saved crops")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--show-progress", action="store_true", help="Show frame-level progress bar while processing")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of body crops per GPU forward pass")
    parser.add_argument("--num-cpu-workers", type=int, default=4, help="CPU worker threads for post-detection processing (quality, frontal, eyes, save)")
    parser.add_argument("--prefetch-frames", type=int, default=8, help="Max number of frames the reader thread can buffer ahead of the GPU")

    parser.add_argument("--accepted-csv", type=str, default=None, help="Output CSV for accepted crops")
    parser.add_argument("--rejected-csv", type=str, default=None, help="Output CSV for rejected candidates")

    return parser.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1")
    if args.min_front_score < 0.0 or args.min_front_score > 1.0:
        raise ValueError("--min-front-score must be in [0, 1]")
    if args.min_eye_detections < 0:
        raise ValueError("--min-eye-detections must be >= 0")
    if args.min_eye_rel_size <= 0.0 or args.min_eye_rel_size >= 0.5:
        raise ValueError("--min-eye-rel-size must be in (0, 0.5)")
    if args.min_face_area_ratio < 0.0 or args.max_face_area_ratio > 1.0 or args.min_face_area_ratio >= args.max_face_area_ratio:
        raise ValueError("face area ratio bounds must satisfy 0 <= min < max <= 1")
    if args.min_face_center_y < 0.0 or args.max_face_center_y > 1.0 or args.min_face_center_y >= args.max_face_center_y:
        raise ValueError("face center y bounds must satisfy 0 <= min < max <= 1")
    if args.nms_iou_thres <= 0.0 or args.nms_iou_thres >= 1.0:
        raise ValueError("--nms-iou-thres must be in (0, 1)")

    chimp_names = [normalize_token(name) for name in args.chimp_names if normalize_token(name)]
    alias_map = parse_aliases(args.name_alias)
    resolve_name = build_name_resolver(chimp_names, alias_map)

    for chimp_name in chimp_names:
        (output_root / chimp_name).mkdir(parents=True, exist_ok=True)

    track_video_pairs: list[tuple[Path, Path]] = []

    if args.track_path and args.video_path:
        track_video_pairs = [(Path(args.track_path).resolve(), Path(args.video_path).resolve())]
    elif args.tracks_dir and args.videos_dir:
        tracks_dir = Path(args.tracks_dir).resolve()
        videos_dir = Path(args.videos_dir).resolve()

        tracks = gather_files(tracks_dir, ["*.txt"], recursive=args.recursive)

        exts = [ext.strip().lstrip(".").lower() for ext in args.video_exts.split(",") if ext.strip()]
        video_patterns = [f"*.{ext}" for ext in exts]
        videos = gather_files(videos_dir, video_patterns, recursive=args.recursive)

        track_video_pairs, unmatched, ambiguous = build_pairs(tracks, videos)

        print(f"Tracks found: {len(tracks)} | Videos found: {len(videos)} | Pairs: {len(track_video_pairs)}")
        if len(unmatched) > 0:
            print("[WARN] Unmatched track files:")
            for p in unmatched[:20]:
                print("  -", p)
            if len(unmatched) > 20:
                print(f"  ... +{len(unmatched) - 20} more")

        if len(ambiguous) > 0:
            print("[WARN] Ambiguous track files (multiple matching videos):")
            for track_path, options in ambiguous[:20]:
                print("  -", track_path)
                for option in options[:3]:
                    print("      ->", option)
                if len(options) > 3:
                    print(f"      -> ... +{len(options) - 3} more")
            if len(ambiguous) > 20:
                print(f"  ... +{len(ambiguous) - 20} more")
    else:
        raise ValueError("Provide either (--track-path and --video-path) or (--tracks-dir and --videos-dir)")

    if len(track_video_pairs) == 0:
        print("No track/video pairs to process.")
        return

    chimpufe_src = Path(args.chimpufe_src).resolve()
    yolox_weights = Path(args.yolox_weights).resolve()

    if not chimpufe_src.exists():
        raise FileNotFoundError(f"ChimpUFE src not found: {chimpufe_src}")
    if not yolox_weights.exists():
        raise FileNotFoundError(f"YOLOX weights not found: {yolox_weights}")

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)

    face_model = load_yolox(chimpufe_src=chimpufe_src, weights_path=yolox_weights, device=device)

    # Eye check uses an internal heuristic tuned for chimps; no external
    # cascade file is needed. Kept as None for backward compatibility.
    eye_cascade = None

    accepted_all: list[dict] = []
    rejected_all: list[dict] = []
    accepted_counter: Counter = Counter()

    for idx, (track_path, video_path) in enumerate(track_video_pairs, start=1):
        print(f"[{idx}/{len(track_video_pairs)}] Processing")
        print("  track:", track_path)
        print("  video:", video_path)

        accepted_rows, rejected_rows = process_pair(
            track_path=track_path,
            video_path=video_path,
            output_root=output_root,
            face_model=face_model,
            device=device,
            resolve_name=resolve_name,
            eye_cascade=eye_cascade,
            args=args,
            accepted_counter=accepted_counter,
        )

        accepted_all.extend(accepted_rows)
        rejected_all.extend(rejected_rows)

        print(f"  accepted: {len(accepted_rows)} | rejected: {len(rejected_rows)}")

    accepted_csv = Path(args.accepted_csv).resolve() if args.accepted_csv else (output_root / "accepted_crops.csv")
    rejected_csv = Path(args.rejected_csv).resolve() if args.rejected_csv else (output_root / "rejected_candidates.csv")

    write_csv(accepted_csv, accepted_all)
    write_csv(rejected_csv, rejected_all)

    print("\n===== Summary =====")
    print("Total accepted:", len(accepted_all))
    print("Total rejected:", len(rejected_all))
    print("Accepted CSV:", accepted_csv)
    print("Rejected CSV:", rejected_csv)
    print("Per chimp accepted counts:")
    for chimp_name in sorted(chimp_names):
        print(f"  {chimp_name:12s} {accepted_counter[chimp_name]}")


if __name__ == "__main__":
    main()