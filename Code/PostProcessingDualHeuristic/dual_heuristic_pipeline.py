#!/usr/bin/env python3
"""Dual-heuristic post-processing for ByteTrack chimpanzee tracklets.

This module is a concise, self-contained implementation of the method described
in the thesis section "Dual-Heuristic Spatio-Temporal Post-Processing":

1. Parse ByteTrack's #-separated per-frame output.
2. Sample representative body crops from each tracklet.
3. Detect/select informative face crops.
4. Embed the selected faces with the original ChimpUFE backbone.
5. Average and L2-normalize each tracklet into one signature.
6. Build a cosine-distance matrix, constrained by co-alive tracklets.
7. Choose K with the physical lower bound and silhouette search.
8. Cluster tracklets with average-linkage agglomerative clustering.
9. Write the same per-frame format with anonymous cluster IDs.

Most of the code present below has been refactored and re-organized using AI agents,
but the overall logic and heuristics are preserved from the original implementation in the thesis experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


Box = tuple[float, float, float, float]
UnassignedPolicy = Literal["unique", "keep", "drop"]


@dataclass(slots=True)
class Detection:
    track_id: str
    box: Box


@dataclass(slots=True)
class Tracklet:
    track_id: str
    frames: list[int] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.frames[0]

    @property
    def end_frame(self) -> int:
        return self.frames[-1]

    @property
    def frame_set(self) -> set[int]:
        return set(self.frames)


@dataclass(slots=True)
class FaceCropCandidate:
    crop: np.ndarray
    frame_id: int
    detection_confidence: float
    quality_score: float
    reason: str = "ok"


@dataclass(slots=True)
class DualHeuristicConfig:
    samples_per_track: int = 5
    crop_pool_size: int = 30
    min_track_len: int = 30
    max_cluster_k: int = 20
    apply_temporal_penalty: bool = True
    temporal_penalty: float = 100.0
    cluster_prefix: str = "cluster_"
    cluster_start_index: int = 0
    unassigned_policy: UnassignedPolicy = "unique"
    use_spatial_recovery: bool = False
    recovery_gap: int = 60
    recovery_dist: float = 200.0
    centroid_filter_threshold: float | None = None
    device: str = "auto"

    # Face-crop quality controls recovered from the legacy experiments.
    face_det_conf: float = 0.20
    min_face_conf_keep: float = 0.32
    min_face_side_px: int = 20
    min_face_area_ratio: float = 0.015
    max_face_area_ratio: float = 0.60
    max_face_aspect_ratio: float = 2.2
    min_face_sharpness: float = 25.0
    min_face_contrast_std: float = 18.0
    target_face_area_ratio: float = 0.14
    min_face_quality_score_keep: float = 0.58
    max_face_center_y_ratio: float = 0.62
    min_face_center_x_ratio: float = 0.07
    max_face_center_x_ratio: float = 0.93
    min_face_border_margin_px: int = 3


@dataclass(slots=True)
class DualHeuristicResult:
    output_path: str
    n_frames: int
    n_tracklets: int
    n_candidate_tracklets: int
    n_embedded_tracklets: int
    n_unembedded_tracklets: int
    k_min: int
    selected_k: int
    silhouette_scores: dict[str, float]
    track_to_cluster: dict[str, str]
    distance_matrix_path: str | None = None
    diagnostics_path: str | None = None

    def to_json_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Tracking file IO
# ---------------------------------------------------------------------------


def parse_tracking_file(path: str | Path) -> tuple[list[list[Detection]], dict[str, Tracklet]]:
    """Parse the #-separated per-frame tracking format used in ChimpRec."""
    frames: list[list[Detection]] = []
    tracks: dict[str, Tracklet] = {}
    current_frame = -1

    with Path(path).open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line == "#":
                current_frame += 1
                frames.append([])
                continue

            if current_frame < 0:
                current_frame = 0
                frames.append([])

            parts = line.split()
            if len(parts) < 5:
                continue

            track_id = parts[0]
            try:
                box = tuple(float(value) for value in parts[1:5])
            except ValueError:
                continue
            if len(box) != 4:
                continue

            detection = Detection(track_id=track_id, box=box)  # type: ignore[arg-type]
            frames[current_frame].append(detection)
            if track_id not in tracks:
                tracks[track_id] = Tracklet(track_id=track_id)
            tracks[track_id].frames.append(current_frame)
            tracks[track_id].boxes.append(detection.box)

    return frames, tracks


def _format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def write_tracking_file(frames: Sequence[Sequence[Detection]], path: str | Path) -> None:
    """Write detections in the same #-separated per-frame format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for frame in frames:
            handle.write("#\n")
            for detection in frame:
                box_text = " ".join(_format_float(value) for value in detection.box)
                handle.write(f"{detection.track_id} {box_text}\n")


# ---------------------------------------------------------------------------
# ChimpUFE embedding and face detection
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class ChimpUFEEmbedder:
    """Wrapper around ChimpUFE's ViT backbone."""

    def __init__(self, weights_path: str | Path, chimpufe_root: str | Path, device: str = "auto"):
        import cv2
        import torch
        from PIL import Image
        from torchvision import transforms

        self.cv2 = cv2
        self.Image = Image
        self.torch = torch
        self.device = _resolve_device(device)

        chimpufe_root = Path(chimpufe_root).resolve()
        if str(chimpufe_root) not in sys.path:
            sys.path.insert(0, str(chimpufe_root))

        try:
            from src.face_embedder.vision_transformer import vit_base
        except ImportError as exc:
            raise ImportError(
                f"Cannot import ChimpUFE vit_base from {chimpufe_root}. "
                "Set --chimpufe-root to the ChimpUFE repository directory."
            ) from exc

        self.model = vit_base(patch_size=14)
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("teacher", checkpoint.get("model", checkpoint))
        self.model.load_state_dict(self._remap_chimpufe_state_dict(state_dict), strict=False)
        self.model.to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _remap_chimpufe_state_dict(self, state_dict: dict) -> dict:
        new_state_dict = {}
        model_keys = sorted(k for k in self.model.state_dict().keys() if "blocks." in k)
        ckpt_keys = sorted(
            k
            for k in state_dict.keys()
            if "blocks." in k.replace("backbone.", "").replace("module.", "") and "ls" not in k
        )
        for model_key, checkpoint_key in zip(model_keys, ckpt_keys):
            new_state_dict[model_key] = state_dict[checkpoint_key]
        for key, value in state_dict.items():
            cleaned_key = key.replace("backbone.", "").replace("module.", "")
            if "blocks." not in cleaned_key:
                new_state_dict[cleaned_key] = value
        return new_state_dict

    def embed(self, bgr_images: Sequence[np.ndarray], batch_size: int = 32) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        if not bgr_images:
            return embeddings

        with self.torch.inference_mode():
            for start in range(0, len(bgr_images), batch_size):
                tensors = []
                for image in bgr_images[start : start + batch_size]:
                    if image is None or image.size == 0:
                        continue
                    rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
                    pil = self.Image.fromarray(rgb)
                    tensors.append(self.transform(pil))
                if not tensors:
                    continue
                batch = self.torch.stack(tensors).to(self.device)
                output = self.model(batch)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output_np = output.detach().float().cpu().numpy()
                embeddings.extend(output_np)
        return embeddings


class UltralyticsFaceDetector:
    def __init__(self, model_path: str | Path, device: str = "auto"):
        from ultralytics import YOLO

        self.device = _resolve_device(device)
        self.model = YOLO(str(model_path))
        self.model.to(self.device)

    def detect_best(
        self,
        images: Sequence[np.ndarray],
        conf: float,
        batch_size: int = 32,
    ) -> list[tuple[float, Box] | None]:
        if not images:
            return []
        results = self.model.predict(
            list(images),
            conf=conf,
            verbose=False,
            stream=False,
            device=self.device,
            batch=min(batch_size, len(images)),
        )
        detections: list[tuple[float, Box] | None] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                detections.append(None)
                continue
            confidences = result.boxes.conf.detach().cpu().numpy()
            best_idx = int(np.argmax(confidences))
            xyxy = result.boxes.xyxy[best_idx].detach().cpu().numpy().astype(float)
            detections.append((float(confidences[best_idx]), tuple(xyxy)))  # type: ignore[arg-type]
        return detections


class YoloXFaceDetector:
    """Minimal loader for ChimpUFE YOLOX .pth face-detector weights."""

    def __init__(self, weights_path: str | Path, chimpufe_root: str | Path, device: str = "auto"):
        import torch
        import torchvision

        self.torch = torch
        self.torchvision = torchvision
        self.device = torch.device(_resolve_device(device))
        self.test_size = (800, 1440)
        self.nms_threshold = 0.45
        self.num_classes = 1

        chimpufe_root = Path(chimpufe_root).resolve()
        if str(chimpufe_root) not in sys.path:
            sys.path.insert(0, str(chimpufe_root))

        try:
            from src.tracker.yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
        except ImportError as exc:
            raise ImportError(
                f"Cannot import YOLOX modules from {chimpufe_root}. "
                "Use an Ultralytics .pt detector or set --chimpufe-root correctly."
            ) from exc

        backbone = YOLOPAFPN(1.33, 1.25, in_channels=[256, 512, 1024])
        head = YOLOXHead(self.num_classes, 1.25, in_channels=[256, 512, 1024])
        model = YOLOX(backbone, head)
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03
        model.head.initialize_biases(1e-2)

        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()
        self.model = model

    def _preprocess_batch(self, images: Sequence[np.ndarray]):
        processed = []
        ratios = []
        for image in images:
            image_h, image_w = image.shape[:2]
            ratio = min(self.test_size[0] / image_h, self.test_size[1] / image_w)
            resized_w, resized_h = int(image_w * ratio), int(image_h * ratio)
            padded = np.ones((self.test_size[0], self.test_size[1], 3), dtype=np.uint8) * 114
            resized = self._resize(image, resized_w, resized_h)
            padded[:resized_h, :resized_w] = resized
            processed.append(np.ascontiguousarray(padded.transpose((2, 0, 1)), dtype=np.float32))
            ratios.append(ratio)
        if not processed:
            return None, []
        batch = self.torch.from_numpy(np.stack(processed, axis=0)).to(self.device).float()
        return batch, ratios

    @staticmethod
    def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
        import cv2

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    def _postprocess(self, predictions, conf_threshold: float):
        box_corner = predictions.new(predictions.shape)
        box_corner[:, :, 0] = predictions[:, :, 0] - predictions[:, :, 2] / 2
        box_corner[:, :, 1] = predictions[:, :, 1] - predictions[:, :, 3] / 2
        box_corner[:, :, 2] = predictions[:, :, 0] + predictions[:, :, 2] / 2
        box_corner[:, :, 3] = predictions[:, :, 1] + predictions[:, :, 3] / 2
        predictions[:, :, :4] = box_corner[:, :, :4]

        outputs = [None for _ in range(len(predictions))]
        for image_idx, image_pred in enumerate(predictions):
            if image_pred.size(0) == 0:
                continue
            class_conf = self.torch.ones((image_pred.size(0), 1), device=image_pred.device, dtype=image_pred.dtype)
            class_pred = self.torch.zeros((image_pred.size(0), 1), device=image_pred.device, dtype=self.torch.long)
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_threshold).squeeze()
            detections = self.torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if detections.size(0) == 0:
                continue
            nms_input = detections[:, 4] * detections[:, 5]
            keep = self.torchvision.ops.batched_nms(detections[:, :4], nms_input, detections[:, 6], self.nms_threshold)
            outputs[image_idx] = detections[keep]
        return outputs

    def detect_best(
        self,
        images: Sequence[np.ndarray],
        conf: float,
        batch_size: int = 32,
    ) -> list[tuple[float, Box] | None]:
        if not images:
            return []
        all_best: list[tuple[float, Box] | None] = []
        with self.torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch, ratios = self._preprocess_batch(images[start : start + batch_size])
                if batch is None:
                    continue
                predictions = self.model(batch)
                outputs = self._postprocess(predictions, conf)
                for output, ratio in zip(outputs, ratios):
                    if output is None or output.size(0) == 0:
                        all_best.append(None)
                        continue
                    scores = output[:, 4] * output[:, 5]
                    best_idx = int(self.torch.argmax(scores).item())
                    score = float(scores[best_idx].item())
                    xyxy = tuple(float(value) for value in (output[best_idx, :4] / ratio).cpu().tolist())
                    all_best.append((score, xyxy))  # type: ignore[arg-type]
        return all_best


def build_face_detector(model_path: str | Path, chimpufe_root: str | Path, device: str = "auto"):
    if str(model_path).lower().endswith(".pth"):
        return YoloXFaceDetector(model_path, chimpufe_root=chimpufe_root, device=device)
    return UltralyticsFaceDetector(model_path, device=device)


# ---------------------------------------------------------------------------
# Crop selection and signature construction
# ---------------------------------------------------------------------------


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _sample_indices(length: int, pool_size: int) -> Iterable[int]:
    if length <= 0:
        return []
    if length <= pool_size:
        return range(length)
    return np.linspace(0, length - 1, pool_size, dtype=int)


def _clamp_box(box: Box, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _face_quality_score(metrics: dict[str, float], confidence: float, config: DualHeuristicConfig) -> float:
    area_score = max(
        0.0,
        1.0 - abs(metrics["area_ratio"] - config.target_face_area_ratio) / max(config.target_face_area_ratio, 1e-6),
    )
    sharp_score = min(1.0, metrics["sharpness"] / max(config.min_face_sharpness * 3.0, 1.0))
    contrast_score = min(1.0, metrics["contrast_std"] / max(config.min_face_contrast_std * 2.5, 1.0))
    confidence_score = max(0.0, min(1.0, confidence))
    return (0.35 * confidence_score) + (0.30 * sharp_score) + (0.20 * area_score) + (0.15 * contrast_score)


def _evaluate_face_crop(
    face_crop: np.ndarray,
    body_crop: np.ndarray,
    confidence: float,
    face_box: Box,
    config: DualHeuristicConfig,
) -> tuple[bool, float, str]:
    import cv2

    if confidence < config.min_face_conf_keep:
        return False, 0.0, "low_confidence"

    face_h, face_w = face_crop.shape[:2]
    if face_h < config.min_face_side_px or face_w < config.min_face_side_px:
        return False, 0.0, "too_small"

    body_h, body_w = body_crop.shape[:2]
    area_ratio = float(face_h * face_w) / float(max(1, body_h * body_w))
    if area_ratio < config.min_face_area_ratio or area_ratio > config.max_face_area_ratio:
        return False, 0.0, "bad_area_ratio"

    aspect = max(face_w / (face_h + 1e-6), face_h / (face_w + 1e-6))
    if aspect > config.max_face_aspect_ratio:
        return False, 0.0, "bad_aspect_ratio"

    fx1, fy1, fx2, fy2 = face_box
    center_x_ratio = ((fx1 + fx2) * 0.5) / max(body_w, 1)
    center_y_ratio = ((fy1 + fy2) * 0.5) / max(body_h, 1)
    if center_y_ratio > config.max_face_center_y_ratio:
        return False, 0.0, "too_low_in_body"
    if center_x_ratio < config.min_face_center_x_ratio or center_x_ratio > config.max_face_center_x_ratio:
        return False, 0.0, "too_lateral"

    margin = config.min_face_border_margin_px
    if fx1 <= margin or fy1 <= margin or fx2 >= body_w - margin or fy2 >= body_h - margin:
        return False, 0.0, "touches_border"

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    contrast_std = float(np.std(gray))
    if contrast_std < config.min_face_contrast_std:
        return False, 0.0, "low_contrast"

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_threshold = config.min_face_sharpness if confidence < 0.45 else config.min_face_sharpness * 0.6
    if sharpness < sharpness_threshold:
        return False, 0.0, "too_blurry"

    metrics = {
        "area_ratio": area_ratio,
        "sharpness": sharpness,
        "contrast_std": contrast_std,
    }
    quality_score = _face_quality_score(metrics, confidence, config)
    if quality_score < config.min_face_quality_score_keep:
        return False, quality_score, "low_quality_score"

    return True, quality_score, "ok"


def extract_representative_face_crops(
    video_path: str | Path,
    tracklet: Tracklet,
    face_detector,
    config: DualHeuristicConfig,
    save_crops_dir: str | Path | None = None,
) -> list[np.ndarray]:
    """Return up to config.samples_per_track high-quality face crops for one tracklet."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    sampled: list[tuple[int, np.ndarray]] = []

    for track_index in _sample_indices(len(tracklet.frames), config.crop_pool_size):
        frame_id = tracklet.frames[int(track_index)]
        box = tracklet.boxes[int(track_index)]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        clamped = _clamp_box(box, frame.shape)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped
        body_crop = frame[y1:y2, x1:x2]
        if body_crop.size > 0:
            sampled.append((frame_id, body_crop))
    cap.release()

    if not sampled:
        return []

    detections = face_detector.detect_best(
        [body_crop for _, body_crop in sampled],
        conf=config.face_det_conf,
        batch_size=32,
    )

    candidates: list[FaceCropCandidate] = []
    for (frame_id, body_crop), detection in zip(sampled, detections):
        if detection is None:
            continue
        confidence, face_xyxy = detection
        clamped_face = _clamp_box(face_xyxy, body_crop.shape)
        if clamped_face is None:
            continue
        fx1, fy1, fx2, fy2 = clamped_face
        face_crop = body_crop[fy1:fy2, fx1:fx2]
        if face_crop.size == 0:
            continue
        is_valid, quality_score, reason = _evaluate_face_crop(
            face_crop,
            body_crop,
            confidence,
            (float(fx1), float(fy1), float(fx2), float(fy2)),
            config,
        )
        if is_valid:
            candidates.append(
                FaceCropCandidate(
                    crop=face_crop,
                    frame_id=frame_id,
                    detection_confidence=confidence,
                    quality_score=quality_score,
                    reason=reason,
                )
            )

    if not candidates:
        return []

    candidates.sort(key=lambda item: item.quality_score, reverse=True)
    selected = candidates[: config.samples_per_track]

    if save_crops_dir is not None:
        crop_dir = Path(save_crops_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
        for rank, candidate in enumerate(selected):
            filename = (
                f"track_{_safe_token(tracklet.track_id)}_rank_{rank}_"
                f"frame_{candidate.frame_id}_score_{candidate.quality_score:.3f}.jpg"
            )
            cv2.imwrite(str(crop_dir / filename), candidate.crop)

    return [candidate.crop for candidate in selected]


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector) + 1e-10
    return vector / norm


def build_track_signature(
    embeddings: Sequence[np.ndarray],
    centroid_filter_threshold: float | None = None,
) -> np.ndarray | None:
    """Average per-crop embeddings into one L2-normalized track signature."""
    if not embeddings:
        return None
    normalized = np.array([_l2_normalize(np.asarray(embedding, dtype=np.float32)) for embedding in embeddings])
    if centroid_filter_threshold is not None and len(normalized) >= 2:
        centroid = _l2_normalize(np.mean(normalized, axis=0))
        keep = np.dot(normalized, centroid) > centroid_filter_threshold
        if np.any(keep):
            normalized = normalized[keep]
    signature = np.mean(normalized, axis=0)
    return _l2_normalize(signature).astype(np.float32)


def build_signatures_from_video(
    video_path: str | Path,
    tracklets: Sequence[Tracklet],
    face_detector,
    embedder: ChimpUFEEmbedder,
    config: DualHeuristicConfig,
    save_crops_dir: str | Path | None = None,
) -> tuple[list[Tracklet], np.ndarray, list[Tracklet]]:
    """Extract crops and signatures for candidate tracklets."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **_: x  # type: ignore[assignment]

    embedded_tracklets: list[Tracklet] = []
    signatures: list[np.ndarray] = []
    unembedded_tracklets: list[Tracklet] = []

    candidates = [tracklet for tracklet in tracklets if len(tracklet.frames) > config.min_track_len]
    for tracklet in tqdm(candidates, desc="Dual-heuristic crops"):
        crops = extract_representative_face_crops(
            video_path=video_path,
            tracklet=tracklet,
            face_detector=face_detector,
            config=config,
            save_crops_dir=save_crops_dir,
        )
        if not crops:
            unembedded_tracklets.append(tracklet)
            continue
        embeddings = embedder.embed(crops)
        signature = build_track_signature(embeddings, config.centroid_filter_threshold)
        if signature is None:
            unembedded_tracklets.append(tracklet)
            continue
        embedded_tracklets.append(tracklet)
        signatures.append(signature)

    if signatures:
        return embedded_tracklets, np.vstack(signatures), unembedded_tracklets
    return embedded_tracklets, np.empty((0, 0), dtype=np.float32), unembedded_tracklets


def load_precomputed_signatures(
    npz_path: str | Path,
    tracklets: Sequence[Tracklet],
) -> tuple[list[Tracklet], np.ndarray, list[Tracklet]]:
    """Load signatures from an .npz with arrays `track_ids` and `signatures`."""
    data = np.load(npz_path, allow_pickle=True)
    if "track_ids" not in data or "signatures" not in data:
        raise ValueError("Signature NPZ must contain arrays named 'track_ids' and 'signatures'.")
    track_ids = [str(track_id) for track_id in data["track_ids"].tolist()]
    signatures = np.asarray(data["signatures"], dtype=np.float32)
    if len(track_ids) != len(signatures):
        raise ValueError("Signature NPZ has different numbers of track_ids and signatures.")

    by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
    embedded_tracklets: list[Tracklet] = []
    aligned_signatures: list[np.ndarray] = []
    used_ids: set[str] = set()
    for track_id, signature in zip(track_ids, signatures):
        if track_id not in by_id:
            continue
        embedded_tracklets.append(by_id[track_id])
        aligned_signatures.append(_l2_normalize(signature).astype(np.float32))
        used_ids.add(track_id)

    unembedded_tracklets = [tracklet for tracklet in tracklets if tracklet.track_id not in used_ids]
    if aligned_signatures:
        return embedded_tracklets, np.vstack(aligned_signatures), unembedded_tracklets
    return embedded_tracklets, np.empty((0, 0), dtype=np.float32), unembedded_tracklets


# ---------------------------------------------------------------------------
# Dual heuristic: distance matrix, K selection, clustering
# ---------------------------------------------------------------------------


def cosine_distance_matrix(signatures: np.ndarray) -> np.ndarray:
    signatures = np.asarray(signatures, dtype=np.float32)
    if signatures.ndim != 2 or signatures.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    normalized = signatures / (np.linalg.norm(signatures, axis=1, keepdims=True) + 1e-10)
    distance = 1.0 - np.dot(normalized, normalized.T)
    distance[distance < 0] = 0.0
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32)


def apply_coalive_penalty(distance_matrix: np.ndarray, tracklets: Sequence[Tracklet], penalty: float = 100.0) -> np.ndarray:
    constrained = np.array(distance_matrix, copy=True)
    frame_sets = [tracklet.frame_set for tracklet in tracklets]
    for idx_a in range(len(tracklets)):
        for idx_b in range(idx_a + 1, len(tracklets)):
            if not frame_sets[idx_a].isdisjoint(frame_sets[idx_b]):
                constrained[idx_a, idx_b] = penalty
                constrained[idx_b, idx_a] = penalty
    return constrained


def physical_minimum_cluster_count(tracklets: Sequence[Tracklet]) -> int:
    frame_occupancy: dict[int, int] = {}
    for tracklet in tracklets:
        for frame_id in tracklet.frames:
            frame_occupancy[frame_id] = frame_occupancy.get(frame_id, 0) + 1
    return max(frame_occupancy.values()) if frame_occupancy else 1


def _agglomerative_labels(distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    n_items = len(distance_matrix)
    if n_items == 0:
        return np.array([], dtype=int)
    if n_clusters <= 1:
        return np.zeros(n_items, dtype=int)
    if n_clusters >= n_items:
        return np.arange(n_items, dtype=int)
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
    return model.fit_predict(distance_matrix)


def select_cluster_count(
    tracklets: Sequence[Tracklet],
    distance_matrix: np.ndarray,
    max_k: int = 20,
) -> tuple[int, int, dict[str, float]]:
    """Choose K with K_min=max concurrent tracklets and silhouette search."""
    n_tracklets = len(tracklets)
    if n_tracklets <= 1:
        return n_tracklets, n_tracklets, {}

    k_min = physical_minimum_cluster_count(tracklets)
    if n_tracklets == 2:
        return 2, k_min, {}

    first_k = max(2, k_min)
    last_k = min(max_k, n_tracklets - 1)
    if first_k > last_k:
        return min(max(k_min, 1), n_tracklets), k_min, {}

    best_k = first_k
    best_score = -math.inf
    scores: dict[str, float] = {}
    for k_value in range(first_k, last_k + 1):
        labels = _agglomerative_labels(distance_matrix, k_value)
        if len(set(labels.tolist())) <= 1 or len(set(labels.tolist())) >= n_tracklets:
            continue
        try:
            score = float(silhouette_score(distance_matrix, labels, metric="precomputed"))
        except Exception:
            score = -math.inf
        scores[str(k_value)] = score
        if score > best_score:
            best_score = score
            best_k = k_value

    return best_k, k_min, scores


def cluster_tracklets(
    tracklets: Sequence[Tracklet],
    distance_matrix: np.ndarray,
    n_clusters: int,
    cluster_prefix: str = "cluster_",
    start_index: int = 0,
) -> dict[str, str]:
    labels = _agglomerative_labels(distance_matrix, n_clusters)
    return {
        tracklet.track_id: f"{cluster_prefix}{int(label) + start_index}"
        for tracklet, label in zip(tracklets, labels)
    }


def _track_center(box: Box) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def recover_unembedded_tracks(
    embedded_tracklets: Sequence[Tracklet],
    unembedded_tracklets: Sequence[Tracklet],
    track_to_cluster: dict[str, str],
    max_gap: int = 60,
    max_dist: float = 200.0,
) -> dict[str, str]:
    """Optional legacy compatibility step based on endpoint proximity."""
    anchors = []
    for tracklet in embedded_tracklets:
        cluster_id = track_to_cluster.get(tracklet.track_id)
        if cluster_id is None:
            continue
        anchors.append(
            {
                "cluster_id": cluster_id,
                "start_frame": tracklet.start_frame,
                "end_frame": tracklet.end_frame,
                "start_center": _track_center(tracklet.boxes[0]),
                "end_center": _track_center(tracklet.boxes[-1]),
                "frames": tracklet.frame_set,
            }
        )

    updated = dict(track_to_cluster)
    occupied_frames: dict[str, set[int]] = {}
    for tracklet in embedded_tracklets:
        cluster_id = updated.get(tracklet.track_id)
        if cluster_id is not None:
            occupied_frames.setdefault(cluster_id, set()).update(tracklet.frames)

    for tracklet in unembedded_tracklets:
        start_center = _track_center(tracklet.boxes[0])
        end_center = _track_center(tracklet.boxes[-1])
        frames = tracklet.frame_set
        best_cluster = None
        best_dist = math.inf

        for anchor in anchors:
            cluster_id = anchor["cluster_id"]
            if cluster_id in occupied_frames and not frames.isdisjoint(occupied_frames[cluster_id]):
                continue

            gap_forward = tracklet.start_frame - int(anchor["end_frame"])
            if 0 < gap_forward < max_gap:
                dist = float(np.linalg.norm(start_center - anchor["end_center"]))
                if dist < max_dist and dist < best_dist:
                    best_dist, best_cluster = dist, cluster_id

            gap_backward = int(anchor["start_frame"]) - tracklet.end_frame
            if 0 < gap_backward < max_gap:
                dist = float(np.linalg.norm(end_center - anchor["start_center"]))
                if dist < max_dist and dist < best_dist:
                    best_dist, best_cluster = dist, cluster_id

        if best_cluster is not None:
            updated[tracklet.track_id] = best_cluster
            occupied_frames.setdefault(best_cluster, set()).update(frames)

    return updated


def relabel_frames(
    frames: Sequence[Sequence[Detection]],
    track_to_cluster: dict[str, str],
    unassigned_policy: UnassignedPolicy = "unique",
    cluster_prefix: str = "cluster_",
    next_cluster_index: int = 0,
) -> tuple[list[list[Detection]], dict[str, str]]:
    final_map = dict(track_to_cluster)
    next_index = next_cluster_index
    relabelled: list[list[Detection]] = []

    for frame in frames:
        new_frame: list[Detection] = []
        for detection in frame:
            mapped_id = final_map.get(detection.track_id)
            if mapped_id is None:
                if unassigned_policy == "drop":
                    continue
                if unassigned_policy == "keep":
                    mapped_id = detection.track_id
                else:
                    mapped_id = f"{cluster_prefix}{next_index}"
                    final_map[detection.track_id] = mapped_id
                    next_index += 1
            new_frame.append(Detection(track_id=mapped_id, box=detection.box))
        relabelled.append(new_frame)
    return relabelled, final_map


# ---------------------------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------------------------


def run_dual_heuristic(
    tracker_txt: str | Path,
    output_txt: str | Path,
    *,
    video_path: str | Path | None = None,
    face_model_path: str | Path | None = None,
    chimpufe_weights_path: str | Path | None = None,
    chimpufe_root: str | Path | None = None,
    signatures_npz: str | Path | None = None,
    config: DualHeuristicConfig | None = None,
    diagnostics_json: str | Path | None = None,
    distance_matrix_npy: str | Path | None = None,
    save_crops_dir: str | Path | None = None,
) -> DualHeuristicResult:
    config = config or DualHeuristicConfig()
    frames, tracklets_by_id = parse_tracking_file(tracker_txt)
    tracklets = sorted(tracklets_by_id.values(), key=lambda item: (item.start_frame, item.track_id))
    candidate_tracklets = [tracklet for tracklet in tracklets if len(tracklet.frames) > config.min_track_len]
    short_tracklets = [tracklet for tracklet in tracklets if len(tracklet.frames) <= config.min_track_len]

    if signatures_npz is not None:
        embedded_tracklets, signatures, unembedded_tracklets = load_precomputed_signatures(
            signatures_npz,
            candidate_tracklets,
        )
    else:
        missing = [
            name
            for name, value in {
                "video_path": video_path,
                "face_model_path": face_model_path,
                "chimpufe_weights_path": chimpufe_weights_path,
                "chimpufe_root": chimpufe_root,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "Missing required inputs for feature extraction: " + ", ".join(missing) + ". "
                "Use --signatures-npz to skip model-based extraction."
            )
        face_detector = build_face_detector(face_model_path, chimpufe_root=chimpufe_root, device=config.device)
        embedder = ChimpUFEEmbedder(chimpufe_weights_path, chimpufe_root=chimpufe_root, device=config.device)
        embedded_tracklets, signatures, unembedded_tracklets = build_signatures_from_video(
            video_path=video_path,
            tracklets=tracklets,
            face_detector=face_detector,
            embedder=embedder,
            config=config,
            save_crops_dir=save_crops_dir,
        )

    track_to_cluster: dict[str, str] = {}
    selected_k = 0
    k_min = 0
    silhouette_scores: dict[str, float] = {}
    distance_matrix = np.empty((0, 0), dtype=np.float32)

    if len(embedded_tracklets) >= 1:
        distance_matrix = cosine_distance_matrix(signatures)
        if config.apply_temporal_penalty:
            distance_matrix = apply_coalive_penalty(distance_matrix, embedded_tracklets, config.temporal_penalty)

        if len(embedded_tracklets) == 1:
            selected_k = 1
            k_min = 1
        else:
            selected_k, k_min, silhouette_scores = select_cluster_count(
                embedded_tracklets,
                distance_matrix,
                max_k=config.max_cluster_k,
            )

        track_to_cluster = cluster_tracklets(
            embedded_tracklets,
            distance_matrix,
            n_clusters=selected_k,
            cluster_prefix=config.cluster_prefix,
            start_index=config.cluster_start_index,
        )

    unclustered_tracklets = [
        tracklet
        for tracklet in [*unembedded_tracklets, *short_tracklets]
        if tracklet.track_id not in track_to_cluster
    ]
    if config.use_spatial_recovery and track_to_cluster:
        track_to_cluster = recover_unembedded_tracks(
            embedded_tracklets=embedded_tracklets,
            unembedded_tracklets=unclustered_tracklets,
            track_to_cluster=track_to_cluster,
            max_gap=config.recovery_gap,
            max_dist=config.recovery_dist,
        )

    next_cluster_index = config.cluster_start_index + max(selected_k, 0)
    relabelled, final_track_to_cluster = relabel_frames(
        frames,
        track_to_cluster=track_to_cluster,
        unassigned_policy=config.unassigned_policy,
        cluster_prefix=config.cluster_prefix,
        next_cluster_index=next_cluster_index,
    )
    write_tracking_file(relabelled, output_txt)

    distance_path: str | None = None
    if distance_matrix_npy is not None and distance_matrix.size:
        matrix_path = Path(distance_matrix_npy)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(matrix_path, distance_matrix)
        distance_path = str(matrix_path)

    result = DualHeuristicResult(
        output_path=str(output_txt),
        n_frames=len(frames),
        n_tracklets=len(tracklets),
        n_candidate_tracklets=len(candidate_tracklets),
        n_embedded_tracklets=len(embedded_tracklets),
        n_unembedded_tracklets=len(unclustered_tracklets),
        k_min=k_min,
        selected_k=selected_k,
        silhouette_scores=silhouette_scores,
        track_to_cluster=final_track_to_cluster,
        distance_matrix_path=distance_path,
        diagnostics_path=str(diagnostics_json) if diagnostics_json is not None else None,
    )

    if diagnostics_json is not None:
        diagnostics_path = Path(diagnostics_json)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(result.to_json_dict(), indent=2))

    return result


def default_chimpufe_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "ChimpUFE",
        here.parent / "PostProcessing" / "ChimpUFE",
        here.parent / "FinalPipeline" / "ChimpUFE",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dual-heuristic ChimpUFE tracklet clustering.")
    parser.add_argument("--tracks", required=True, help="Input #-separated ByteTrack text file.")
    parser.add_argument("--output", required=True, help="Output text file with anonymous cluster IDs.")
    parser.add_argument("--video", help="Input video used to extract body and face crops.")
    parser.add_argument("--face-model", help="Face detector weights: Ultralytics .pt or ChimpUFE YOLOX .pth.")
    parser.add_argument("--chimpufe-weights", help="Original ChimpUFE backbone weights (.pth).")
    parser.add_argument("--chimpufe-root", default=str(default_chimpufe_root()), help="Path to the ChimpUFE repository.")
    parser.add_argument("--signatures-npz", help="Optional precomputed signatures NPZ with track_ids and signatures.")
    parser.add_argument("--diagnostics-json", help="Optional JSON diagnostics output.")
    parser.add_argument("--distance-matrix-npy", help="Optional .npy dump of the final distance matrix.")
    parser.add_argument("--save-crops-dir", help="Optional directory for selected face-crop previews.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--samples-per-track", type=int, default=5)
    parser.add_argument("--crop-pool-size", type=int, default=30)
    parser.add_argument("--min-track-len", type=int, default=30)
    parser.add_argument("--max-cluster-k", type=int, default=20)
    parser.add_argument("--cluster-prefix", default="cluster_")
    parser.add_argument("--unassigned-policy", choices=["unique", "keep", "drop"], default="unique")
    parser.add_argument("--disable-temporal-penalty", action="store_true")
    parser.add_argument("--use-spatial-recovery", action="store_true")
    parser.add_argument("--centroid-filter-threshold", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> DualHeuristicResult:
    args = build_arg_parser().parse_args(argv)
    config = DualHeuristicConfig(
        samples_per_track=args.samples_per_track,
        crop_pool_size=args.crop_pool_size,
        min_track_len=args.min_track_len,
        max_cluster_k=args.max_cluster_k,
        apply_temporal_penalty=not args.disable_temporal_penalty,
        cluster_prefix=args.cluster_prefix,
        unassigned_policy=args.unassigned_policy,
        use_spatial_recovery=args.use_spatial_recovery,
        centroid_filter_threshold=args.centroid_filter_threshold,
        device=args.device,
    )
    result = run_dual_heuristic(
        tracker_txt=args.tracks,
        output_txt=args.output,
        video_path=args.video,
        face_model_path=args.face_model,
        chimpufe_weights_path=args.chimpufe_weights,
        chimpufe_root=args.chimpufe_root,
        signatures_npz=args.signatures_npz,
        config=config,
        diagnostics_json=args.diagnostics_json,
        distance_matrix_npy=args.distance_matrix_npy,
        save_crops_dir=args.save_crops_dir,
    )
    print(json.dumps(result.to_json_dict(), indent=2))
    return result


if __name__ == "__main__":
    main()
