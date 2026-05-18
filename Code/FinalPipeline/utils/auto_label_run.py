#!/usr/bin/env python3
"""
auto_label_run.py — Chimpanzee auto-labelling pipeline (final).

Pipeline overview
-----------------
1. YOLO v8 body detection + ByteTrack multi-object tracking.
2. YOLOX face detection (ChimpUFE) to extract face crops per tracklet.
3. Per-track classification & 768-d ViT-base embedding via the fine-tuned
   ChimpUFE backbone (ArcFace head, val_top1 ≈ 0.889, 20 classes).
4. Greedy single-link embedding clustering with temporal and label-conflict
   guards (IoU-aware overlap tolerance, MERGE_SIM=0.55).
5. Cross-cluster same-name merge (first pass).
6. Cluster-aggregated classification using pooled crops across all cluster
   members (relaxed gate: CLUSTER_CONF=0.22, CLUSTER_MARGIN=0.04).
7. Second cross-cluster same-name merge (post-aggregation).
8. Embedding propagation from labelled to unlabelled clusters
   (PROP_SIM=0.50, no temporal overlap allowed).
9. Final cluster label assignment.
10. Per-frame dedup: instead of demoting whole clusters to unknown_N,
    only the lower-confidence box is dropped on frames where two boxes
    share the same identity.  Every other frame is unaffected, which
    preserves the "one identity per frame" annotation invariant while
    recovering the AssRe metric that whole-cluster demotion destroys.
11. Gap interpolation: linearly interpolate short gaps (≤ GAP_MAX_FRAMES)
    between fragments sharing the same identity within a cluster.
12. HOTA + recognition evaluation (when GT is available).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR         = Path(__file__).resolve().parent          # .../FinalPipeline/utils
PIPELINE_DIR       = SCRIPT_DIR.parent                        # .../FinalPipeline
PROJECT_ROOT       = PIPELINE_DIR.parent.parent               # .../ChimpRec
VIDEO_DIR          = PROJECT_ROOT / "ChimpVideos" / "input"
GT_DIR             = PROJECT_ROOT / "ChimpVideos" / "GT"
BODY_MODEL_PATH    = PIPELINE_DIR / "weights" / "Body_detection_model.pt"
YOLOX_FACE_WEIGHTS = PIPELINE_DIR / "weights" / "yolox_best_only_model.pth"
CLASSIFIER_WEIGHTS = PIPELINE_DIR / "weights" / "fine_tune_20.pt"
OUTPUT_DIR         = PIPELINE_DIR / "auto_label_outputs"
CHIMPPIC_DIR       = PROJECT_ROOT / "ChimpPic"

sys.path.insert(0, str(PIPELINE_DIR / "ChimpUFE"))   # face embedder + YOLOX
sys.path.insert(0, str(PIPELINE_DIR / "metric"))     # HOTA & visualize

from src.face_embedder.vision_transformer import vit_base                   
from src.tracker.yolox.models import YOLOX, YOLOPAFPN, YOLOXHead            

# HOTA / visualize are only used by the optional CLI evaluation block in
# ``main()``.  Import lazily so the notebook works even when ``metric/`` is
# missing.
try:
    from HOTA import HOTA, parse_chimp_file, calculate_iou                   
except ImportError:
    HOTA = parse_chimp_file = calculate_iou = None  # type: ignore

try:
    from visualize import draw_tracks                                        
except ImportError:
    draw_tracks = None  # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BYTETRACK_PARAMS = {
    "track_high_thresh": 0.6, "track_low_thresh": 0.1, "new_track_thresh": 0.7,
    "track_buffer": 70, "match_thresh": 0.9, "fuse_score": True,
    "detection_conf": 0.5,
}

YOLOX_TEST_SIZE   = (800, 1440)
YOLOX_CONF        = 0.20
YOLOX_NMS         = 0.65

MIN_TRACK_LEN     = 30       # tracklets shorter than this number of frames are ignored
FACE_POOL_SIZE    = 60       # frames sampled per track for face detection
FACE_TOP_K        = 12       # top-K face crops kept per track
SHARPNESS_FLOOR   = 25.0     # minimum Laplacian variance for a crop
SIZE_FLOOR        = 0.005    # minimum face-area / body-area ratio

CLASSIFIER_CONF   = 0.40     # per-track classification gate (score1 >= this)
CLASSIFIER_MARGIN = 0.05     # per-track margin gate (score1 - score2 >= this)

# Embedding clustering
MERGE_SIM         = 0.55     # cosine threshold for greedy single-link merge
MERGE_OVERLAP_TOL = 5        # overlap frames allowed before temporal rejection
IOU_DUP_THR       = 0.5      # mean IoU during co-alive frames: above = phantom

# Cluster-level rescue
CLUSTER_CONF      = 0.22     # cluster-aggregated gate (relaxed; more samples)
CLUSTER_MARGIN    = 0.04     # cluster-aggregated margin gate
MIN_CLUSTER_CROPS = 6        # minimum crops to attempt cluster classification
PROP_SIM          = 0.50     # min cosine sim for label propagation

# Output
GAP_MAX_FRAMES    = 30       # max gap (frames) to linearly interpolate
DROP_LEN          = 30       # tracklets shorter than this are dropped (None)

ID_MATCH_IOU      = 0.5
METRIC_KEYS = ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr"]

# ---------------------------------------------------------------------------
# Name aliasing — canonical GT spelling
# ---------------------------------------------------------------------------
NAME_ALIASES = {
    "Talisa":     "Talissa",
    "Tanganica":  "Tanganyika",
    "Muke":       "Muki",
    "Mazingira":  "Mazingara",
    "Talissa":    "Talissa",
    "Tanganyika": "Tanganyika",
    "Muki":       "Muki",
    "Mazingara":  "Mazingara",
}


def _gt_alias(name):
    if name is None:
        return None
    return NAME_ALIASES.get(str(name), str(name))


# ===========================================================================
# YOLOX face detector
# ===========================================================================
class YoloXFacePredictor:
    def __init__(self, ckpt_file, device, test_size=YOLOX_TEST_SIZE,
                 conf=YOLOX_CONF, nms=YOLOX_NMS):
        depth, width = 1.33, 1.25
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
        head     = YOLOXHead(1, width, in_channels=in_channels)
        model    = YOLOX(backbone, head)
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
        model.head.initialize_biases(1e-2)
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.test_size = test_size
        self.conf = conf
        self.nms  = nms

    def _preprocess(self, img):
        h, w = img.shape[:2]; ph, pw = self.test_size
        padded = np.full((ph, pw, 3), 114, dtype=np.uint8)
        r = min(ph / h, pw / w); nw, nh = int(w * r), int(h * r)
        padded[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(padded.transpose((2, 0, 1)), dtype=np.float32), r

    @torch.no_grad()
    def detect(self, frame_bgr):
        img, r = self._preprocess(frame_bgr)
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device).float()
        out = self.model(tensor)
        boxes = out.new_zeros(out.shape)
        boxes[:, :, 0] = out[:, :, 0] - out[:, :, 2] / 2
        boxes[:, :, 1] = out[:, :, 1] - out[:, :, 3] / 2
        boxes[:, :, 2] = out[:, :, 0] + out[:, :, 2] / 2
        boxes[:, :, 3] = out[:, :, 1] + out[:, :, 3] / 2
        out[:, :, :4] = boxes[:, :, :4]
        pred = out[0]
        cls_conf, _ = torch.max(pred[:, 5:6], 1, keepdim=True)
        conf_mask = (pred[:, 4] * cls_conf.squeeze() >= self.conf).squeeze()
        dets = torch.cat((pred[:, :5], cls_conf), dim=1)[conf_mask]
        if dets.size(0) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        keep = torchvision.ops.nms(dets[:, :4], dets[:, 4] * dets[:, 5], self.nms)
        dets = dets[keep].cpu().numpy()
        dets[:, :4] /= r
        scores = dets[:, 4] * dets[:, 5]
        return np.concatenate([dets[:, :4], scores[:, None]], axis=1)


# ===========================================================================
# Classifier — exposes both logits AND embeddings
# ===========================================================================
class ChimpClassifier(nn.Module):
    def __init__(self, num_classes=20, embed_dim=768):
        super().__init__()
        self.backbone = vit_base(
            img_size=224, patch_size=14, init_values=1e-5,
            ffn_layer="mlp", block_chunks=4,
            qkv_bias=True, proj_bias=True, ffn_bias=True,
            num_register_tokens=0, interpolate_offset=0.1,
            interpolate_antialias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes, bias=False)

    @torch.no_grad()
    def features(self, x):
        feat = self.norm(self.backbone(x))
        return F.normalize(feat, dim=1)

    @torch.no_grad()
    def forward(self, x):
        feat_n = self.features(x)
        w_n = F.normalize(self.head.weight, dim=1)
        return feat_n, feat_n @ w_n.T


def load_classifier(weights_path, device):
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_name = {v: k for k, v in class_to_idx.items()}
    state = ckpt.get("ema_state", ckpt["model_state"])
    model = ChimpClassifier(num_classes=len(class_to_idx))
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"  ChimpUFE fine-tuned (stage={ckpt['stage']}, val_top1={ckpt['val_top1']:.3f})")
    return model, idx_to_name


FACE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def encode_crops(model, cv2_crops, device, batch_size=32):
    """Return (embeddings [N,768], cosine_logits [N,K])."""
    if not cv2_crops:
        return np.zeros((0, 768)), np.zeros((0, 0))
    tensors = [FACE_TRANSFORM(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
               for c in cv2_crops]
    embs, logs = [], []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        e, l = model(batch)
        embs.append(e.cpu().numpy())
        logs.append(l.cpu().numpy())
    return np.concatenate(embs, 0), np.concatenate(logs, 0)


# ===========================================================================
# ByteTrack
# ===========================================================================
def _xyxy_to_xywh(xyxy):
    w = xyxy[:, 2] - xyxy[:, 0]; h = xyxy[:, 3] - xyxy[:, 1]
    cx = xyxy[:, 0] + w / 2; cy = xyxy[:, 1] + h / 2
    return np.stack([cx, cy, w, h], axis=1)


def run_bytetrack(video_path, body_model, params):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    print(f"  Detecting on {total} frames @ {fps:.1f} fps ...")
    min_conf = min(0.1, params["detection_conf"])
    cached = []
    for r in tqdm(
        body_model.predict(source=str(video_path), conf=min_conf, stream=True, verbose=False),
        total=total, desc="YOLO body",
    ):
        if r.boxes is not None and len(r.boxes) > 0:
            cached.append(np.hstack([
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy()[:, None],
                r.boxes.cls.cpu().numpy()[:, None],
            ]))
        else:
            cached.append(np.empty((0, 6)))
    args = SimpleNamespace(**{k: params[k] for k in
                              ("track_high_thresh", "track_low_thresh",
                               "new_track_thresh", "track_buffer",
                               "match_thresh", "fuse_score")})
    tracker = BYTETracker(args=args, frame_rate=max(1, int(fps)))
    all_frame_tracks = []
    for dets in tqdm(cached, desc="ByteTrack"):
        if len(dets):
            dets = dets[dets[:, 4] >= params["detection_conf"]]
        if len(dets) == 0:
            all_frame_tracks.append([])
            tracker.frame_id += 1
            continue
        det_obj = SimpleNamespace(
            conf=dets[:, 4], cls=dets[:, 5], xywh=_xyxy_to_xywh(dets[:, :4])
        )
        tracked = tracker.update(det_obj, img=None)
        ft = []
        if tracked is not None and len(tracked):
            for row in tracked:
                ft.append((int(row[4]), row[:4].tolist()))
        all_frame_tracks.append(ft)
    return all_frame_tracks, fps


def frame_tracks_to_track_dict(all_frame_tracks):
    tracks = {}
    for f_idx, ft in enumerate(all_frame_tracks):
        for tid, box in ft:
            tracks.setdefault(tid, {"frames": [], "boxes": [], "id": tid})
            tracks[tid]["frames"].append(f_idx)
            tracks[tid]["boxes"].append(box)
    return tracks


# ===========================================================================
# Face crops
# ===========================================================================
def _laplacian_var(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def collect_face_crops(video_path, tracks_to_sample, face_predictor,
                       pool_size=FACE_POOL_SIZE):
    frames_needed = defaultdict(list)
    for t in tracks_to_sample:
        n = len(t["frames"])
        idx_iter = range(n) if n <= pool_size else np.linspace(0, n-1, pool_size, dtype=int)
        for idx in idx_iter:
            frames_needed[t["frames"][idx]].append((t["id"], t["boxes"][idx]))
    sorted_frames = sorted(frames_needed.keys())
    print(f"  Sampling {len(sorted_frames)} unique frames for {len(tracks_to_sample)} tracks")
    track_crops = defaultdict(list)
    cap = cv2.VideoCapture(str(video_path))
    last_read = -1
    for f_id in tqdm(sorted_frames, desc="YOLOX face", leave=False):
        if f_id == last_read + 1:
            ret, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_id)
            ret, frame = cap.read()
        last_read = f_id
        if not ret:
            continue
        face_dets = face_predictor.detect(frame)
        if len(face_dets) == 0:
            continue
        for tid, bbox in frames_needed[f_id]:
            bx1, by1, bx2, by2 = bbox
            body_w = max(1.0, bx2 - bx1); body_h = max(1.0, by2 - by1)
            body_area = body_w * body_h
            best = None
            for fx1, fy1, fx2, fy2, conf in face_dets:
                cx = (fx1 + fx2) / 2; cy = (fy1 + fy2) / 2
                if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
                    continue
                ix1 = int(max(bx1, fx1, 0)); iy1 = int(max(by1, fy1, 0))
                ix2 = int(min(bx2, fx2, frame.shape[1]))
                iy2 = int(min(by2, fy2, frame.shape[0]))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                crop = frame[iy1:iy2, ix1:ix2].copy()
                if crop.size == 0:
                    continue
                f_area = (ix2 - ix1) * (iy2 - iy1)
                size_ratio = f_area / body_area
                sharp = _laplacian_var(crop)
                sharp_n = min(1.0, sharp / 200.0)
                size_n  = min(1.0, size_ratio / 0.10)
                score = float(conf) * sharp_n * size_n
                cand = {"crop": crop, "conf": float(conf),
                        "sharp": sharp, "size": size_ratio,
                        "frame": int(f_id), "score": float(score)}
                if best is None or cand["score"] > best["score"]:
                    best = cand
            if best is None:
                continue
            if best["sharp"] < SHARPNESS_FLOOR or best["size"] < SIZE_FLOOR:
                continue
            track_crops[tid].append(best)
            if len(track_crops[tid]) > 2 * FACE_TOP_K:
                track_crops[tid].sort(key=lambda c: c["score"], reverse=True)
                del track_crops[tid][2 * FACE_TOP_K:]
    cap.release()
    return track_crops


def select_top_crops(crops, top_k=FACE_TOP_K):
    if not crops:
        return []
    return sorted(crops, key=lambda c: c["score"], reverse=True)[:top_k]


# ===========================================================================
# Per-track classification + embedding
# ===========================================================================
def classify_and_embed_track(crops_dicts, classifier, idx_to_name,
                             conf_min=CLASSIFIER_CONF,
                             margin_min=CLASSIFIER_MARGIN):
    """Return (label_or_None, embedding_768, info_dict)."""
    crops = [c["crop"] for c in crops_dicts]
    embs, cos = encode_crops(classifier, crops, DEVICE)
    if cos.size == 0:
        return None, None, None
    mean_cos = cos.mean(axis=0)
    mean_emb = embs.mean(axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
    order = np.argsort(mean_cos)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    s1, s2 = float(mean_cos[top1]), float(mean_cos[top2])
    info = {
        "top1": _gt_alias(idx_to_name[top1]), "score1": s1,
        "top2": _gt_alias(idx_to_name[top2]), "score2": s2,
        "top3": _gt_alias(idx_to_name[int(order[2])]),
        "score3": float(mean_cos[int(order[2])]),
        "margin": s1 - s2, "n_crops": len(crops),
    }
    label = None
    if s1 >= conf_min and (s1 - s2) >= margin_min:
        label = _gt_alias(idx_to_name[top1])
    return label, mean_emb, info


# ===========================================================================
# Embedding-based track clustering
# ===========================================================================
class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}
        self.members = {x: {x} for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.p[rb] = ra
        self.members[ra].update(self.members[rb])
        del self.members[rb]
        return ra


def _track_overlap_iou(t1, t2):
    """Return (overlap_frames, mean_iou_during_overlap)."""
    f1 = {f: b for f, b in zip(t1["frames"], t1["boxes"])}
    f2 = {f: b for f, b in zip(t2["frames"], t2["boxes"])}
    common = f1.keys() & f2.keys()
    if not common:
        return 0, 0.0
    ious = []
    for f in common:
        ax1, ay1, ax2, ay2 = f1[f]
        bx1, by1, bx2, by2 = f2[f]
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        aa = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        ab = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = aa + ab - inter
        ious.append(inter / union if union > 0 else 0.0)
    return len(common), float(np.mean(ious))


def _overlap_bans_merge(t1, t2, overlap_tol, iou_dup_thr=IOU_DUP_THR):
    """Return True if the temporal overlap between t1 and t2 is real enough
    to forbid merging them into the same cluster.

    A short overlap (≤ overlap_tol frames) is tolerated (ByteTrack hand-offs).
    A longer overlap with high mean IoU is tolerated (duplicate phantom of the
    same animal).  Everything else bans the merge.
    """
    n_over, mean_iou = _track_overlap_iou(t1, t2)
    if n_over <= overlap_tol:
        return False
    return mean_iou < iou_dup_thr


def _clusters_overlap(members_a, members_b, by_id, overlap_tol):
    for ma in members_a:
        ta = by_id[ma]
        for mb in members_b:
            tb = by_id[mb]
            if _overlap_bans_merge(ta, tb, overlap_tol):
                return True
    return False


def cluster_tracks_by_embedding(tracks_with_emb, id_map_first_pass,
                                merge_sim=MERGE_SIM,
                                overlap_tol=MERGE_OVERLAP_TOL):
    """Greedy single-link agglomerative clustering of tracks.

    Constraints:
      - cosine similarity ≥ merge_sim,
      - no temporal overlap between cluster members (IoU-aware),
      - no conflicting classifier labels.
    """
    ids  = [t["id"] for t in tracks_with_emb]
    embs = np.stack([t["_emb"] for t in tracks_with_emb])
    by_id = {t["id"]: t for t in tracks_with_emb}

    labels_of = defaultdict(set)
    for tid in ids:
        if tid in id_map_first_pass:
            labels_of[tid].add(id_map_first_pass[tid])

    sim = embs @ embs.T
    n = len(ids)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= merge_sim:
                edges.append((float(sim[i, j]), ids[i], ids[j]))
    edges.sort(reverse=True)
    print(f"  Candidate edges (sim >= {merge_sim}): {len(edges)}")

    uf = UnionFind(ids)
    n_merge = n_reject_overlap = n_reject_label = 0

    for s, a, b in edges:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        la, lb = labels_of[ra], labels_of[rb]
        if la and lb and (la != lb):
            n_reject_label += 1
            continue
        bad = False
        for ma in uf.members[ra]:
            if bad:
                break
            ta = by_id[ma]
            for mb in uf.members[rb]:
                tb = by_id[mb]
                if _overlap_bans_merge(ta, tb, overlap_tol):
                    bad = True
                    break
        if bad:
            n_reject_overlap += 1
            continue
        new_root = uf.union(ra, rb)
        labels_of[new_root] = la | lb
        old_root = ra if new_root == rb else rb
        labels_of.pop(old_root, None)
        n_merge += 1

    print(f"  Merges accepted: {n_merge}  "
          f"(rejected: overlap={n_reject_overlap}, label-conflict={n_reject_label})")
    return uf, labels_of


def cross_cluster_name_merge(uf, labels_of, tracks_with_emb,
                             overlap_tol=MERGE_OVERLAP_TOL):
    """Merge clusters that share the same classifier label and have no
    temporal conflict.  Closes long temporal gaps that single-link similarity
    cannot bridge."""
    by_id = {t["id"]: t for t in tracks_with_emb}
    by_name = defaultdict(list)
    for root, names in labels_of.items():
        if len(names) == 1:
            by_name[next(iter(names))].append(root)

    n_merge = n_reject = 0
    for name, roots in by_name.items():
        if len(roots) < 2:
            continue
        base = roots[0]
        for r in roots[1:]:
            base_root  = uf.find(base)
            other_root = uf.find(r)
            if base_root == other_root:
                continue
            if _clusters_overlap(uf.members[base_root], uf.members[other_root],
                                 by_id, overlap_tol):
                n_reject += 1
                continue
            new_root = uf.union(base_root, other_root)
            labels_of[new_root] = {name}
            old = base_root if new_root == other_root else other_root
            labels_of.pop(old, None)
            n_merge += 1
    print(f"  Cross-cluster name-merges: accepted={n_merge}, rejected={n_reject}")
    return uf, labels_of


# ===========================================================================
# Cluster-aggregated classification + label propagation
# ===========================================================================
def aggregate_cluster_classifications(uf, track_crops, track_list,
                                      classifier, idx_to_name,
                                      conf_min=CLUSTER_CONF,
                                      margin_min=CLUSTER_MARGIN,
                                      min_crops=MIN_CLUSTER_CROPS):
    """Pool all face crops across each cluster's members and re-classify.

    Returns:
      cluster_label : root -> name | None
      cluster_emb   : root -> 768-d normalised embedding | None
      cluster_info  : root -> info dict | None
    """
    crops_by_track = {t["id"]: select_top_crops(track_crops.get(t["id"], []))
                      for t in track_list}
    cluster_label = {}
    cluster_emb   = {}
    cluster_info  = {}
    n_rescued = 0
    for root, members in uf.members.items():
        all_crops = []
        for m in members:
            all_crops.extend(crops_by_track.get(m, []))
        if len(all_crops) < min_crops:
            cluster_label[root] = None
            cluster_emb[root]   = None
            cluster_info[root]  = None
            continue
        cv2_crops = [c["crop"] for c in all_crops]
        embs, cos = encode_crops(classifier, cv2_crops, DEVICE)
        if cos.size == 0:
            cluster_label[root] = None
            cluster_emb[root]   = None
            cluster_info[root]  = None
            continue
        mean_cos = cos.mean(axis=0)
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
        order = np.argsort(mean_cos)[::-1]
        top1 = int(order[0]); top2 = int(order[1])
        s1 = float(mean_cos[top1]); s2 = float(mean_cos[top2])
        info = {
            "top1": _gt_alias(idx_to_name[top1]), "score1": s1,
            "top2": _gt_alias(idx_to_name[top2]), "score2": s2,
            "top3": _gt_alias(idx_to_name[int(order[2])]),
            "score3": float(mean_cos[int(order[2])]),
            "margin": s1 - s2, "n_crops": len(cv2_crops),
        }
        label = None
        if s1 >= conf_min and (s1 - s2) >= margin_min:
            label = _gt_alias(idx_to_name[top1])
            n_rescued += 1
        cluster_label[root] = label
        cluster_emb[root]   = mean_emb
        cluster_info[root]  = info
    print(f"  Cluster-aggregated classification: {n_rescued} clusters labelled "
          f"(of {len(uf.members)} total)")
    return cluster_label, cluster_emb, cluster_info


def merge_cluster_labels(uf, labels_of, cluster_label):
    """Inject cluster-aggregated labels into labels_of for the next
    cross-cluster name-merge pass.  Returns number of clusters newly named."""
    n_new = 0
    for root in list(uf.members.keys()):
        if labels_of.get(root):
            continue
        nm = cluster_label.get(root)
        if nm is not None:
            labels_of[root] = {nm}
            n_new += 1
    return n_new


def propagate_labels_by_embedding(uf, labels_of, cluster_emb, tracks_with_emb,
                                  prop_sim=PROP_SIM,
                                  overlap_tol=MERGE_OVERLAP_TOL):
    """For each unlabelled cluster, propagate from the nearest labelled cluster
    (by cosine on cluster-mean embeddings) if sim ≥ prop_sim and no temporal
    conflict.  Clusters are NOT merged, instead they share a name which boosts AssRe.
    """
    by_id = {t["id"]: t for t in tracks_with_emb}
    labelled = [(r, next(iter(s)), cluster_emb.get(r))
                for r, s in labels_of.items()
                if len(s) == 1 and cluster_emb.get(r) is not None]
    if not labelled:
        print("  Embedding propagation: no labelled clusters with embeddings.")
        return 0
    L_emb  = np.stack([e for _, _, e in labelled])
    L_root = [r for r, _, _ in labelled]
    L_name = [n for _, n, _ in labelled]
    n_prop = n_reject_overlap = 0
    for root in list(uf.members.keys()):
        if labels_of.get(root):
            continue
        emb = cluster_emb.get(root)
        if emb is None:
            continue
        sims = L_emb @ emb
        order = np.argsort(sims)[::-1]
        for idx in order:
            s = float(sims[idx])
            if s < prop_sim:
                break
            target_root = L_root[idx]
            target_name = L_name[idx]
            if _clusters_overlap(uf.members[root], uf.members[target_root],
                                 by_id, overlap_tol):
                n_reject_overlap += 1
                continue
            labels_of[root] = {target_name}
            n_prop += 1
            break
    print(f"  Embedding propagation (sim >= {prop_sim}): "
          f"accepted={n_prop}, overlap-rejected={n_reject_overlap}")
    return n_prop


def assign_cluster_labels(track_list, uf, labels_of, info_map):
    """Pick a final name per cluster (priority: explicit label > best-scoring
    member > unlabelled).  Returns {tid -> name | None}."""
    final = {}
    for root in list(uf.members.keys()):
        members = uf.members[root]
        names = labels_of.get(root, set())
        chosen = None
        if len(names) == 1:
            chosen = next(iter(names))
        else:
            best_score = -1.0; best_name = None
            for m in members:
                inf = info_map.get(m)
                if inf is None:
                    continue
                if inf["score1"] > best_score:
                    best_score = inf["score1"]
                    best_name  = inf["top1"]
            if best_name is not None and best_score >= 0.30:
                chosen = best_name
        for m in members:
            final[m] = chosen
    return final


# ===========================================================================
# Visual crop dump
# ===========================================================================
def dump_track_visuals(video_name, tracks, track_crops, id_map, info_map,
                       cluster_of):
    out_root = CHIMPPIC_DIR / video_name
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "_per_track_summary.csv"
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["track_id", "n_frames", "frame_start", "frame_end",
                    "n_crops_kept", "top1", "score1", "margin",
                    "top2", "score2", "top3", "score3",
                    "cluster_root", "final_label", "status"])
        for t in tracks:
            tid = t["id"]
            crops = select_top_crops(track_crops.get(tid, []))
            info  = info_map.get(tid)
            final_label = id_map.get(tid)
            if final_label is None or (isinstance(final_label, str)
                                       and final_label.startswith("unknown")):
                status = "unknown"
            elif info is None or info.get("top1") != final_label:
                status = "cluster-propagated"
            else:
                status = "labeled"
            w.writerow([
                tid, len(t["frames"]), t["frames"][0], t["frames"][-1],
                len(crops),
                info["top1"]   if info else "",
                f"{info['score1']:.4f}" if info else "",
                f"{info['margin']:.4f}" if info else "",
                info["top2"]   if info else "",
                f"{info['score2']:.4f}" if info else "",
                info["top3"]   if info else "",
                f"{info['score3']:.4f}" if info else "",
                cluster_of.get(tid, ""),
                final_label if final_label is not None else "",
                status,
            ])
            if not crops:
                continue
            pred_tag = (final_label or (info["top1"] if info else "noclass")
                        ).replace(" ", "_")
            track_dir = out_root / f"{status}__pred-{pred_tag}__t{tid:04d}"
            track_dir.mkdir(parents=True, exist_ok=True)
            for rank, c in enumerate(crops):
                fn = (f"crop_{rank:02d}_f{c['frame']:06d}"
                      f"_s{int(c['score']*1000):04d}"
                      f"_c{int(c['conf']*100):03d}"
                      f"_sh{int(c['sharp']):04d}"
                      f"_sz{int(c['size']*1000):03d}.jpg")
                cv2.imwrite(str(track_dir / fn), c["crop"])
            with open(track_dir / "_summary.txt", "w") as fts:
                fts.write(f"track_id={tid}\n")
                fts.write(
                    f"frames={t['frames'][0]}..{t['frames'][-1]} "
                    f"(n={len(t['frames'])})\n"
                )
                fts.write(f"crops_kept={len(crops)}\n")
                fts.write(f"status={status}\n")
                if info is not None:
                    fts.write(f"top1={info['top1']} score={info['score1']:.4f} "
                              f"margin={info['margin']:.4f}\n")
                    fts.write(f"top2={info['top2']} score={info['score2']:.4f}\n")
                    fts.write(f"top3={info['top3']} score={info['score3']:.4f}\n")
                fts.write(f"cluster_root={cluster_of.get(tid)}\n")
                fts.write(f"final_label={final_label}\n")
    print(f"  Crops written under {out_root}")
    print(f"  Per-track CSV: {csv_path}")


# ===========================================================================
# Gap interpolation
# ===========================================================================
def interpolate_gaps(all_frame_tracks, id_map, cluster_of,
                     gap_max=GAP_MAX_FRAMES):
    """Linearly interpolate boxes inside short gaps between fragments sharing
    the same final name within the same cluster.

    Operates in-place on ``all_frame_tracks`` and returns the number of
    synthetic boxes inserted.
    """
    groups = defaultdict(list)
    for f_idx, ft in enumerate(all_frame_tracks):
        for tid, box in ft:
            name = id_map.get(tid)
            if name is None or str(name).startswith("unknown"):
                continue
            root = cluster_of.get(tid, tid)
            groups[(root, name)].append((f_idx, tid, box))

    inserted_per_frame = defaultdict(list)
    n_added = 0
    for (root, name), entries in groups.items():
        entries.sort(key=lambda x: x[0])
        by_frame = {}
        for f_idx, tid, box in entries:
            by_frame.setdefault(f_idx, (tid, box))
        frames_sorted = sorted(by_frame.keys())
        for i in range(len(frames_sorted) - 1):
            f_a = frames_sorted[i]; f_b = frames_sorted[i + 1]
            gap = f_b - f_a
            if gap <= 1 or gap > gap_max:
                continue
            tid_a, box_a = by_frame[f_a]
            ba = np.array(box_a, dtype=np.float64)
            bb = np.array(by_frame[f_b][1], dtype=np.float64)
            for k in range(1, gap):
                t = k / gap
                interp = (1.0 - t) * ba + t * bb
                inserted_per_frame[f_a + k].append((tid_a, interp.tolist()))
                n_added += 1

    if n_added == 0:
        return 0
    for f_idx, extras in inserted_per_frame.items():
        if 0 <= f_idx < len(all_frame_tracks):
            all_frame_tracks[f_idx].extend(extras)
    return n_added


# ===========================================================================
# Output: per-frame dedup write
# ===========================================================================
def _track_confidence(tid, cluster_of, cluster_info, info_map):
    """Classifier confidence for track ``tid``.

    Prefers the cluster-aggregated score (many crops, lower variance);
    falls back to per-track score; finally 0.0.
    """
    root = cluster_of.get(tid)
    if root is not None and cluster_info is not None:
        cinf = cluster_info.get(root)
        if cinf is not None:
            return float(cinf.get("score1", 0.0))
    inf = info_map.get(tid) if info_map is not None else None
    if inf is not None:
        return float(inf.get("score1", 0.0))
    return 0.0


def write_tracks(all_frame_tracks, id_map, output_path,
                 cluster_of, cluster_info, info_map,
                 drop_log=None):
    """Write tracks with per-frame dedup.

    For each frame, if two or more boxes share the same identity, only the
    box from the highest-confidence cluster is written; the others are
    silently dropped on that frame only.  The per-frame uniqueness invariant
    is therefore satisfied without demoting any entire cluster.

    ``drop_log`` (optional ``defaultdict(int)``) receives per-name drop counts.
    Returns ``(n_written, n_dropped)``.
    """
    n_written = n_dropped = 0
    with open(output_path, "w") as fh:
        for ft in all_frame_tracks:
            fh.write("#\n")
            by_name: dict = defaultdict(list)
            for tid, box in ft:
                nm = id_map.get(tid)
                if nm is None:
                    continue
                by_name[nm].append((tid, box))

            for nm, entries in by_name.items():
                if len(entries) == 1:
                    tid, box = entries[0]
                    out_nm = _gt_alias(str(nm)).replace(" ", "_")
                    fh.write(f"{out_nm} {box[0]} {box[1]} {box[2]} {box[3]}\n")
                    n_written += 1
                    continue
                # Multiple boxes share a name: keep the highest-confidence one
                ranked = sorted(
                    entries,
                    key=lambda e: _track_confidence(
                        e[0], cluster_of, cluster_info, info_map),
                    reverse=True,
                )
                tid, box = ranked[0]
                out_nm = _gt_alias(str(nm)).replace(" ", "_")
                fh.write(f"{out_nm} {box[0]} {box[1]} {box[2]} {box[3]}\n")
                n_written += 1
                if drop_log is not None:
                    drop_log[nm] += len(entries) - 1
                n_dropped += len(entries) - 1
    return n_written, n_dropped


# ===========================================================================
# Evaluation
# ===========================================================================
def evaluate_hota(pred_file, gt_file):
    pred_frames = parse_chimp_file(str(pred_file))
    gt_frames   = parse_chimp_file(str(gt_file))
    n = min(len(pred_frames), len(gt_frames))
    if n == 0:
        return None
    pred_frames, gt_frames = pred_frames[:n], gt_frames[:n]
    gt_ids   = sorted({d["id"] for f in gt_frames   for d in f})
    pred_ids = sorted({d["id"] for f in pred_frames for d in f})
    if not gt_ids or not pred_ids:
        return None
    g2i = {nm: i for i, nm in enumerate(gt_ids)}
    p2i = {nm: i for i, nm in enumerate(pred_ids)}
    data = {
        "num_tracker_dets": 0, "num_gt_dets": 0,
        "num_tracker_ids": len(pred_ids), "num_gt_ids": len(gt_ids),
        "gt_ids": [], "tracker_ids": [], "similarity_scores": [],
    }
    for t in range(n):
        gts, preds = gt_frames[t], pred_frames[t]
        data["num_gt_dets"]      += len(gts)
        data["num_tracker_dets"] += len(preds)
        data["gt_ids"].append(
            np.array([g2i[d["id"]] for d in gts], dtype=np.int32))
        data["tracker_ids"].append(
            np.array([p2i[d["id"]] for d in preds], dtype=np.int32))
        sim = np.zeros((len(gts), len(preds)))
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                sim[i, j] = calculate_iou(g["box"], p["box"])
        data["similarity_scores"].append(sim)
    res = HOTA().eval_sequence(data)
    return {k: float(np.mean(res[k]) if isinstance(res[k], np.ndarray) else res[k])
            for k in METRIC_KEYS}


def evaluate_recognition(pred_file, gt_file, iou_thr=ID_MATCH_IOU):
    pred_frames = parse_chimp_file(str(pred_file))
    gt_frames   = parse_chimp_file(str(gt_file))
    n = min(len(pred_frames), len(gt_frames))
    if n == 0:
        return None
    pred_frames, gt_frames = pred_frames[:n], gt_frames[:n]
    correct = wrong = matched = n_gt_total = 0
    per_gt_total   = defaultdict(int)
    per_gt_correct = defaultdict(int)
    confusion      = defaultdict(lambda: defaultdict(int))
    for gts, preds in zip(gt_frames, pred_frames):
        n_gt_total += len(gts)
        for g in gts:
            per_gt_total[g["id"]] += 1
        if not gts or not preds:
            for g in gts:
                confusion[g["id"]]["__missed__"] += 1
            continue
        ious = np.zeros((len(gts), len(preds)))
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                ious[i, j] = calculate_iou(g["box"], p["box"])
        used_g, used_p = set(), set()
        flat = sorted([(ious[i, j], i, j) for i in range(len(gts))
                                           for j in range(len(preds))],
                      reverse=True)
        for iou, i, j in flat:
            if iou < iou_thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i); used_p.add(j)
            matched += 1
            gname = gts[i]["id"]; pname = preds[j]["id"]
            if gname.lower() == pname.lower():
                correct += 1
                per_gt_correct[gname] += 1
            else:
                wrong += 1
            confusion[gname][pname] += 1
        for i, g in enumerate(gts):
            if i not in used_g:
                confusion[g["id"]]["__missed__"] += 1
    return {
        "IDAcc": correct / matched if matched else 0.0,
        "IDRec": correct / n_gt_total if n_gt_total else 0.0,
        "matched": matched, "correct": correct, "wrong": wrong,
        "n_gt_total": n_gt_total,
        "per_id_recall": {nm: per_gt_correct[nm] / per_gt_total[nm]
                          for nm in per_gt_total if per_gt_total[nm]},
        "per_id_total": dict(per_gt_total),
        "confusion": {g: dict(d) for g, d in confusion.items()},
    }


def print_recognition(rep):
    if rep is None:
        print("  (skip)")
        return
    print(f"  IDAcc   : {rep['IDAcc']:.4f}   "
          f"({rep['correct']}/{rep['matched']} matched preds correctly named)")
    print(f"  IDRec   : {rep['IDRec']:.4f}   "
          f"({rep['correct']}/{rep['n_gt_total']} GT dets correctly named)")
    print("  Per-GT recall:")
    for nm in sorted(rep["per_id_recall"].keys()):
        tot = rep["per_id_total"][nm]
        rec = rep["per_id_recall"][nm]
        print(f"    {nm:14s}: {rec:.3f}  ({int(round(rec*tot))}/{tot})")
    print("  Confusion (GT → predicted, top-5):")
    for nm in sorted(rep["confusion"].keys()):
        items = sorted(rep["confusion"][nm].items(), key=lambda x: -x[1])[:5]
        print(f"    {nm:14s}: " + ", ".join(f"{p}:{c}" for p, c in items))


# ===========================================================================
# Main
# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="ChimpRec auto-labelling pipeline (final version)."
    )
    p.add_argument("--video",          default="20241018 - 07h56",
                   help="Video name (without .MP4 extension).")
    p.add_argument("--no-viz",         action="store_true",
                   help="Skip rendering the output video overlay.")
    p.add_argument("--no-pics",        action="store_true",
                   help="Skip dumping face crops to ChimpPic/.")
    p.add_argument("--merge-sim",      type=float, default=MERGE_SIM,
                   help="Cosine similarity threshold for embedding clustering.")
    p.add_argument("--cluster-conf",   type=float, default=CLUSTER_CONF,
                   help="Confidence gate for cluster-aggregated classification.")
    p.add_argument("--cluster-margin", type=float, default=CLUSTER_MARGIN,
                   help="Margin gate for cluster-aggregated classification.")
    p.add_argument("--prop-sim",       type=float, default=PROP_SIM,
                   help="Cosine similarity threshold for label propagation.")
    p.add_argument("--gap-max",        type=int,   default=GAP_MAX_FRAMES,
                   help="Maximum gap (frames) to linearly interpolate.")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    video_name = args.video
    video_path = VIDEO_DIR / f"{video_name}.MP4"
    gt_path    = GT_DIR    / f"{video_name}-GT.txt"
    out_txt    = OUTPUT_DIR / f"{video_name}_autolabel.txt"
    out_mp4    = OUTPUT_DIR / f"{video_name}_autolabel.mp4"

    print("=" * 60)
    print(f"Auto-labelling (final): {video_path}")
    print(f"Device: {DEVICE}   merge_sim={args.merge_sim}  "
          f"cluster_conf={args.cluster_conf}  prop_sim={args.prop_sim}")
    print(f"overlap_tol={MERGE_OVERLAP_TOL}f  iou_dup_thr={IOU_DUP_THR}  "
          f"gap_max={args.gap_max}f")
    print("dedup: per-frame (keep highest-confidence box when same name "
          "appears more than once per frame)")
    print("=" * 60)

    # ----- 1. Models -----
    print("\n[1/9] Loading models ...")
    body_model = YOLO(str(BODY_MODEL_PATH))
    face_pred  = YoloXFacePredictor(YOLOX_FACE_WEIGHTS, DEVICE)
    classifier, idx_to_name = load_classifier(CLASSIFIER_WEIGHTS, DEVICE)
    idx_to_name = {i: n.capitalize() for i, n in idx_to_name.items()}
    print(f"  Classes ({len(idx_to_name)}): {list(idx_to_name.values())}")

    # ----- 2. ByteTrack -----
    print("\n[2/9] Body detection + ByteTrack ...")
    all_frame_tracks, fps = run_bytetrack(video_path, body_model, BYTETRACK_PARAMS)
    tracks = frame_tracks_to_track_dict(all_frame_tracks)
    track_list = sorted(tracks.values(), key=lambda x: x["frames"][0])
    print(f"  Raw tracklets: {len(track_list)}")
    long_tracks = [t for t in track_list if len(t["frames"]) > MIN_TRACK_LEN]
    print(f"  Long: {len(long_tracks)} | short: {len(track_list) - len(long_tracks)}")

    # ----- 3. Face extraction -----
    print("\n[3/9] Extracting face crops with ChimpUFE YOLOX ...")
    track_crops = collect_face_crops(video_path, track_list, face_pred,
                                     pool_size=FACE_POOL_SIZE)
    print(f"  Tracks with at least one face crop: "
          f"{len(track_crops)}/{len(track_list)}")

    # ----- 4. First-pass classification + per-track embeddings -----
    print("\n[4/9] Classification + embedding extraction ...")
    id_map_first = {}
    info_map     = {}
    track_emb    = {}
    for t in tqdm(track_list, desc="Classify+Emb"):
        if len(t["frames"]) <= MIN_TRACK_LEN:
            continue
        crops = select_top_crops(track_crops.get(t["id"], []))
        if not crops:
            continue
        label, emb, info = classify_and_embed_track(crops, classifier, idx_to_name)
        if info is not None:
            info_map[t["id"]] = info
        if emb is not None:
            track_emb[t["id"]] = emb
        if label is not None:
            id_map_first[t["id"]] = label
    print(f"  First-pass labelled: {len(id_map_first)}/{len(long_tracks)}")
    print(f"  Tracks with embedding: {len(track_emb)}")

    # ----- 5. Embedding clustering -----
    print(f"\n[5/9] Clustering tracks by embedding (sim >= {args.merge_sim}) ...")
    tracks_with_emb = []
    for t in track_list:
        if t["id"] in track_emb:
            tw = dict(t)
            tw["_emb"] = track_emb[t["id"]]
            tracks_with_emb.append(tw)
    uf, labels_of = cluster_tracks_by_embedding(
        tracks_with_emb, id_map_first, merge_sim=args.merge_sim,
    )

    # ----- 6. Cross-cluster name-merge (1st pass) -----
    print("\n[6/9] Cross-cluster same-name merging (first pass) ...")
    uf, labels_of = cross_cluster_name_merge(uf, labels_of, tracks_with_emb)

    # ----- 7. Cluster-aggregated classification -----
    print(f"\n[7/9] Cluster-aggregated classification "
          f"(conf>={args.cluster_conf}, margin>={args.cluster_margin}) ...")
    cluster_label, cluster_emb, cluster_info = aggregate_cluster_classifications(
        uf, track_crops, track_list, classifier, idx_to_name,
        conf_min=args.cluster_conf, margin_min=args.cluster_margin,
    )
    n_inj = merge_cluster_labels(uf, labels_of, cluster_label)
    print(f"  Injected {n_inj} aggregated labels into cluster label sets.")

    # ----- Cross-cluster name-merge (2nd pass, post-aggregation) -----
    print("  Cross-cluster same-name merging (second pass, post-aggregation) ...")
    uf, labels_of = cross_cluster_name_merge(uf, labels_of, tracks_with_emb)

    # ----- 7b. Embedding propagation -----
    propagate_labels_by_embedding(
        uf, labels_of, cluster_emb, tracks_with_emb,
        prop_sim=args.prop_sim, overlap_tol=MERGE_OVERLAP_TOL,
    )

    # ----- 7c. Final label assignment -----
    cluster_of = {}
    for root, members in uf.members.items():
        for m in members:
            cluster_of[m] = root
    final_id_map = assign_cluster_labels(track_list, uf, labels_of, info_map)

    id_map = {}
    for t in track_list:
        id_map[t["id"]] = final_id_map.get(t["id"])

    n_unknown = 0
    for t in track_list:
        if id_map.get(t["id"]) is not None:
            continue
        if len(t["frames"]) < DROP_LEN:
            id_map[t["id"]] = None
        else:
            n_unknown += 1
            id_map[t["id"]] = f"unknown_{n_unknown}"

    name_counts: dict = defaultdict(int)
    for val in id_map.values():
        if val is not None and not str(val).startswith("unknown"):
            name_counts[val] += 1
    n_labelled_tracks = sum(
        1 for val in id_map.values()
        if val is not None and not str(val).startswith("unknown")
    )
    print(f"\n  Final: labelled-tracks={n_labelled_tracks}, "
          f"unknown={n_unknown}, "
          f"dropped={sum(1 for v in id_map.values() if v is None)}, "
          f"distinct named identities={len(name_counts)}")
    for nm, c in sorted(name_counts.items(), key=lambda x: -x[1]):
        print(f"    {nm:14s}: {c} tracks")

    # ----- 8. Visual crop dump -----
    if not args.no_pics:
        print("\n[8/9] Dumping crops to ChimpPic/ ...")
        dump_track_visuals(video_name, track_list, track_crops,
                           id_map, info_map, cluster_of)
    else:
        print("\n[8/9] (--no-pics) skipping crop dump")

    # ----- 8a. Write output with per-frame dedup -----
    drop_log: dict = defaultdict(int)
    n_written, n_dropped = write_tracks(
        all_frame_tracks, id_map, out_txt,
        cluster_of, cluster_info, info_map, drop_log=drop_log,
    )
    print(f"  [write] {n_written} boxes written, "
          f"{n_dropped} boxes dropped (per-frame conflicts)")
    if drop_log:
        print("  Per-frame drops by name (top 10):")
        for nm, c in sorted(drop_log.items(), key=lambda x: -x[1])[:10]:
            print(f"    {nm:14s}: {c} boxes")
    print(f"  Wrote {out_txt}")

    # ----- 8b. Gap interpolation -----
    n_added = interpolate_gaps(all_frame_tracks, id_map, cluster_of,
                               gap_max=args.gap_max)
    print(f"\n[8b/9] Gap interpolation (gap_max={args.gap_max}): "
          f"{n_added} synthetic boxes added.")
    if n_added > 0:
        drop_log.clear()
        n_written, n_dropped = write_tracks(
            all_frame_tracks, id_map, out_txt,
            cluster_of, cluster_info, info_map, drop_log=drop_log,
        )
        print(f"  [rewrite] {n_written} boxes, {n_dropped} dropped")
        print(f"  Re-wrote {out_txt} with interpolated boxes")

    # ----- 9. Evaluation -----
    if gt_path.exists():
        print("\n[9/9] HOTA ...")
        m = evaluate_hota(out_txt, gt_path)
        if m:
            for k in METRIC_KEYS:
                print(f"    {k:6s}: {m[k]:.4f}")
        print("\n[9/9] Recognition accuracy ...")
        rep = evaluate_recognition(out_txt, gt_path)
        print_recognition(rep)
    else:
        print(f"\n[9/9] No GT at {gt_path} — skipping evaluation")

    if not args.no_viz:
        if draw_tracks is not None:
            print("\n[viz] Drawing tracks on video ...")
            draw_tracks(str(video_path), str(out_txt), str(out_mp4),
                        label_tag="Auto-Label (final)")
            print(f"  Wrote {out_mp4}")
        else:
            print("\n[viz] draw_tracks not available — skipping")

    print("\nDone.")


if __name__ == "__main__":
    main()
