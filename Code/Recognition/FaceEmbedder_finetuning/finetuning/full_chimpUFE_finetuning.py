

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2
import os

# Limit CPU BLAS/OMP threads before importing numpy to avoid pthread_create failures
# when many DataLoader workers are used (prevents OpenBLAS RLIMIT_NPROC errors).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
# Limit PyTorch intra/inter-op threads to avoid oversubscription
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Canonical chimp names (lowercase). Used to merge spelling variants.
CANONICAL_NAMES = [
    "amadi", "banalia", "binasera", "djiku", "ivan",
    "jeje", "kalemi", "kassongo", "kira", "lwama",
    "malago", "maniema", "mazingira", "muke", "nganja",
    "nzuri", "penda", "talisa", "tanganica", "tingitingi",
]

# Aliases observed in the data -> canonical name
NAME_ALIASES = {
    "kalimi": "kalemi",
    "talissa": "talisa",
    "mazingara": "mazingira",
    "muki": "muke",
}

# Initial codes used in test/ filenames and student folder names
INITIAL_TO_NAME = {
    "AD": "amadi", "BL": "banalia", "BS": "binasera", "DK": "djiku",
    "IV": "ivan", "JJ": "jeje", "KM": "kalemi", "KG": "kassongo",
    "KR": "kira", "LM": "lwama", "MG": "malago", "MM": "maniema",
    "MZ": "mazingira", "MK": "muke", "NJ": "nganja", "NR": "nzuri",
    "PD": "penda", "TS": "talisa", "TC": "tanganica", "TT": "tingitingi",
}


# ============================================================================
# Utilities
# ============================================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "Code").exists() and (p / "ChimpPic").exists():
            return p
    return start


def canonical_name(raw: str) -> str | None:
    """Normalize a folder/file token to a canonical chimp name."""
    s = raw.strip().lower()
    if s in CANONICAL_NAMES:
        return s
    if s in NAME_ALIASES:
        return NAME_ALIASES[s]
    # Folder names like "Amadi - AD" -> first token
    first = s.split(" - ")[0].split("-")[0].strip()
    if first in CANONICAL_NAMES:
        return first
    if first in NAME_ALIASES:
        return NAME_ALIASES[first]
    # Try initial code suffix at end of folder name (e.g. "Kassongo -KG" -> KG)
    parts = raw.replace("-", " ").split()
    for tok in parts[::-1]:
        tok_up = tok.strip().upper()
        if tok_up in INITIAL_TO_NAME:
            return INITIAL_TO_NAME[tok_up]
    return None


def name_from_test_filename(fname: str) -> str | None:
    """Test files look like 'AD1.png', 'BL3.png'."""
    stem = Path(fname).stem
    # Strip trailing digits
    code = "".join(ch for ch in stem if ch.isalpha()).upper()
    if not code:
        return None
    # Match longest known initial prefix
    for k in sorted(INITIAL_TO_NAME.keys(), key=len, reverse=True):
        if code.startswith(k):
            return INITIAL_TO_NAME[k]
    return None


# ============================================================================
# Dataset
# ============================================================================

@dataclass
class Sample:
    img_path: Path
    target: int
    tier: int  # 1, 2 or 3
    weight: float


def collect_tier1(root: Path, class_to_idx: dict[str, int], tier_weight: float) -> list[Sample]:
    samples: list[Sample] = []
    if not root.exists():
        return samples
    for chimp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        canon = canonical_name(chimp_dir.name)
        if canon is None or canon not in class_to_idx:
            continue
        target = class_to_idx[canon]
        # Support either flat dir or images/ subdir
        candidate_dirs = [chimp_dir]
        img_subdir = chimp_dir / "images"
        if img_subdir.exists():
            candidate_dirs.append(img_subdir)
        for d in candidate_dirs:
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    samples.append(Sample(p, target, tier=1, weight=tier_weight))
    return samples


def collect_tier2_split(root: Path, class_to_idx: dict[str, int], tier_weight: float,
                        val_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Per-chimp deterministic train/val split of student annotations.

    The split is per-chimp so every class is represented in val (as long as it
    has at least 2 images).
    """
    rng = random.Random(seed)
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    if not root.exists():
        return train_samples, val_samples
    for chimp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        canon = canonical_name(chimp_dir.name)
        if canon is None or canon not in class_to_idx:
            continue
        target = class_to_idx[canon]
        files = sorted(p for p in chimp_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not files:
            continue
        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * val_fraction))) if len(files) >= 2 else 0
        val_files = files[:n_val]
        train_files = files[n_val:]
        for p in train_files:
            train_samples.append(Sample(p, target, tier=2, weight=tier_weight))
        for p in val_files:
            # Validation weight is irrelevant (used for metric, not loss).
            val_samples.append(Sample(p, target, tier=2, weight=1.0))
    return train_samples, val_samples


def collect_tier3(root: Path | None, class_to_idx: dict[str, int], tier_weight: float,
                  max_per_class: int = 0, seed: int = 42) -> list[Sample]:
    """Collect tier-3 (noisy) crops, optionally capping per chimp.

    Capping limits how much any single chimp's noisy data can dominate. With
    max_per_class=0 (default) all crops are kept, but the sampler/loss weights
    will still strongly downweight tier-3.
    """
    samples: list[Sample] = []
    if root is None or not root.exists():
        return samples
    rng = random.Random(seed)
    for chimp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        canon = canonical_name(chimp_dir.name)
        if canon is None or canon not in class_to_idx:
            continue
        target = class_to_idx[canon]
        files = sorted(p for p in chimp_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if max_per_class > 0 and len(files) > max_per_class:
            rng.shuffle(files)
            files = files[:max_per_class]
        for p in files:
            samples.append(Sample(p, target, tier=3, weight=tier_weight))
    return samples


class ChimpDataset(Dataset):
    def __init__(self, samples: list[Sample], tier_transforms: dict[int, transforms.Compose]):
        self.samples = samples
        self.tier_transforms = tier_transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        n = len(self.samples)
        debug = bool(os.environ.get("DEBUG_BATCHES"))
        for offset in range(min(8, n)):
            s = self.samples[(idx + offset) % n]
            try:
                if debug:
                    print(f"[getitem] pid={os.getpid()} idx={idx} off={offset} file={s.img_path}", flush=True)
                img = Image.open(s.img_path).convert("RGB")
                tfm = self.tier_transforms[s.tier]
                x = tfm(img)
                return x, s.target, s.weight, s.tier
            except (OSError, UnidentifiedImageError):
                if debug:
                    print(f"[getitem] failed open pid={os.getpid()} file={s.img_path}", flush=True)
                continue
        raise RuntimeError(f"Could not load image near idx {idx}")


def build_transforms(image_size: int, disable_tier3_aug: bool = False) -> dict[int, transforms.Compose]:
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Tier 1: heavy
    tier1 = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=15, fill=0)], p=0.7),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.06, 0.06),
                                                       scale=(0.9, 1.1), shear=5)], p=0.5),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.25),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.5, 2.0)),
    ])

    # Tier 2: moderate
    tier2 = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=8)], p=0.4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
        transforms.ToTensor(),
        norm,
    ])

    # Tier 3: minimal (just basic geometry) — or zero augmentation when requested.
    if disable_tier3_aug:
        tier3 = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            norm,
        ])
    else:
        tier3 = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            norm,
        ])

    eval_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        norm,
    ])

    return {1: tier1, 2: tier2, 3: tier3, 0: eval_tfm}


# ============================================================================
# Sampler combining class balance and tier preference
# ============================================================================

def build_combined_sampler(samples: list[Sample], num_classes: int,
                           tier_sample_weights: dict[int, float],
                           max_resample: float = 0.0) -> WeightedRandomSampler:
    """Weight = (1 / class_count) * tier_sample_weight[tier].

    Tier sample weights here control how often a tier is *seen*, separate from
    its loss weight.

    If ``max_resample > 0``, individual sample weights are iteratively clamped
    so that no single image is drawn more than ``max_resample`` times per
    epoch in expectation. This prevents extreme oversampling of tiny classes.
    """
    class_counts = Counter(s.target for s in samples)
    weights = np.array([
        (1.0 / class_counts[s.target]) * tier_sample_weights.get(s.tier, 1.0)
        for s in samples
    ], dtype=np.float64)
    n = len(weights)
    if max_resample and max_resample > 0:
        for _ in range(50):
            expected = n * weights / weights.sum()
            over = expected > max_resample
            if not over.any():
                break
            weights[over] *= max_resample / expected[over]
    w = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=n, replacement=True)


# ============================================================================
# EMA (exponential moving average of weights)
# ============================================================================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ============================================================================
# Model
# ============================================================================

def build_chimpufe_backbone(chimpufe_src: Path, weights_path: Path):
    if str(chimpufe_src) not in sys.path:
        sys.path.append(str(chimpufe_src))
    from face_embedder.vision_transformer import vit_base  # type: ignore

    model = vit_base(
        img_size=224, patch_size=14, init_values=1e-05, ffn_layer="mlp",
        block_chunks=4, qkv_bias=True, proj_bias=True, ffn_bias=True,
        num_register_tokens=0, interpolate_offset=0.1, interpolate_antialias=False,
    )
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    teacher = ckpt["teacher"]
    teacher = {k.replace("backbone.", ""): v for k, v in teacher.items()}
    msg = model.load_state_dict(teacher, strict=False)
    print(f"Backbone loaded. Missing: {len(msg.missing_keys)}  Unexpected: {len(msg.unexpected_keys)}")
    return model


class ArcMarginProduct(nn.Module):
    """ArcFace head: cos(theta + m) margin penalty, cosine logits scaled by s."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.30):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None):
        emb = F.normalize(embeddings, dim=1)
        w = F.normalize(self.weight, dim=1)
        cosine = F.linear(emb, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return self.s * cosine
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * target_logits + (1.0 - one_hot) * cosine
        return self.s * logits


class ChimpIdentityNet(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, image_size: int = 224,
                 use_arcface: bool = False, dropout: float = 0.2,
                 arc_s: float = 30.0, arc_m: float = 0.30):
        super().__init__()
        self.backbone = backbone
        self.use_arcface = use_arcface
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat_dim = self.backbone(dummy).shape[-1]
        self.feat_dim = feat_dim
        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)
        if use_arcface:
            self.head = ArcMarginProduct(feat_dim, num_classes, s=arc_s, m=arc_m)
        else:
            self.head = nn.Linear(feat_dim, num_classes)

    def features(self, x):
        return self.norm(self.backbone(x))

    def forward(self, x, labels: torch.Tensor | None = None):
        feat = self.dropout(self.features(x))
        if self.use_arcface:
            return self.head(feat, labels)
        return self.head(feat)


# ============================================================================
# Train / Eval
# ============================================================================

def accuracy_topk(logits, targets, topk=(1,)):
    maxk = max(topk)
    bs = targets.size(0)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        out.append((correct_k / bs).item())
    return out


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr_ratio: float = 0.01) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, use_arcface: bool,
             num_classes: int):
    """Compute loss, top-1/5, per-class accuracy, macro/weighted P/R/F1, per-tier acc."""
    from sklearn.metrics import precision_recall_fscore_support

    model.eval()
    losses, top1s, top5s = [], [], []
    y_true, y_pred = [], []
    tier_correct = defaultdict(int)
    tier_total = defaultdict(int)
    crit = nn.CrossEntropyLoss()
    for images, targets, _w, tiers in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        if use_arcface:
            logits = model(images, labels=None)
        else:
            logits = model(images)
        loss = crit(logits, targets)
        top1, top5 = accuracy_topk(logits, targets, topk=(1, min(5, logits.size(1))))
        losses.append(loss.item()); top1s.append(top1); top5s.append(top5)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        # Per-tier accuracy
        correct_mask = (preds == targets).cpu().tolist()
        for t, ok in zip(tiers.tolist(), correct_mask):
            tier_total[int(t)] += 1
            tier_correct[int(t)] += int(ok)

    if y_true:
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0,
            labels=list(range(num_classes)),
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0,
            labels=list(range(num_classes)),
        )
        # Per-class accuracy (= recall when labels are present)
        per_class_correct = np.zeros(num_classes)
        per_class_total = np.zeros(num_classes)
        for t, p in zip(y_true, y_pred):
            per_class_total[t] += 1
            per_class_correct[t] += int(t == p)
        per_class_acc = (per_class_correct / np.maximum(per_class_total, 1)).tolist()
    else:
        prec_macro = rec_macro = f1_macro = 0.0
        prec_weighted = rec_weighted = f1_weighted = 0.0
        per_class_acc = [0.0] * num_classes

    per_tier_acc = {int(t): (tier_correct[t] / max(1, tier_total[t]))
                    for t in tier_total}

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "top1": float(np.mean(top1s)) if top1s else 0.0,
        "top5": float(np.mean(top5s)) if top5s else 0.0,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class_acc": per_class_acc,
        "per_tier_acc": per_tier_acc,
        "y_true": y_true, "y_pred": y_pred,
    }


def train_one_epoch(model, loader, optimizer, device, scaler, *,
                    use_arcface: bool, grad_clip: float,
                    base_lrs: list[float], warmup_steps: int, total_steps: int,
                    global_step_ref: list[int],
                    accum_steps: int = 1, log_every: int = 50,
                    ema: ModelEMA | None = None,
                    num_classes: int = 20):
    model.train()
    losses, top1s, top5s = [], [], []
    grad_norms = []
    n_samples = 0
    n_per_tier = {1: 0, 2: 0, 3: 0}
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)

    for batch_idx, (images, targets, weights, tiers) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True, dtype=torch.float32)
        n_samples += images.size(0)
        for t in tiers.tolist():
            n_per_tier[int(t)] = n_per_tier.get(int(t), 0) + 1

        # Cosine + warmup LR per param group
        step = global_step_ref[0]
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = cosine_warmup_lr(step, total_steps, warmup_steps, base_lr)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            if use_arcface:
                logits = model(images, labels=targets)
            else:
                logits = model(images)
            ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.1)
            loss = (ce * weights).mean() / accum_steps

        if not torch.isfinite(loss):
            print(f"  WARNING non-finite loss at batch {batch_idx}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if batch_idx % accum_steps == 0:
            if scaler is not None:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                else:
                    scaler.unscale_(optimizer)
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                grad_norms.append(float(gnorm))
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip and grad_clip > 0:
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                else:
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                grad_norms.append(float(gnorm))
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            global_step_ref[0] += 1

        with torch.no_grad():
            top1, top5 = accuracy_topk(
                logits.detach(), targets, topk=(1, min(5, logits.size(1)))
            )
        losses.append(loss.item() * accum_steps)
        top1s.append(top1)
        top5s.append(top5)

        do_batch_logging = (log_every is not None and log_every > 0)
        if do_batch_logging and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == n_batches):
            cur_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            ips = n_samples / max(elapsed, 1e-6)
            eta = (n_batches - batch_idx) * elapsed / max(batch_idx, 1)
            tier_share = " ".join(
                f"t{t}={n_per_tier.get(t,0)/max(n_samples,1)*100:4.1f}%"
                for t in (1, 2, 3)
            )
            gn = grad_norms[-1] if grad_norms else float("nan")
            print(
                f"    [{batch_idx:>4}/{n_batches}] "
                f"loss={losses[-1]:.4f} top1={top1:6.2f} top5={top5:6.2f} "
                f"lr={cur_lr:.2e} gn={gn:5.2f} "
                f"ips={ips:5.0f} eta={eta:5.0f}s | {tier_share}",
                flush=True,
            )

    secs = time.time() - t0
    lrs = [pg["lr"] for pg in optimizer.param_groups]
    gpu_mem_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
                  if torch.cuda.is_available() else 0.0)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "top1": float(np.mean(top1s)) if top1s else 0.0,
        "top5": float(np.mean(top5s)) if top5s else 0.0,
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
        "secs": secs,
        "samples_per_sec": n_samples / max(secs, 1e-6),
        "lr_group0": lrs[0] if lrs else 0.0,
        "lr_group1": lrs[1] if len(lrs) > 1 else 0.0,
        "gpu_mem_peak_mb": gpu_mem_mb,
        "n_seen_tier1": n_per_tier.get(1, 0),
        "n_seen_tier2": n_per_tier.get(2, 0),
        "n_seen_tier3": n_per_tier.get(3, 0),
    }


@torch.no_grad()
def build_class_prototypes(model, loader, device, num_classes):
    model.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.long)
    for images, targets, _w, _t in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        feats = F.normalize(model.features(images), dim=1)
        if sums is None:
            sums = torch.zeros(num_classes, feats.shape[1], device=device)
        sums.index_add_(0, targets, feats)
        counts.index_add_(0, targets.cpu(), torch.ones_like(targets.cpu(), dtype=torch.long))
    counts = counts.to(device=device, dtype=sums.dtype).clamp_min(1.0).unsqueeze(1)
    return F.normalize(sums / counts, dim=1)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db-root", type=str, default=None)
    p.add_argument("--tier3-root", type=str, default=None,
                   help="Folder containing per-chimp subdirs of noisy video crops.")
    p.add_argument("--out-dir", type=str, default=None)

    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--epochs-stage1", type=int, default=6, help="Linear-probe (frozen backbone)")
    p.add_argument("--epochs-stage2", type=int, default=80, help="Full fine-tune")
    p.add_argument("--warmup-epochs", type=float, default=2.0)

    p.add_argument("--lr-stage1", type=float, default=2e-3)
    p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Per-tier loss weights (used inside loss)
    p.add_argument("--tier1-loss-weight", type=float, default=1.0)
    p.add_argument("--tier2-loss-weight", type=float, default=0.8)
    p.add_argument("--tier3-loss-weight", type=float, default=0.3)
    # Per-tier sample-frequency boost (used by sampler)
    p.add_argument("--tier1-sample-mult", type=float, default=5.0)
    p.add_argument("--tier2-sample-mult", type=float, default=3.0)
    p.add_argument("--tier3-sample-mult", type=float, default=1.0)
    # Tier-3 cap to prevent noisy data from dominating any single class
    p.add_argument("--tier3-max-per-class", type=int, default=0,
                   help="Cap noisy video crops per chimp (0 = unlimited).")
    # Per-image oversampling cap (caps small classes' replication factor)
    p.add_argument("--max-resample", type=float, default=0.0,
                   help="Max times any single image is drawn per epoch in "
                        "expectation (0 = unlimited). Ex: 3.0 means a tiny "
                        "class will not be replicated more than 3x per image.")
    # Logging frequency
    p.add_argument("--log-every", type=int, default=10,
                   help="Print a per-batch line every N batches (and at batch 1 and last).")
    # Tier-2 student split
    p.add_argument("--tier2-val-fraction", type=float, default=0.15,
                   help="Fraction of tier-2 (student) data held out for validation.")
    # Disable augmentation on tier-3 (already very redundant from video frames)
    p.add_argument("--no-tier3-augmentation", action="store_true", default=False,
                   help="Disable random augmentations on tier-3 crops; only resize+normalize.")

    # Explicit dataset roots (override defaults under --db-root)
    p.add_argument("--train-root", type=str, default=None,
                   help="Tier-1 train folder. Default: <db-root>/face_crops_chimpRec/train")
    p.add_argument("--val-root", type=str, default=None,
                   help="Tier-1 val folder. Default: <db-root>/face_crops_chimpRec/val")
    p.add_argument("--student-root", type=str, default=None,
                   help="Tier-2 student folder. Default: <db-root>/student_face_crops")

    p.add_argument("--use-arcface", action="store_true", default=False)
    p.add_argument("--arc-s", type=float, default=30.0)
    p.add_argument("--arc-m", type=float, default=0.30)

    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--ema-decay", type=float, default=0.9995)

    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--pretrained-weights", type=str, default=None)
    p.add_argument("--chimpufe-src", type=str, default=None)

    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--compile", action="store_true", default=False,
                   help="Use torch.compile() (PyTorch 2.x).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        print("GPU:", torch.cuda.get_device_name(0))

    seed_everything(args.seed)

    script_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(script_dir)
    db_root = Path(args.db_root).resolve() if args.db_root else (repo_root / "ChimpPic" / "face_recognition_db")
    tier3_root = Path(args.tier3_root).resolve() if args.tier3_root else None
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (script_dir / "runs" / "chimpufe_intensive")
    out_dir.mkdir(parents=True, exist_ok=True)

    chimpufe_src = (Path(args.chimpufe_src).resolve() if args.chimpufe_src
                    else repo_root / "Code" / "Tracking" / "ChimpUFE" / "src")
    pretrained_weights = (Path(args.pretrained_weights).resolve() if args.pretrained_weights
                          else repo_root / "Code" / "Tracking" / "ChimpUFE" / "assets" / "weights" / "25-08-29T11-49-28_340k.pth")

    train_root   = Path(args.train_root).resolve()   if args.train_root   else (db_root / "face_crops_chimpRec" / "train")
    val_root     = Path(args.val_root).resolve()     if args.val_root     else (db_root / "face_crops_chimpRec" / "val")
    test_root    = db_root / "test"
    student_root = Path(args.student_root).resolve() if args.student_root else (db_root / "student_face_crops")

    print("repo_root :", repo_root)
    print("db_root   :", db_root)
    print("tier3_root:", tier3_root)
    print("out_dir   :", out_dir)
    print("chimpufe  :", chimpufe_src)
    print("weights   :", pretrained_weights)

    assert train_root.exists() and val_root.exists(), "train/ and val/ required"
    assert chimpufe_src.exists(), f"ChimpUFE src missing: {chimpufe_src}"
    assert pretrained_weights.exists(), f"Pretrained weights missing: {pretrained_weights}"

    class_to_idx = {n: i for i, n in enumerate(CANONICAL_NAMES)}
    idx_to_class = {i: n for n, i in class_to_idx.items()}

    tfms = build_transforms(args.image_size, disable_tier3_aug=args.no_tier3_augmentation)

    train_samples_t1 = collect_tier1(train_root, class_to_idx, args.tier1_loss_weight)
    val_samples_t1 = collect_tier1(val_root, class_to_idx, 1.0)
    train_samples_t2, val_samples_t2 = collect_tier2_split(
        student_root, class_to_idx, args.tier2_loss_weight,
        val_fraction=args.tier2_val_fraction, seed=args.seed,
    )
    train_samples_t3 = collect_tier3(
        tier3_root, class_to_idx, args.tier3_loss_weight,
        max_per_class=args.tier3_max_per_class, seed=args.seed,
    )

    train_samples = train_samples_t1 + train_samples_t2 + train_samples_t3
    val_samples = val_samples_t1 + val_samples_t2

    print(f"\nDataset sizes:")
    print(f"  TRAIN tier1 (HQ)        : {len(train_samples_t1):>6}")
    print(f"  TRAIN tier2 (student)   : {len(train_samples_t2):>6}")
    print(f"  TRAIN tier3 (video)     : {len(train_samples_t3):>6}"
          f"  (cap/class={args.tier3_max_per_class})")
    print(f"  TRAIN TOTAL             : {len(train_samples):>6}")
    print(f"  VAL   tier1 (HQ)        : {len(val_samples_t1):>6}")
    print(f"  VAL   tier2 (student)   : {len(val_samples_t2):>6}")
    print(f"  VAL   TOTAL             : {len(val_samples):>6}")
    print(f"  TEST (HELD OUT, never seen during training): "
          f"{len(list(test_root.iterdir())) if test_root.exists() else 0}")

    # Safety check: ensure no test image leaks into train/val
    test_files = set()
    if test_root.exists():
        test_files = {p.name for p in test_root.iterdir() if p.suffix.lower() in IMG_EXTS}
    leaked = [s.img_path.name for s in (train_samples + val_samples)
              if s.img_path.name in test_files]
    if leaked:
        raise RuntimeError(f"DATA LEAK: {len(leaked)} test filenames found in train/val: "
                           f"{leaked[:5]}")
    print("Data-leakage check OK (test/ is fully held out).")

    # ---- Save run config + dataset manifest for reproducibility / thesis ----
    run_config = {
        "args": vars(args),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "seed": args.seed,
        "class_to_idx": class_to_idx,
        "tier_loss_weights": {1: args.tier1_loss_weight, 2: args.tier2_loss_weight, 3: args.tier3_loss_weight},
        "tier_sample_mults": {1: args.tier1_sample_mult, 2: args.tier2_sample_mult, 3: args.tier3_sample_mult},
        "tier3_max_per_class": args.tier3_max_per_class,
        "max_resample": args.max_resample,
        "tier2_val_fraction": args.tier2_val_fraction,
        "dataset_sizes": {
            "train_tier1": len(train_samples_t1),
            "train_tier2": len(train_samples_t2),
            "train_tier3": len(train_samples_t3),
            "train_total": len(train_samples),
            "val_tier1": len(val_samples_t1),
            "val_tier2": len(val_samples_t2),
            "val_total": len(val_samples),
            "test_held_out": len(list(test_root.iterdir())) if test_root.exists() else 0,
        },
        "per_class_train_counts": {
            idx_to_class[i]: {
                "tier1": sum(1 for s in train_samples_t1 if s.target == i),
                "tier2": sum(1 for s in train_samples_t2 if s.target == i),
                "tier3": sum(1 for s in train_samples_t3 if s.target == i),
            } for i in range(len(class_to_idx))
        },
        "per_class_val_counts": {
            idx_to_class[i]: {
                "tier1": sum(1 for s in val_samples_t1 if s.target == i),
                "tier2": sum(1 for s in val_samples_t2 if s.target == i),
            } for i in range(len(class_to_idx))
        },
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"Saved run config: {out_dir / 'run_config.json'}")

    # Per-sample manifest (to know exactly which file went where)
    manifest_rows = []
    for s in train_samples:
        manifest_rows.append({"split": "train", "tier": s.tier,
                              "chimp": idx_to_class[s.target],
                              "loss_weight": s.weight,
                              "img_path": str(s.img_path)})
    for s in val_samples:
        manifest_rows.append({"split": "val", "tier": s.tier,
                              "chimp": idx_to_class[s.target],
                              "loss_weight": s.weight,
                              "img_path": str(s.img_path)})
    pd.DataFrame(manifest_rows).to_csv(out_dir / "dataset_manifest.csv", index=False)

    print("\nPer-class counts (train):")
    counts_per_class = defaultdict(lambda: [0, 0, 0])  # [t1,t2,t3]
    for s in train_samples:
        counts_per_class[s.target][s.tier - 1] += 1
    for i in range(len(class_to_idx)):
        c = counts_per_class[i]
        print(f"  {idx_to_class[i]:12s}  t1:{c[0]:>4}  t2:{c[1]:>4}  t3:{c[2]:>4}  total:{sum(c):>5}")

    train_ds = ChimpDataset(train_samples, tfms)

    # Optional debug hook for DataLoader workers. Set DEBUG_WORKERS=1 in the
    # environment to see worker start messages in the job stdout.
    def _worker_init_fn(worker_id):
        if os.environ.get("DEBUG_WORKERS"):
            print(f"[worker_init] worker_id={worker_id} pid={os.getpid()} cwd={os.getcwd()}", flush=True)

    # Validation always uses the eval transform (tier 0). Mark all as tier 0.
    val_ds = ChimpDataset(
        [Sample(s.img_path, s.target, 0, 1.0) for s in val_samples], tfms,
    )

    sampler = build_combined_sampler(
        train_samples, len(class_to_idx),
        tier_sample_weights={
            1: args.tier1_sample_mult,
            2: args.tier2_sample_mult,
            3: args.tier3_sample_mult,
        },
        max_resample=args.max_resample,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=False,
        persistent_workers=False, drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
        persistent_workers=False,
        worker_init_fn=_worker_init_fn,
    )

    backbone = build_chimpufe_backbone(chimpufe_src, pretrained_weights)
    model = ChimpIdentityNet(
        backbone, num_classes=len(class_to_idx), image_size=args.image_size,
        use_arcface=args.use_arcface, arc_s=args.arc_s, arc_m=args.arc_m,
    ).to(device, memory_format=torch.channels_last)

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = []
    best_val_top1 = -1.0
    best_state = None
    patience_counter = 0
    global_step_ref = [0]

    def save_ckpt(path: Path, mdl, epoch, stage, val_top1):
        payload = {
            "epoch": epoch, "stage": stage, "val_top1": val_top1,
            "model_state": mdl.state_dict(),
            "ema_state": ema.module.state_dict() if ema is not None else None,
            "class_to_idx": class_to_idx,
            "config": vars(args),
        }
        torch.save(payload, path)

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if ema is not None and ck.get("ema_state") is not None:
            ema.module.load_state_dict(ck["ema_state"])
        best_val_top1 = ck.get("val_top1", -1.0)
        print(f"Resumed from {args.resume} (val_top1={best_val_top1:.4f})")

    # ---- Stage 1: linear probe ----
    def set_backbone_trainable(flag: bool):
        for p in model.backbone.parameters():
            p.requires_grad_(flag)

    def run_stage(name: str, epochs: int, optimizer, base_lrs, warmup_steps, total_steps):
        nonlocal best_val_top1, best_state, patience_counter
        print(f"\n=== {name}  ({epochs} epochs) ===")
        for epoch in range(1, epochs + 1):
            print(f"\n[{name}] Epoch {epoch}/{epochs}")
            stats = train_one_epoch(
                model, train_loader, optimizer, device, scaler,
                use_arcface=args.use_arcface, grad_clip=args.grad_clip,
                base_lrs=base_lrs, warmup_steps=warmup_steps, total_steps=total_steps,
                global_step_ref=global_step_ref, accum_steps=args.accum_steps,
                ema=ema, num_classes=len(class_to_idx), log_every=args.log_every,
            )
            eval_target = ema.module if ema is not None else model
            val = evaluate(eval_target, val_loader, device, args.use_arcface,
                           num_classes=len(class_to_idx))

            row = {
                "stage": name, "epoch": epoch,
                "global_step": global_step_ref[0],
                "train_loss": stats["loss"],
                "train_top1": stats["top1"],
                "train_top5": stats["top5"],
                "grad_norm_mean": stats["grad_norm_mean"],
                "grad_norm_max": stats["grad_norm_max"],
                "lr_group0": stats["lr_group0"],
                "lr_group1": stats["lr_group1"],
                "samples_per_sec": stats["samples_per_sec"],
                "gpu_mem_peak_mb": stats["gpu_mem_peak_mb"],
                "val_loss": val["loss"],
                "val_top1": val["top1"],
                "val_top5": val["top5"],
                "val_precision_macro": val["precision_macro"],
                "val_recall_macro": val["recall_macro"],
                "val_f1_macro": val["f1_macro"],
                "val_precision_weighted": val["precision_weighted"],
                "val_recall_weighted": val["recall_weighted"],
                "val_f1_weighted": val["f1_weighted"],
                "val_acc_tier1": val["per_tier_acc"].get(1, float("nan")),
                "val_acc_tier2": val["per_tier_acc"].get(2, float("nan")),
                "n_seen_tier1": stats.get("n_seen_tier1", 0),
                "n_seen_tier2": stats.get("n_seen_tier2", 0),
                "n_seen_tier3": stats.get("n_seen_tier3", 0),
                "secs": stats["secs"],
                "wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            # Per-class accuracies as separate columns for easy plotting
            for ci, cname in idx_to_class.items():
                row[f"val_acc__{cname}"] = val["per_class_acc"][ci]
            history.append(row)

            n_seen = stats.get("n_seen_tier1", 0) + stats.get("n_seen_tier2", 0) + stats.get("n_seen_tier3", 0)
            tier_pct = (
                f"t1 {stats.get('n_seen_tier1',0)/max(n_seen,1)*100:4.1f}% "
                f"t2 {stats.get('n_seen_tier2',0)/max(n_seen,1)*100:4.1f}% "
                f"t3 {stats.get('n_seen_tier3',0)/max(n_seen,1)*100:4.1f}%"
            )
            print(f"\n[{name}] EPOCH {epoch:02d}/{epochs} SUMMARY", flush=True)
            print(
                f"  train  : loss {stats['loss']:.4f}  top1 {stats['top1']*100:6.2f}%  top5 {stats['top5']*100:6.2f}%"
                f"  | grad mean {stats['grad_norm_mean']:.2f} max {stats['grad_norm_max']:.2f}",
                flush=True,
            )
            print(
                f"  val    : loss {val['loss']:.4f}  top1 {val['top1']*100:6.2f}%  top5 {val['top5']*100:6.2f}%"
                f"  | F1-macro {val['f1_macro']:.4f}  prec {val['precision_macro']:.4f}  rec {val['recall_macro']:.4f}",
                flush=True,
            )
            print(
                f"  per-tier acc : t1 {val['per_tier_acc'].get(1, 0)*100:5.2f}%  t2 {val['per_tier_acc'].get(2, 0)*100:5.2f}%",
                flush=True,
            )
            print(
                f"  speed  : {stats['samples_per_sec']:.0f} im/s   epoch {stats['secs']:.0f}s   gpu peak {stats['gpu_mem_peak_mb']:.0f}MB",
                flush=True,
            )
            print(f"  lr     : g0={stats['lr_group0']:.2e}  g1={stats['lr_group1']:.2e}", flush=True)
            print(f"  tiers  : seen {n_seen} samples ({tier_pct})", flush=True)

            # Update best and save if improved
            if val['top1'] > best_val_top1:
                best_val_top1 = val['top1']
                best_state = eval_target.state_dict()
                save_ckpt(out_dir / "best_model.pt", eval_target, epoch, name, val["top1"])
                print(f"  -> NEW BEST val_top1={val['top1']:.4f}", flush=True)

            # Save BOTH json (rich) and csv (easy plotting) every epoch
            with open(out_dir / "train_history.json", "w") as f:
                json.dump(history, f, indent=2)
            pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

            if val["top1"] > best_val_top1 + 1e-4:
                best_val_top1 = val["top1"]
                best_state = copy.deepcopy(eval_target.state_dict())
                patience_counter = 0
                save_ckpt(out_dir / "best_model.pt", eval_target, epoch, name, best_val_top1)
                copy2(out_dir / "best_model.pt", out_dir / "best_model.pth")
                print(f"  -> NEW BEST val_top1={best_val_top1:.4f}")
            else:
                patience_counter += 1
                print(f"  -> no improvement ({patience_counter}/{args.early_stopping_patience})")

            # Always save last
            save_ckpt(out_dir / "last_model.pt", eval_target, epoch, name, val["top1"])

            if patience_counter >= args.early_stopping_patience:
                print("Early stopping.")
                return

    steps_per_epoch = max(1, len(train_loader) // max(1, args.accum_steps))
    warmup_steps_total = int(args.warmup_epochs * steps_per_epoch)

    # Stage 1
    if args.epochs_stage1 > 0:
        set_backbone_trainable(False)
        opt1 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr_stage1, weight_decay=args.weight_decay,
        )
        total1 = args.epochs_stage1 * steps_per_epoch
        run_stage("stage1_linear_probe", args.epochs_stage1, opt1,
                  base_lrs=[args.lr_stage1], warmup_steps=warmup_steps_total, total_steps=total1)

    # Stage 2
    if args.epochs_stage2 > 0:
        set_backbone_trainable(True)
        head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
        opt2 = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr_backbone},
                {"params": head_params, "lr": args.lr_head},
            ],
            weight_decay=args.weight_decay,
        )
        total2 = args.epochs_stage2 * steps_per_epoch
        global_step_ref[0] = 0  # reset schedule for stage 2
        run_stage("stage2_full_finetune", args.epochs_stage2, opt2,
                  base_lrs=[args.lr_backbone, args.lr_head],
                  warmup_steps=warmup_steps_total, total_steps=total2)

    # ---- Final eval and reports ----
    eval_target = ema.module if ema is not None else model
    if best_state is not None:
        eval_target.load_state_dict(best_state)
        print("Loaded best checkpoint for final report.")

    val = evaluate(eval_target, val_loader, device, args.use_arcface,
                   num_classes=len(class_to_idx))
    print(f"\nFinal VAL: loss {val['loss']:.4f}  top1 {val['top1']:.4f}  top5 {val['top5']:.4f}"
          f"  f1_macro {val['f1_macro']:.4f}")

    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(val["y_true"], val["y_pred"],
                                   target_names=target_names, digits=3, zero_division=0)
    print("\nVAL classification report:\n" + report)
    (out_dir / "val_classification_report.txt").write_text(report)

    # Also save report as a structured CSV (per-class precision/recall/f1/support)
    from sklearn.metrics import classification_report as _cr
    rep_dict = _cr(val["y_true"], val["y_pred"], target_names=target_names,
                   digits=3, zero_division=0, output_dict=True)
    pd.DataFrame(rep_dict).T.to_csv(out_dir / "val_classification_report.csv")

    cm = confusion_matrix(val["y_true"], val["y_pred"], labels=list(range(len(idx_to_class))))
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(out_dir / "val_confusion_matrix.csv")

    # Final per-sample VAL predictions (for any post-hoc analysis you want in the thesis)
    val_pred_rows = []
    for s, yt, yp in zip(val_ds.samples, val["y_true"], val["y_pred"]):
        val_pred_rows.append({
            "img_path": str(s.img_path),
            "file_name": s.img_path.name,
            "ground_truth": idx_to_class[yt],
            "pred": idx_to_class[yp],
            "correct": int(yt == yp),
        })
    pd.DataFrame(val_pred_rows).to_csv(out_dir / "val_predictions.csv", index=False)

    # Final summary JSON for thesis tables
    final_summary = {
        "best_val_top1": best_val_top1,
        "final_val_top1": val["top1"],
        "final_val_top5": val["top5"],
        "final_val_loss": val["loss"],
        "final_val_f1_macro": val["f1_macro"],
        "final_val_f1_weighted": val["f1_weighted"],
        "final_val_precision_macro": val["precision_macro"],
        "final_val_recall_macro": val["recall_macro"],
        "final_val_per_tier_acc": val["per_tier_acc"],
        "final_val_per_class_acc": {idx_to_class[i]: v
                                     for i, v in enumerate(val["per_class_acc"])},
        "num_classes": len(class_to_idx),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
    }
    with open(out_dir / "final_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2, default=float)

    # ---- Test inference using prototypes from tier-1 train only ----
    if test_root.exists():
        train_eval_ds = ChimpDataset(
            [Sample(s.img_path, s.target, 0, 1.0) for s in train_samples_t1], tfms,
        )
        train_eval_loader = DataLoader(
            train_eval_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        prototypes = build_class_prototypes(eval_target, train_eval_loader, device, len(class_to_idx))
        print("Built tier-1 prototypes for test inference.")

        rows = []
        eval_tfm = tfms[0]
        eval_target.eval()
        with torch.no_grad():
            for p in sorted(test_root.iterdir()):
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                try:
                    img = Image.open(p).convert("RGB")
                except (OSError, UnidentifiedImageError):
                    continue
                x = eval_tfm(img).unsqueeze(0).to(device, memory_format=torch.channels_last)
                feat = F.normalize(eval_target.features(x), dim=1)
                sims = (feat @ prototypes.t())[0]
                top = torch.topk(sims, k=min(3, sims.numel()))
                pred_idx = int(top.indices[0])
                pred = idx_to_class[pred_idx]
                gt = name_from_test_filename(p.name)
                row = {
                    "image": p.name, "ground_truth": gt or "",
                    "pred": pred, "correct": int(gt == pred) if gt else "",
                    "sim_top1": float(top.values[0]),
                    "sim_top2": float(top.values[1]) if top.values.numel() > 1 else float("nan"),
                    "margin": float(top.values[0] - top.values[1]) if top.values.numel() > 1 else float("nan"),
                    "pred_top2": idx_to_class[int(top.indices[1])] if top.indices.numel() > 1 else "",
                    "pred_top3": idx_to_class[int(top.indices[2])] if top.indices.numel() > 2 else "",
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "test_predictions.csv", index=False)
        if len(df):
            known = df[df["ground_truth"] != ""]
            if len(known):
                acc = float(known["correct"].astype(int).mean())
                print(f"\nTest accuracy (resolvable filenames, {len(known)}/{len(df)}): {acc:.4f}")
        print(f"Saved test predictions: {out_dir / 'test_predictions.csv'}")

    print(f"\nDone. Outputs in: {out_dir}")

    # ---- Auto-generate progress plots ----
    try:
        import subprocess, sys as _sys
        plot_script = script_dir / "make_thesis_plots.py"
        if plot_script.exists():
            print(f"\nGenerating progress plots via {plot_script.name} ...")
            subprocess.run(
                [_sys.executable, str(plot_script), "--run-dir", str(out_dir)],
                check=False,
            )
            print(f"Plots saved to: {out_dir / 'figures'}")
    except Exception as e:
        print(f"Plot generation failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
