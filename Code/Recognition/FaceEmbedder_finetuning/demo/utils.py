"""
Helper module for the chimpanzee identification demo notebook.

Keeps the notebook short and readable by hosting all the heavy lifting:
- dependency check
- model definition + checkpoint loading
- gallery embedding + prototype building
- identify_chimp() with optional side-by-side plotting
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (defaults — can be overridden by the notebook)
# ---------------------------------------------------------------------------
IMAGE_SIZE   = 224
TOP_K        = 5
TEMPERATURE  = 10.0
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

INITIAL_TO_NAME = {
    "AD": "Amadi",    "BL": "Banalia",  "BS": "Binasera",  "DK": "Djiku",
    "IV": "Ivan",     "JJ": "Jeje",     "KM": "Kalemi",    "KG": "Kassongo",
    "KR": "Kira",     "LM": "Lwama",    "MG": "Malago",    "MM": "Maniema",
    "MZ": "Mazingira","MK": "Muke",     "NJ": "Nganja",    "NR": "Nzuri",
    "PD": "Penda",    "TS": "Talisa",   "TC": "Tanganica", "TT": "Tingitingi",
}

_REQUIRED = {
    "torch":       "torch",
    "torchvision": "torchvision",
    "PIL":         "pillow",
    "numpy":       "numpy",
    "pandas":      "pandas",
    "matplotlib":  "matplotlib",
}


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def ensure_dependencies():
    """Install any missing required packages (safe to re-run)."""
    missing = [pkg for mod, pkg in _REQUIRED.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", *missing])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _build_backbone():
    from face_embedder.vision_transformer import vit_base
    return vit_base(
        img_size=224, patch_size=14, init_values=1e-05, ffn_layer="mlp",
        block_chunks=4, qkv_bias=True, proj_bias=True, ffn_bias=True,
        num_register_tokens=0, interpolate_offset=0.1,
        interpolate_antialias=False,
    )


def _build_model_classes():
    """Build ArcMarginProduct / ChimpIdentityNet lazily so torch is imported
    only after ensure_dependencies() ran."""
    import torch
    import torch.nn as nn

    class ArcMarginProduct(nn.Module):
        """Same head shape as during training (used only for state-dict loading)."""
        def __init__(self, in_features, out_features, s=30.0, m=0.30):
            super().__init__()
            self.s, self.m = s, m
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.xavier_normal_(self.weight)

    class ChimpIdentityNet(nn.Module):
        def __init__(self, backbone, num_classes, use_arcface, dropout=0.2):
            super().__init__()
            self.backbone = backbone
            with torch.no_grad():
                feat_dim = self.backbone(
                    torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
                ).shape[-1]
            self.feat_dim = feat_dim
            self.norm = nn.LayerNorm(feat_dim)
            self.dropout = nn.Dropout(dropout)
            self.head = (ArcMarginProduct(feat_dim, num_classes)
                         if use_arcface else nn.Linear(feat_dim, num_classes))

        def features(self, x):
            return self.norm(self.backbone(x))

    return ArcMarginProduct, ChimpIdentityNet


def load_model(model_path: Path, device):
    """Load checkpoint and return (model, eval_tfm, class_to_idx, cfg)."""
    import torch
    from torchvision import transforms

    _, ChimpIdentityNet = _build_model_classes()

    print("Loading checkpoint ...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    class_to_idx = ckpt.get("class_to_idx", {})
    cfg = ckpt.get("config", {}) or {}
    use_arcface = bool(cfg.get("use_arcface", True))
    num_classes = max(len(class_to_idx), 1)

    backbone = _build_backbone()
    model = ChimpIdentityNet(backbone, num_classes=num_classes,
                             use_arcface=use_arcface).to(device)
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    print(f"Loaded. classes={num_classes}  arcface={use_arcface}  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")

    eval_tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, eval_tfm, class_to_idx, cfg


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------
def list_images(folder: Path):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def embed_image(img, model, eval_tfm, device):
    """Return the L2-normalised embedding of a PIL image."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        x = eval_tfm(img).unsqueeze(0).to(device)
        feat = F.normalize(model.features(x), dim=1).cpu().numpy()[0]
    return feat


def build_gallery(gallery_dir: Path, model, eval_tfm, device):
    """Embed every image in `gallery_dir/<CODE>/` and build per-chimp prototypes.

    Returns a dict with keys: paths, labels, embs, class_codes, proto_index,
    prototypes.
    """
    import numpy as np
    from PIL import Image

    paths, labels, embs = [], [], []
    for chimp_dir in sorted(p for p in gallery_dir.iterdir() if p.is_dir()):
        code = chimp_dir.name.upper()
        files = list_images(chimp_dir)
        if not files:
            print(f"  (skipping empty folder: {chimp_dir.name})")
            continue
        for p in files:
            try:
                emb = embed_image(Image.open(p).convert("RGB"),
                                  model, eval_tfm, device)
            except Exception as e:
                print(f"  ! could not read {p.name}: {e}")
                continue
            paths.append(p)
            labels.append(code)
            embs.append(emb)
        print(f"  {code} ({INITIAL_TO_NAME.get(code, '?'):<11}) "
              f"-> {sum(1 for l in labels if l == code)} images")

    embs = np.stack(embs)
    labels = np.array(labels)
    class_codes = sorted(set(labels))
    proto_index = {c: np.where(labels == c)[0] for c in class_codes}
    prototypes = np.stack([embs[proto_index[c]].mean(0) for c in class_codes])
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12

    print(f"\nGallery ready: {len(embs)} images across "
          f"{len(class_codes)} chimps.")
    return {
        "paths":        paths,
        "labels":       labels,
        "embs":         embs,
        "class_codes":  class_codes,
        "proto_index":  proto_index,
        "prototypes":   prototypes,
    }


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------
def identify_chimp(image_path, gallery, model, eval_tfm, device,
                   top_k: int = TOP_K, temperature: float = TEMPERATURE,
                   show: bool = True):
    """Identify the chimp in `image_path`. Returns a pandas DataFrame of
    top-k candidates and optionally shows a side-by-side plot."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image

    image_path = Path(image_path)
    assert image_path.exists(), f"File not found: {image_path}"

    prototypes   = gallery["prototypes"]
    class_codes  = gallery["class_codes"]
    proto_index  = gallery["proto_index"]
    gallery_embs = gallery["embs"]
    gallery_paths = gallery["paths"]

    img = Image.open(image_path).convert("RGB")
    feat = embed_image(img, model, eval_tfm, device)

    sims = prototypes @ feat
    s = sims * temperature
    probs = np.exp(s - s.max())
    probs /= probs.sum()
    order = np.argsort(-sims)[:top_k]

    df = pd.DataFrame({
        "rank":       np.arange(1, len(order) + 1),
        "code":       [class_codes[i] for i in order],
        "name":       [INITIAL_TO_NAME.get(class_codes[i], "?") for i in order],
        "cosine_sim": sims[order],
        "confidence": probs[order],
    })

    if show:
        thumbs = []
        for code in df["code"]:
            idxs = proto_index[code]
            class_sims = gallery_embs[idxs] @ feat
            best = idxs[int(np.argmax(class_sims))]
            thumbs.append((code, gallery_paths[best], float(class_sims.max())))

        n = len(thumbs) + 1
        fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2))
        axes[0].imshow(img)
        axes[0].set_title(f"QUERY\n{image_path.name}", fontsize=9)
        axes[0].axis("off")
        for ax, (code, path, sim), conf in zip(axes[1:], thumbs, df["confidence"]):
            try:
                ax.imshow(Image.open(path).convert("RGB"))
            except Exception:
                ax.text(0.5, 0.5, "?", ha="center", va="center")
            name = INITIAL_TO_NAME.get(code, "?")
            ax.set_title(f"{name} ({code})\nsim={sim:.2f}  conf={conf*100:.1f}%",
                         fontsize=9)
            ax.axis("off")
        top_name = INITIAL_TO_NAME.get(df["code"].iloc[0], "?")
        plt.suptitle(f"Predicted: {top_name} ({df['code'].iloc[0]})  "
                     f"— confidence {df['confidence'].iloc[0]*100:.1f}%",
                     fontsize=11, y=1.02)
        plt.tight_layout()
        plt.show()
    return df
