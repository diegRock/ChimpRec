import torch
import sys
import os

# Import local model
sys.path.append("ChimpUFE")
from src.face_embedder.vision_transformer import vit_base

print("--- MODEL STRUCTURE ---")
model = vit_base(patch_size=14)
# Print just the top-level keys of the state dict to see nesting
keys = sorted(list(model.state_dict().keys()))
# Group by first 2 parts to see structure
groups = set()
for k in keys:
    parts = k.split(".")
    if len(parts) > 2:
        groups.add(f"{parts[0]}.{parts[1]}")
    else:
        groups.add(k)

for g in sorted(list(groups)):
    print(f"  {g}")

print("\n--- CHECKPOINT STRUCTURE ---")
weights_path = "assets/weights/25-08-29T11-49-28_340k.pth"
if not os.path.exists(weights_path): weights_path = "ChimpUFE/" + weights_path

checkpoint = torch.load(weights_path, map_location="cpu")
if "teacher" in checkpoint: state_dict = checkpoint["teacher"]
elif "model" in checkpoint: state_dict = checkpoint["model"]
else: state_dict = checkpoint

# Group checkpoint keys
groups_ckpt = set()
for k in state_dict.keys():
    k = k.replace("backbone.", "").replace("module.", "")
    parts = k.split(".")
    if len(parts) > 2:
        groups_ckpt.add(f"{parts[0]}.{parts[1]}")
    else:
        groups_ckpt.add(k)

for g in sorted(list(groups_ckpt)):
    print(f"  {g}")