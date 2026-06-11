# FaceEmbedder Finetuning

This folder contains the FaceEmbedder finetuning project for chimp recognition.

The goal is to build a complete pipeline that:

1. extracts quality chimp face crops from video and tracking annotations,
2. trains or finetunes a face embedding model on those crops,
3. evaluates the trained model on held-out data.

This README gives a high-level overview of the module. Detailed guides for each subcomponent are available in the subfolders:

- `data_engine/` — data preparation and face crop extraction
- `finetuning/` — training and finetuning the face embedding model
- `Evaluation/` — performance measurement and evaluation scripts
- `demo/` — example usage and quick tests

---

## Project structure

```
Code/Recognition/FaceEmbedder_finetuning/
  README.md
  data_engine/
  finetuning/
  Evaluation/
  demo/
```

### What each part does

- `data_engine/` prepares the dataset by extracting chimp face crops from video frames and named track annotations.
- `finetuning/` trains the face embedding model on the extracted crops and supports model checkpointing.
- `Evaluation/` measures model quality using verification or identification metrics and generates reports.
- `demo/` contains minimal example commands or scripts to verify the pipeline works.

---

## Why this module exists

The FaceEmbedder finetuning module is designed to support iterative development of chimp face recognition:

- build a reliable dataset from noisy video and tracking annotations,
- produce a fine-grained face embedding model for chimp identity recognition,
- evaluate model improvements across training runs.

This module sits inside the larger ChimpRec repository and focuses specifically on the face recognition branch of the project.

---

## Prerequisites

This code is tested with Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r Code/requirements.txt
```

The project also depends on the external `ChimpUFE` package. Clone it into `Code/chimplib`:

```bash
cd Code/chimplib
git clone https://github.com/v-iashin/ChimpUFE.git
```

---

## High-level workflow

### 1) Prepare data

Use `data_engine/` to extract face crops from raw video and tracking annotations. The extractor creates one output folder per chimp identity and writes CSV logs for accepted and rejected crops.

### 2) Train or finetune the model

Use `finetuning/` to train a face embedding model on the extracted crops. This step typically consumes the structured face dataset produced by `data_engine/`.

### 3) Evaluate results

Use `Evaluation/` to compute recognition metrics, inspect failures, and compare different finetuning runs.

---

## Recommended directory layout

A clean dataset layout helps avoid confusion:

```
data/
  raw/
    videos/
    tracks/
output/
  extracted_faces/
  models/
  evaluation/
```

- `data/raw/videos/` contains source recording files.
- `data/raw/tracks/` contains named track annotation files.
- `output/extracted_faces/` contains per-chimp face crops.
- `output/models/` holds trained checkpoints.
- `output/evaluation/` stores evaluation reports.

---

## Main entry points

The exact command syntax is implemented inside each subfolder. At a high level, the workflow is:

- `data_engine/` – extract and filter face crops from video + tracks
- `finetuning/` – run training / finetuning with the prepared dataset
- `Evaluation/` – run evaluation scripts on saved checkpoints

This README is intentionally high-level. Use the next README files inside `data_engine/`, `finetuning/`, and `Evaluation/` for step-by-step instructions.

---

## Notes

- Keep raw data separate from generated outputs.
- Use canonical chimp identity labels consistently across data preparation and training.
- Track files should match video names so pairing is automatic.
- Store extracted crops in per-identity folders for easy dataset loading.
- Save model checkpoints and evaluation results with descriptive names.

## Reproducibility — dataset organization

To reproduce dataset preparation steps and locate inputs/outputs, follow the conventions implied by the repository folder names (paths are relative to the repository root).

- `ChimpVideos/` — workspace for session data and processing:
  - `ChimpVideos/input/` — place raw video files, track files, and session manifests here.
  - `ChimpVideos/output/` — processing outputs and interim artifacts produced by extraction scripts.
  - `ChimpVideos/GroundThrouth/` — ground-truth annotations or reference labels (name preserved).
  - Date-stamped `*.txt` files (e.g., `20241018 - 07h56.txt`) — session manifests or processing logs.
  - `counting_chimp.py` — utility script used to produce counts or derived metrics.

- `ChimpVideosPic/face_recognition_db/` — final curated face-image dataset used for training and evaluation. Expect either per-identity folders or a dataset manifest inside this folder.

Reproducible process (name and order of steps to reproduce data preparation):

1. Place raw recordings, track files, and any session manifests into `ChimpVideos/input/`.
2. Run the data-preparation scripts (data engine / extractors). These read `ChimpVideos/input/` and write intermediate results to `ChimpVideos/output/` and logs to the date-stamped `.txt` files.
3. Store or consult ground-truth annotations in `ChimpVideos/GroundThrouth/` if available for evaluation.
4. Export the final curated face-image dataset to `ChimpVideosPic/face_recognition_db/`. Treat this folder as the canonical input for `finetuning/` and `Evaluation/`.

Notes:
- This mapping is derived from folder and file names only. Verify actual file contents, naming conventions, and any manifest formats before running full experiments.
- When running experiments, record the exact preprocessing script names, command-line arguments, and timestamps so runs can be reproduced from `ChimpVideos` → `ChimpVideosPic`.

