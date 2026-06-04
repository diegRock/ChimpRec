# PostProcessingDualHeuristic — Setup & Installation

This folder contains the dual-heuristic post-processing pipeline for re-identifying chimpanzees in tracked videos. It takes frozen ByteTrack output and performs embedding-based clustering with silhouette-score K selection.

## Overview

```
PostProcessingDualHeuristic/
├── pipeline_dual_heuristic.ipynb  # Main notebook (edit paths, run pipeline, inspect results)
├── dual_heuristic_pipeline.py     # Core implementation (CLI + library)
├── requirements.txt               # Python dependencies
├── README.md                      # Algorithm documentation
├── ChimpUFE/                      # Face embedding model (ViT backbone)
│   └── src/
│       ├── face_embedder/         # Vision Transformer implementation
│       └── tracker/yolox/         # YOLOX face detector
└── weights/                       # Model files (NOT in git — see below)
    ├── 25-08-29T11-49-28_340k.pth  # ChimpUFE ViT weights
    └── yolox_best_only_model.pth   # Face detector weights
```

## Setup on a New Machine

### 1. Create & activate Python venv

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

### 3. Obtain Model Weights

Model weights are **not stored in this Git repository** (too large). They must be downloaded  separately.

#### Download from Google Drive

1. Visit the link: [ChimpRec model weights](https://drive.google.com/drive/folders/1flt3_ifyvCKNnm3gnPVIodq9-6n38-R7?usp=sharing)
2. Download the following two files into the `weights/` folder:
   - `25-08-29T11-49-28_340k.pth` (416 MB) — ChimpUFE ViT backbone
   - `yolox_best_only_model.pth` (379 MB) — YOLOX face detector

After download, verify:

```bash
ls -lh weights/
# Output should show:
# -rw-r--r--  416M  25-08-29T11-49-28_340k.pth
# -rw-r--r--  379M  yolox_best_only_model.pth
```

### 4. Prepare input data structure

The pipeline expects ByteTrack output in the following format:

```
ChimpVideos/
├── input/
│   ├── video1.MP4
│   ├── video2.MP4
│   └── ... (add your videos here)
└── output/
    └── temp/
        └── raw_output/
            ├── video1.txt    # ByteTrack output (#-separated format)
            ├── video2.txt
            └── ...
```

The ByteTrack `.txt` format is:
```
frame_id, track_id, x1, y1, x2, y2
```

### 5. Run the pipeline

#### Option A: Use the Jupyter notebook (recommended for single videos)

```bash
jupyter notebook pipeline_dual_heuristic.ipynb
```

1. Edit the `VIDEO_NAME` variable in the first cell to match your video
2. Verify paths and model existence in the configuration cell
3. Run the "Run the post-processing" cell
4. Inspect diagnostics and distance matrix in the final cells

#### Option B: Use the CLI (for batch processing)

```bash
python dual_heuristic_pipeline.py \
  --tracks ChimpVideos/output/temp/raw_output/video1.txt \
  --video ChimpVideos/input/video1.MP4 \
  --face-model weights/yolox_best_only_model.pth \
  --chimpufe-root ChimpUFE \
  --chimpufe-weights weights/25-08-29T11-49-28_340k.pth \
  --output results/video1_dualheuristic.txt \
  --diagnostics-json results/video1_dualheuristic.json \
  --distance-matrix-npy results/video1_distance_matrix.npy
```

For help on all options:

```bash
python dual_heuristic_pipeline.py --help
```

#### Option C: Precomputed signatures (debug clustering only)

If you want to test clustering logic without re-extracting crops and embeddings:

```bash
python dual_heuristic_pipeline.py \
  --tracks ChimpVideos/output/temp/raw_output/video1.txt \
  --signatures-npz precomputed_signatures.npz \
  --output results/video1_debug.txt \
  --diagnostics-json results/video1_debug.json
```

The `.npz` file must contain:
- `track_ids`: array of original ByteTrack track IDs
- `signatures`: array of shape `(N_tracks, embedding_dim)` with L2-normalized embeddings

## Output Format

The pipeline writes a clustered prediction file (default: `results/{video}_dualheuristic.txt`):

```
frame_id, cluster_0, x1, y1, x2, y2
frame_id, cluster_1, x1, y1, x2, y2
frame_id, cluster_2, x1, y1, x2, y2
```

Clusters are named `cluster_0`, `cluster_1`, etc., corresponding to anonymous identities discovered by silhouette-based K selection.

Additionally:

- **Diagnostics JSON** (`{video}_dualheuristic.json`):
  - `n_embedded_tracklets`: number of tracklets with valid embeddings
  - `n_unembedded_tracklets`: tracklets that failed embedding (too short, no face detected)
  - `k_min`: physical lower bound on K
  - `selected_k`: K chosen by silhouette search
  - `silhouette_scores`: array of silhouette scores for K = k_min..20

- **Distance Matrix** (`{video}_distance_matrix.npy`):
  - Shape: `(n_tracklets, n_tracklets)`
  - Values: cosine distances (0 = identical, 1 = orthogonal)
  - Loadable with `numpy.load()`


## Algorithm Overview

1. **Track loading**: Read ByteTrack `.txt` file, filter tracklets by `min_track_len`
2. **Frame sampling**: Uniformly sample frames from each tracklet
3. **Face detection**: Detect and extract faces from body crops using YOLOX
4. **Face ranking**: Score by frontal-ness and sharpness; keep top 5 crops per tracklet
5. **Embedding**: Pass selected crops through ChimpUFE ViT backbone
6. **Aggregation**: Average embeddings per tracklet, L2-normalize
7. **Distance matrix**: Compute pairwise cosine distances with co-alive penalty
8. **K selection**: Physical lower bound + silhouette search (K=2 to 20)
9. **Clustering**: Average-linkage agglomerative clustering
10. **Output**: Write frame-by-frame predictions with cluster IDs

For full algorithm details, see [README.md](README.md).