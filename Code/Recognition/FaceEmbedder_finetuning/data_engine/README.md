# FaceEmbedder — Data Engine

This folder hosts the extractor and small utilities used to prepare the dataset for finetuning the face embedding model.

Note: for global prerequisites, environment setup and the recommended repository layout, see the top-level module README at `..` (FaceEmbedder_finetuning/README.md). This file focuses only on extractor-specific file formats, options and examples.

## Contents

- `extract_face_crops_from_named_tracks.py` — main extractor: pairs track files with videos, detects faces inside tracked body boxes, applies quality filters, and saves per-chimp crops.
- `filter_frontal_faces.py` — optional post-filter to keep only strongly frontal crops.
- `chimpUFE_data_analysis.ipynb` — exploratory notebook for quick dataset checks.

## Track file format

The extractor expects plain-text track files. Frames are grouped in blocks separated by a line containing only `#`.

Line format:

```
<name> <x1> <y1> <x2> <y2>
```

Coordinates may be floats. Example:

```
#
amadi 100 120 220 320
kira 300 110 420 320
#
amadi 102 123 222 323
kira 302 112 422 322
```

Pairing rule: the script matches a track file to a video by filename stem (e.g. `session1.txt` ↔ `session1.mp4`). Use `--tracks-dir` / `--videos-dir` or `--track-path` / `--video-path` to point the extractor at your files.

## Main pipeline (high level)

1. Parse track file into per-frame named bounding boxes.
2. Read video frames and crop each tracked body box.
3. Run YOLOX face detector inside each body crop (batched on GPU when available).
4. Select best face candidate using geometric constraints.
5. Assess crop quality (size, sharpness, brightness, contrast).
6. Optionally filter for frontalness and visible eyes.
7. Save accepted crops under `OUTPUT_ROOT/<chimp_name>/` and write CSV logs.

## Default weights

Default YOLOX weights location (can be overridden):

```
<repo_root>/Models/Face Recognition/yolox_best_only_model.pth
```

## Examples — run the extractor

From this folder (or use the full path to the script):

```bash
python extract_face_crops_from_named_tracks.py \
  --output-root /path/to/output/extracted_faces \
  --tracks-dir /path/to/data/raw/tracks \
  --videos-dir /path/to/data/raw/videos \
  --recursive \
```
## Two-step workflow — extract then optionally filter for frontal faces

Recommended approach:

1. Run the extractor to produce a full set of candidate face crops (no frontal filtering by default).
2. Optionally run `filter_frontal_faces.py` on the extracted crops to keep only frontal, eye-visible images.

Defaults and how to disable checks

- Extraction (single-step):
  - `--front-only` is a boolean flag and is OFF by default (no frontal filtering).
  - `--require-eyes` is a boolean flag and is OFF by default (no eye-visibility requirement).
  - `--min-front-score` default is `0.0` (no frontal threshold by default).
  - `--min-eye-detections` default is `0` (no eye requirement by default).

- Post-filter (`filter_frontal_faces.py`):
  - `--min-frontal-score` default is `0.55`.
  - `--min-eye-count` default is `1`.
  - To disable a check in the post-filter stage, pass `0` for the corresponding threshold (for example `--min-frontal-score 0 --min-eye-count 0`).

Example — typical two-step run (extract then filter):

```bash
python extract_face_crops_from_named_tracks.py \
  --output-root /path/to/output/extracted_faces \
  --tracks-dir /path/to/data/raw/tracks \
  --videos-dir /path/to/data/raw/videos \
  --recursive \
  --show-progress

python filter_frontal_faces.py \
  --input-root /path/to/output/extracted_faces \
  --output-root /path/to/output/extracted_faces_frontal \
  --min-frontal-score 0.55 \
  --min-eye-count 1 \
  --show-progress
```

Example — enable frontal + eye checks during extraction (single-step):

```bash
python extract_face_crops_from_named_tracks.py \
  --output-root /path/to/output/extracted_faces \
  --tracks-dir /path/to/data/raw/tracks \
  --videos-dir /path/to/data/raw/videos \
  --front-only \
  --min-front-score 0.80 \
  --require-eyes \
  --min-eye-detections 1 \
  --show-progress
```
```
  --track-path /path/to/data/raw/tracks/session1.txt \
  --video-path /path/to/data/raw/videos/session1.mp4
```
```
python Code/Recognition/ChimpUFE/filter_frontal_faces.py \
    --input-root        ChimpPic/face_recognition_db/face_crops_for_chimpufe \
    --output-root       ChimpPic/face_recognition_db/face_crops_frontal_55 \
    --min-frontal-score 0.55 \
    --min-eye-count     1 \
    --show-progress
```
## Common options

- `--output-root` (required): root directory for per-chimp folders and CSV logs
- `--chimp-names`: allowed canonical names (script has a default list)
- `--name-alias`: alias mappings (format `old=new`)
- `--yolox-weights`: path to YOLOX checkpoint
- `--face-conf-thres`: face detection confidence threshold
- `--min-face-width`, `--min-face-height`: minimum accepted face size
- `--min-sharpness`, `--min-contrast`, `--min-brightness`, `--max-brightness`: quality tuning parameters
- `--front-only` / `--min-front-score`: frontal-only filtering
- `--require-eyes` / `--min-eye-detections`: stricter eye-visibility filter
- `--max-per-chimp`: cap saved crops per identity
- `--frame-step`: process one frame every N frames

See the script `--help` for the full list of flags and defaults.

## Output and logs

- Accepted crops: `OUTPUT_ROOT/<canonical_chimp_name>/` (filenames: `<video_stem>_f<frame_idx>_<chimp>_<index>.jpg`).
- CSVs (by default written inside `OUTPUT_ROOT`):
  - `accepted_crops.csv` — metadata for saved crops
  - `rejected_candidates.csv` — rejection reasons and quality metrics

## Tips & troubleshooting

- If videos fail to open, check codecs and file accessibility; OpenCV depends on system codecs for some formats.
- If the detector cannot load, pass `--yolox-weights` with a valid checkpoint.
- To debug quality filters, temporarily lower `--min-sharpness` and `--min-contrast`.
- Use `--show-progress` to monitor acceptance/rejection counts live.


