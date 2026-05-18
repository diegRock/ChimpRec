# ChimpRec FinalPipeline — Standalone Deployment

This folder contains the complete auto-labelling + manual-correction pipeline for chimp tracking videos.

## What's inside

```
FinalPipeline/
├── pipeline_final.ipynb          # Main deployment notebook (3 steps)
├── requirements.txt              # Python dependencies (GPU)
├── requirements-nogpu.txt        # Python dependencies (CPU-only)
├── SETUP.md                      # Setup & usage instructions
├── utils/                        # Core pipeline modules
│   ├── __init__.py
│   ├── pipeline_lib.py           # Wrapper our auto-labeller
│   ├── auto_label_run.py         # Core pipeline (embedding clustering, propagation, etc.)
│   └── ui_lib_bytetrack.py       # Manual correction utilities
├── ChimpUFE/                     # Chimp face embedding model source
│   └── src/
│       ├── face_embedder/
│       └── tracker/
├── metric/                       # Metrics & visualization
│   ├── HOTA.py
│   ├── visualize.py
│   ├── GT/                       # Ground-truth annotation files
│   └── prediction/               # Model prediction files for evaluation
├── weights/                      # Model weights (not tracked by git — see below)
│   ├── Body_detection_model.pt
│   ├── yolox_best_only_model.pth
│   └── fine_tune_20.pt
└── results/                      # Output directory (auto-created)
    └── plots/
```

> **Note:** `weights/` is excluded from git (see `.gitignore`). See the [Obtaining model weights](#obtaining-model-weights) section below for download instructions.

## Setup on a new machine

### 1. Install system dependencies

```bash
# Debian/Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
Visit the following website: https://www.gyan.dev/ffmpeg/builds/
Download ffmpeg-git-essentials.7z and extract it
Navigate through your file explorer. Go to the C:\ directory
Create a new folder named "ffmpeg" 
Copy all the content of the archive inside the folder ffmpeg
In utils/ui_lib_bytetrack.py, create a variable FFMPEG = "C:\ffmpeg\bin\ffmpeg.exe"
And replace:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", processed_video,      # video from processed file
        "-i", original_video,       # audio from original file
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        temp_out,
    ]
by:
    cmd = [
        FFMPEG,
        "-y",
        "-i", processed_video,      # video from processed file
        "-i", original_video,       # audio from original file
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        temp_out,
    ]
The idea is that you make a reference to the ffmpeg.exe file that will copy the audio of the source video
You have to do the same process in utils/pipeline_lib.py at line 72. 
Create the variable FFPROBE = "C:\ffmpeg\bin\ffprobe.exe"
And replace:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0",
             str(video_path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
by:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0",
             str(video_path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

```

### 2. Create & activate Python venv

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### 3. Install Python packages

**For GPU (CUDA) support:**
```bash
pip install -r requirements.txt
```

**For CPU-only (no CUDA required):**
```bash
pip install -r requirements-nogpu.txt
```

Note: CPU inference will be significantly slower (~10–50× depending on model). Only use if GPU is unavailable.

### 4. Prepare video directories

Create this structure at the same level as `FinalPipeline/`:

```
ChimpVideos/
├── input/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── manual_annotations/  (auto-created by pipeline)
├── output/
│   ├── temp/
│   ├── treated/
│   └── final/
└── GT/                      (optional, for metrics)
```

Alternatively, edit the notebook's `PROJECT_ROOT` path to point to your video directory.

### 5. Run the pipeline

```bash
jupyter notebook pipeline_final.ipynb
```

Then execute the three steps in order:
1. **Step 1**: Auto-labelling (ByteTrack + clustering + propagation)
2. **Step 2**: Manual corrections (edit `.txt` files, re-render videos)
3. **Step 3**: Final overlay (triangle + name visualization)

## Workflow

1. **Step 1** processes each input video and produces:
   - `temp/raw_output/{video_name}.txt` — raw per-frame predictions
   - `manual_annotations/{video_name}.txt` — pre-filled corrections (edit this)
   - `temp/{video_name}-(temp)-audio.mp4` — preview with bounding boxes

2. Edit the annotation files in `input/manual_annotations/` using the syntax:
   ```
   Amadi: Amadi unknown_3        # merge unknown_3 into Amadi
   Maniema: Maniema             
   unknown_1: unknown_1
   ```
   See `ui_lib_bytetrack.py` for advanced syntax (frame ranges, swaps, etc.).

3. **Step 2** applies your edits and renders corrected videos to `treated/`

4. **Step 3** renders final publication-style videos with triangle overlays to `final/`

## Dependencies

- `torch`, `torchvision` — deep learning
- `ultralytics` — YOLO body detection + ByteTrack
- `opencv-python` — video I/O and visualization
- `torchreid` — Re-ID features (used by ui_lib_bytetrack)
- `numpy`, `scipy`, `tqdm` — utilities

## Obtaining model weights

Model weights are **not stored in this repository** (too large for GitHub). Place the three `.pt`/`.pth` files under `weights/` before running the pipeline. Possible sources:

| Option | When to use |
|--------|-------------|
| **Google Drive** | Find all the weights following this link: [Download]](https://drive.google.com/drive/folders/1flt3_ifyvCKNnm3gnPVIodq9-6n38-R7?usp=sharing) |

## Notes

- Model weights are loaded once at startup (saves ~30 s per video)
- No ground-truth evaluation is included (for deployment on unlabelled videos)
- Crop visualizations are not dumped to disk (comment out `dump_track_visuals` in `utils/pipeline_lib.py` if you need them)
- `HOTA.py` and `visualize.py` are included locally in `metric/`; they don't require external metric code

## Troubleshooting

**ModuleNotFoundError: ChimpUFE source code**
- The ChimpUFE backbone is loaded dynamically. If you see import errors, ensure the `.venv/lib/pythonX.Y/site-packages/ultralytics/` and torch/torchvision are installed correctly.
- Try: `pip install --upgrade ultralytics torch torchvision`

**No video output or frozen preview**
- Ensure `ffmpeg` is installed and in your PATH.
- On Linux: `which ffmpeg`; on macOS: `which ffmpeg`; on Windows: verify your paths

**Model loading fails**
- Verify all three weight files exist in `weights/` (they are not tracked by git — see [Obtaining model weights](#obtaining-model-weights)):
  - `Body_detection_model.pt` (~500 MB)
  - `yolox_best_only_model.pth` (~50 MB)
  - `fine_tune_20.pt` (~200 MB)

## License & Attribution

This pipeline integrates:
- **ByteTrack**: https://github.com/ifzhang/ByteTrack
- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **ChimpUFE**: [Internal chimp face embedding model](https://github.com/v-iashin/ChimpUFE)