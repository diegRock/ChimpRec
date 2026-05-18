"""
Helpers used by ``pipeline_final.ipynb``.

Step 1 of the notebook deploys the auto-labelling pipeline (from
``utils/auto_label_run.py``) on every video that does not yet have a manual
annotation.  Steps 2 and 3 reuse the manual-correction machinery from
``utils/ui_lib_bytetrack.py``.

Compared with ``auto_label_run.main()`` this wrapper:
    * loads the models once and processes a list of videos;
    * does NOT compute HOTA / recognition metrics (no GT available);
    * does NOT dump crop visualisations to ChimpPic;
    * writes a name-prefilled manual annotation file so that the
      manual correction (step 2) starts from the auto-label prediction.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the workspace and import the modules we depend on.
# ---------------------------------------------------------------------------
THIS_DIR      = Path(__file__).resolve().parent            # .../FinalPipeline/utils
PIPELINE_DIR  = THIS_DIR.parent                            # .../FinalPipeline
PROJECT_ROOT  = PIPELINE_DIR.parent.parent                 # .../ChimpRec
CODE_DIR      = PROJECT_ROOT / "Code"
WEIGHTS_DIR   = PIPELINE_DIR / "weights"

# Allow `from chimplib...` etc.
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# --- 1. Import the auto-label module by file path ---------------------------
_AL_PATH = str(THIS_DIR / "auto_label_run.py")
_spec = importlib.util.spec_from_file_location("auto_label_run", _AL_PATH)
al = importlib.util.module_from_spec(_spec)
sys.modules["auto_label_run"] = al
_spec.loader.exec_module(al)

# --- 2. Override model paths to use the local weights copies ----------------
al.BODY_MODEL_PATH    = WEIGHTS_DIR / "Body_detection_model.pt"
al.YOLOX_FACE_WEIGHTS = WEIGHTS_DIR / "yolox_best_only_model.pth"
al.CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "fine_tune_20.pt"

# --- 3. Re-export the manual-edit utilities ---------------------------------
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
import ui_lib_bytetrack as _uib                                    # noqa: E402

raw_tracking_data_reader = _uib.raw_tracking_data_reader
modification_reader      = _uib.modification_reader
data_writer              = _uib.data_writer
edit_raw_output          = _uib.edit_raw_output
draw_bbox_from_file      = _uib.draw_bbox_from_file
mux_audio                = _uib.mux_audio


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def has_audio_stream(video_path: str | os.PathLike) -> bool:
    """True iff `video_path` contains at least one audio stream."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0",
             str(video_path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return bool(out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def prefill_manual_annotation_file(id_map: dict, annotation_path: str | os.PathLike) -> None:
    """
    Write a "Name: id1 id2 ..." annotation file built from a vFinal `id_map`
    (track_id -> predicted name | "unknown_N" | None).

    `name` is used as the per-track id token in the raw output file, so
    grouping by name gives the user a starting point of the form

        Amadi: Amadi
        Maniema: Maniema
        unknown_1: unknown_1
        unknown_2: unknown_2

    The user only has to fix mistakes (e.g. rename `unknown_3` into
    `Amadi`, or split a misclustered track with a swap).
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for _tid, name in id_map.items():
        if name is None:
            continue
        token = al._gt_alias(str(name)).replace(" ", "_")
        grouped[token].append(token)

    # one line per identity; keep known names first, then unknown_*
    known   = sorted(k for k in grouped if not k.startswith("unknown_"))
    unknown = sorted((k for k in grouped if k.startswith("unknown_")),
                     key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0)

    lines = []
    for name in known + unknown:
        ids = sorted(set(grouped[name]))
        lines.append(f"{name}: {' '.join(ids)}")

    Path(annotation_path).parent.mkdir(parents=True, exist_ok=True)
    with open(annotation_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


# ---------------------------------------------------------------------------
# Auto-label wrapper – load the models once, process N videos.
# ---------------------------------------------------------------------------
class VFinalPipeline:
    """
    Thin wrapper around :mod:`auto_label_run`.  Loads the body detector, the
    YOLOX face detector and the ChimpUFE classifier once (saves ~30 s per
    video).
    """

    def __init__(self,
                 body_weights: str | os.PathLike | None = None,
                 yolox_weights: str | os.PathLike | None = None,
                 classifier_weights: str | os.PathLike | None = None,
                 device: str | None = None):

        if body_weights is not None:       al.BODY_MODEL_PATH    = Path(body_weights)
        if yolox_weights is not None:      al.YOLOX_FACE_WEIGHTS = Path(yolox_weights)
        if classifier_weights is not None: al.CLASSIFIER_WEIGHTS = Path(classifier_weights)
        self.device = device or al.DEVICE

        print(f"[VFinalPipeline] device = {self.device}")
        print(f"[VFinalPipeline] body  weights = {al.BODY_MODEL_PATH}")
        print(f"[VFinalPipeline] yolox weights = {al.YOLOX_FACE_WEIGHTS}")
        print(f"[VFinalPipeline] clf   weights = {al.CLASSIFIER_WEIGHTS}")

        self.body_model = al.YOLO(str(al.BODY_MODEL_PATH))
        self.face_pred  = al.YoloXFacePredictor(al.YOLOX_FACE_WEIGHTS, self.device)
        self.classifier, idx_to_name = al.load_classifier(al.CLASSIFIER_WEIGHTS, self.device)
        self.idx_to_name = {i: n.capitalize() for i, n in idx_to_name.items()}
        print(f"[VFinalPipeline] classes ({len(self.idx_to_name)}): "
              f"{list(self.idx_to_name.values())}")

    # ------------------------------------------------------------------
    def process(self,
                video_path: str | os.PathLike,
                output_txt_path: str | os.PathLike,
                *,
                merge_sim: float    | None = None,
                cluster_conf: float | None = None,
                cluster_margin: float | None = None,
                prop_sim: float     | None = None,
                gap_max: int        | None = None) -> dict:
        """
        Runs the full auto-labelling pipeline on a single video and
        writes the result to `output_txt_path` (in the `#`-delimited
        per-frame format used by `raw_tracking_data_reader`).

        Returns the final `id_map` so the caller can also write a
        prefilled manual annotation file.
        """
        merge_sim       = al.MERGE_SIM       if merge_sim       is None else merge_sim
        cluster_conf    = al.CLUSTER_CONF    if cluster_conf    is None else cluster_conf
        cluster_margin  = al.CLUSTER_MARGIN  if cluster_margin  is None else cluster_margin
        prop_sim        = al.PROP_SIM        if prop_sim        is None else prop_sim
        gap_max         = al.GAP_MAX_FRAMES  if gap_max         is None else gap_max

        video_path = Path(video_path)
        output_txt_path = Path(output_txt_path)
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[VFinalPipeline] >>> {video_path.name}")

        # 2. ByteTrack on body detections
        all_frame_tracks, _fps = al.run_bytetrack(
            video_path, self.body_model, al.BYTETRACK_PARAMS,
        )
        tracks = al.frame_tracks_to_track_dict(all_frame_tracks)
        track_list = sorted(tracks.values(), key=lambda x: x["frames"][0])

        # 3. Face crops
        track_crops = al.collect_face_crops(
            video_path, track_list, self.face_pred,
            pool_size=al.FACE_POOL_SIZE,
        )

        # 4. Classify + embed
        id_map_first, info_map, track_emb = {}, {}, {}
        for t in al.tqdm(track_list, desc="Classify+Emb"):
            if len(t["frames"]) <= al.MIN_TRACK_LEN:
                continue
            crops = al.select_top_crops(track_crops.get(t["id"], []))
            if not crops:
                continue
            label, emb, info = al.classify_and_embed_track(
                crops, self.classifier, self.idx_to_name,
            )
            if info  is not None: info_map[t["id"]]  = info
            if emb   is not None: track_emb[t["id"]] = emb
            if label is not None: id_map_first[t["id"]] = label

        # 5. Cluster
        tracks_with_emb = []
        for t in track_list:
            if t["id"] in track_emb:
                tw = dict(t); tw["_emb"] = track_emb[t["id"]]
                tracks_with_emb.append(tw)
        uf, labels_of = al.cluster_tracks_by_embedding(
            tracks_with_emb, id_map_first, merge_sim=merge_sim,
        )

        # 6. Cross-cluster name merge (pass 1)
        uf, labels_of = al.cross_cluster_name_merge(uf, labels_of, tracks_with_emb)

        # 7. Cluster-aggregated classification
        cluster_label, cluster_emb, cluster_info = al.aggregate_cluster_classifications(
            uf, track_crops, track_list, self.classifier, self.idx_to_name,
            conf_min=cluster_conf, margin_min=cluster_margin,
        )
        al.merge_cluster_labels(uf, labels_of, cluster_label)
        uf, labels_of = al.cross_cluster_name_merge(uf, labels_of, tracks_with_emb)

        # 7b. Embedding propagation
        al.propagate_labels_by_embedding(
            uf, labels_of, cluster_emb, tracks_with_emb,
            prop_sim=prop_sim, overlap_tol=al.MERGE_OVERLAP_TOL,
        )

        # Build cluster_of and final id_map
        cluster_of = {}
        for root, members in uf.members.items():
            for m in members:
                cluster_of[m] = root
        final_id_map = al.assign_cluster_labels(track_list, uf, labels_of, info_map)

        id_map = {t["id"]: final_id_map.get(t["id"]) for t in track_list}

        n_unknown = 0
        for t in track_list:
            if id_map.get(t["id"]) is not None:
                continue
            if len(t["frames"]) < al.DROP_LEN:
                id_map[t["id"]] = None
            else:
                n_unknown += 1
                id_map[t["id"]] = f"unknown_{n_unknown}"

        # Stats
        name_counts = defaultdict(int)
        for v in id_map.values():
            if v is not None and not str(v).startswith("unknown_"):
                name_counts[v] += 1
        n_labelled = sum(1 for v in id_map.values()
                         if v is not None and not str(v).startswith("unknown_"))
        n_dropped = sum(1 for v in id_map.values() if v is None)
        print(f"  -> labelled={n_labelled}  unknown={n_unknown}  "
              f"dropped={n_dropped}  identities={len(name_counts)}")
        for nm, c in sorted(name_counts.items(), key=lambda x: -x[1]):
            print(f"     {nm:14s}: {c} tracks")

        # Write raw output (per-frame dedup: highest-confidence box wins
        # when the same name appears more than once in a frame).
        drop_log: dict = defaultdict(int)
        n_written, n_dropped = al.write_tracks(
            all_frame_tracks, id_map, output_txt_path,
            cluster_of, cluster_info, info_map, drop_log=drop_log,
        )
        print(f"  [write] {n_written} boxes written, "
              f"{n_dropped} boxes dropped (per-frame conflicts)")

        # Gap interpolation (linear bbox fill inside short gaps between
        # fragments sharing the same final name within a cluster).
        n_added = al.interpolate_gaps(
            all_frame_tracks, id_map, cluster_of, gap_max=gap_max,
        )
        if n_added > 0:
            print(f"  gap interpolation (gap_max={gap_max}): "
                  f"{n_added} synthetic boxes added.")
            drop_log.clear()
            al.write_tracks(
                all_frame_tracks, id_map, output_txt_path,
                cluster_of, cluster_info, info_map, drop_log=drop_log,
            )

        return id_map


# ---------------------------------------------------------------------------
# Convenience: copy a weight if it does not already exist locally.
# ---------------------------------------------------------------------------
def ensure_weights(verbose: bool = True) -> None:
    """Idempotent re-copy of the three weight files into ./weights/."""
    sources = {
        "Body_detection_model.pt":    CODE_DIR / "Tracking" / "Body_detection_model.pt",
        "yolox_best_only_model.pth":  CODE_DIR / "PostProcessing" / "ChimpUFE" / "assets" / "weights" / "yolox_best_only_model.pth",
        "fine_tune_20.pt":            CODE_DIR / "PostProcessing" / "ChimpUFE" / "assets" / "weights" / "fine_tune_20.pt",
    }
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in sources.items():
        dst = WEIGHTS_DIR / name
        if dst.exists():
            if verbose:
                print(f"[weights] OK  {dst}")
            continue
        if not src.exists():
            print(f"[weights] WARN source missing: {src}")
            continue
        shutil.copy2(src, dst)
        if verbose:
            print(f"[weights] copied {src} -> {dst}")
