from ui_lib_strongSort import *
import torch
import os

# Enable cuDNN autotune for fixed-size video frames
torch.backends.cudnn.benchmark = True
# (Optional) improve matmul kernels on Ampere+
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# Normalize device: use first visible GPU if available
device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

# Path of the body detection model
body_model_path = "../../../Models/Body Detection/Body_detection_model.pt"

# Video paths
input_video_directory = "../../../ChimpVideos/input"
output_video_directory = "../../../ChimpVideos/output"

ignore_S1 = []  # step 1
ignore_S2 = []  # step 2
ignore_S3 = []  # step 3

def has_audio_stream(video_path: str) -> bool:
    try:
        import subprocess, json
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "json",
            video_path,
        ]
        out = subprocess.check_output(probe_cmd).decode("utf-8")
        data = json.loads(out)
        streams = data.get("streams", [])
        return len(streams) > 0
    except Exception:
        return True

# Directories
mannual_annotations_directory = f"{input_video_directory}/manual_annotations"
output_video_directory_temp = f"{output_video_directory}/temp"
raw_text_output_directory = f"{output_video_directory_temp}/raw_output"
treated_directory = f"{output_video_directory}/treated"
final_directory = f"{output_video_directory}/final"

for d in [
    input_video_directory,
    output_video_directory,
    mannual_annotations_directory,
    output_video_directory_temp,
    raw_text_output_directory,
]:
    os.makedirs(d, exist_ok=True)

# YOLOv8s initialization on GPU (if available) with FP16
YOLOv8s = YOLO(body_model_path)
YOLOv8s.to(device_str)
use_half = device_str.startswith("cuda")
# StrongSORT initialization (FP16 if on GPU)
reid_weights_path = "../../../Models/Re-ID/osnet_ain_x1_0_imagenet.pth"
configuration_file_path = "./configs/strong_sort.yaml"
strongsort = build_strongsort(
    reid_weights=reid_weights_path,
    device=device_str,
    fp16=use_half,
    tracker_config_path=configuration_file_path,
)

for input_video in os.listdir(input_video_directory):
    if not (input_video.endswith(".mp4") or input_video.endswith(".MP4")):
        continue

    full_video_path = os.path.join(input_video_directory, input_video)
    video_name = os.path.splitext(input_video)[0]

    if video_name in ignore_S1:
        print(f"{video_name}.mp4 ignored")
        continue

    annotation_file_path = f"{mannual_annotations_directory}/{video_name}.txt"
    try:
        with open(annotation_file_path, "x") as f:
            print(f"{video_name}.txt automatically created in {mannual_annotations_directory}.")
    except FileExistsError:
        print(f"{video_name}.txt already present in {mannual_annotations_directory}.")
    print()

    raw_txt_path = f"{raw_text_output_directory}/{video_name}.txt"
    perform_tracking(
        input_video_path=full_video_path,
        output_text_file_path=raw_txt_path,
        detection_model=YOLOv8s,
        tracker=strongsort,
        confidence_threshold=0.5,
        device=device_str,
        use_half=use_half,
    )
    print(f"Annotations ready for video: {full_video_path}.\n")

    processed_with_audio = f"{output_video_directory_temp}/{video_name}-(temp)-audio.mp4"

    draw_bbox_from_file(
        file_path=raw_txt_path,
        input_video_path=full_video_path,
        output_video_path=processed_with_audio,
        annotation_type="bbox",
        draw_frame_count=True,
    )

    if has_audio_stream(full_video_path):
        print("Adding audio...")
        mux_audio(full_video_path, processed_with_audio, processed_with_audio)
    else:
        print("No audio stream detected; skipping mux.")
    print(f"Treatment done: {full_video_path}.\n")