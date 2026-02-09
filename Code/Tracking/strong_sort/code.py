from ui_lib_strongSort import *

# path of the body detection model
body_model_path = "/etinfo/users/2024/trixen/Documents/Master_thesis/ChimpRec/Code/Tracking/strong_sort/Body_detection_models/Body_detection_model.pt"

# video paths:
# input video directory (without any annotation)
input_video_directory = "/etinfo/users/2024/trixen/Documents/Master_thesis/ChimpRec/ChimpRec_videos/Input"
# output (final version - with human interaction)
output_video_directory = "/etinfo/users/2024/trixen/Documents/Master_thesis/ChimpRec/ChimpRec_videos/Output"

ignore_S1 = [ # related to step 1
    "1",
    "2",
    "3",
    "4",
    "4_cropped",
    "5",
    "6",
    "7",
    "8",
    "12h41_full.MP4",
    "12h41_short.MP4",
    #"20241019 - 13h28.MP4"
]

ignore_S2 = [ # related to step 2
    "1",
    "2",
    "3",
    "4",
    "4_cropped",
    "5",
    "6",
    "7",
    "8",
    "12h41_full.MP4",
    "12h41_short.MP4"
    #"20241019 - 13h28.MP4"
]

ignore_S3 = [ # related to step 3
    "1",
    "2",
    "3",
    "4",
    "4_cropped",
    "5",
    "6",
    "7",
    "8",
    "12h41_full.MP4",
    "12h41_short.MP4",
    #"20241019 - 13h28.MP4"
] 

def has_audio_stream(video_path: str) -> bool:
    """
    Optional: quick check to avoid mux when there is no audio.
    If your mux_audio already tolerates missing audio, you can skip this.
    """
    try:
        import subprocess, json
        probe_cmd = [
            "ffprobe",
            "-v", "error",
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
        # Fallback: assume audio exists to keep behavior; mux_audio should handle errors gracefully
        return True
    
# FIRST STEP CODE (double-click to extend)

# Directories (same as before)
mannual_annotations_directory = f"{input_video_directory}/manual_annotations"
output_video_directory_temp = f"{output_video_directory}/temp"
raw_text_output_directory = f"{output_video_directory_temp}/raw_output"
treated_directory = f"{output_video_directory}/treated"
final_directory = f"{output_video_directory}/final"


# Create the directories if they do not exist yet
os.makedirs(input_video_directory, exist_ok=True)
os.makedirs(output_video_directory, exist_ok=True)
os.makedirs(mannual_annotations_directory, exist_ok=True)
os.makedirs(output_video_directory_temp, exist_ok=True)
os.makedirs(raw_text_output_directory, exist_ok=True)

# YOLOv8s initialisation (same)
YOLOv8s = YOLO(body_model_path)

# StrongSORT initialisation (no manual OSNet or DeepSORT metric needed)
# If you have a reid weights .pt, point to it; otherwise set to None and StrongSORT may auto-handle/download defaults
reid_weights_path = "/etinfo/users/2024/trixen/Documents/Master_thesis/ChimpRec/Code/Tracking/strong_sort/Re-ID_models/osnet_ain_x1_0_imagenet.pth"  # e.g. "/path/to/osnet_x0_25_msmt17.pt" if you have one
configuration_file_path = "/etinfo/users/2024/trixen/Documents/Master_thesis/ChimpRec/Code/Tracking/strong_sort/config/strongsort_config.yaml"  # e.g. "/path/to/strong_sort.yaml" if you have a custom config; otherwise set to None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
strongsort = build_strongsort(reid_weights=reid_weights_path, device=device, fp16=False, tracker_config_path=configuration_file_path)

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
        confidence_threshold=0.5
    )
    print(f"Annotations ready for video: {full_video_path}.\n")

    processed_with_audio = f"{output_video_directory_temp}/{video_name}-(temp)-audio.mp4"

    # Draw and mux (only one output, with audio when available)
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

