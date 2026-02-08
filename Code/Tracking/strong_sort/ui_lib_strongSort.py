import sys

# Add your chimplib and boxmot repo to PYTHONPATH
sys.path.append("/etinfo/users/2024/trixen/Documents/Master_thesis/ChimpRec/Code/chimplib")
sys.path.append("/etinfo/users/2024/trixen/Documents/Master_thesis/ChimpRec/Code/chimplib/boxmot")

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Use the tracker factory from boxmot
from boxmot.tracker_zoo import create_tracker

# Optional: point to the StrongSORT YAML config inside your boxmot clone
BOXMOT_ROOT = "/etinfo/users/2024/trixen/Documents/Master_thesis/ChimpRec/Code/chimplib/boxmot"
STRONGSORT_CFG = os.path.join(
    BOXMOT_ROOT, "boxmot", "trackers", "strongsort", "configs", "strongsort.yaml"
)

colors = [
    (120, 50, 99), (180, 25, 16), (73, 89, 176), (200, 158, 18), (199, 214, 152),
    (181, 37, 229), (118, 73, 165), (136, 3, 53), (40, 47, 142), (246, 26, 168),
    (33, 83, 190), (151, 220, 243), (156, 122, 217), (173, 0, 128), (61, 242, 230),
    (37, 10, 125), (64, 229, 201), (64, 137, 49), (136, 225, 85), (146, 80, 77),
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (255, 165, 0), (255, 255, 255), (0, 0, 0), (128, 0, 0),
    (0, 128, 0), (128, 128, 0), (0, 128, 128), (128, 0, 128), (255, 105, 180),
    (255, 69, 0), (34, 139, 34), (70, 130, 180), (255, 228, 225), (218, 165, 32)
]

class raw_tracking_data_reader():
    def __init__(self, text_file_path):
        self.text_file_path = text_file_path
        self.read()

    def read(self):
        parsed_content = []
        with open(self.text_file_path, 'r') as text_file:
            text_content = text_file.read()
            splitted_content = text_content.split("#\n")
            for i in splitted_content:
                if len(i) < 1:
                    continue
                block = []
                for j in i.split("\n"):
                    block.append(j.split(" "))
                parsed_content.append(block)
            text_file.close()
        self.data = parsed_content

class modification_reader:
    def __init__(self, text_file_path):
        self.text_file_path = text_file_path
        self.swaps = {}
        self.read()

    def read(self):
        parsed_content = []
        unknown_id_index = 0
        with open(self.text_file_path, 'r') as text_file:
            text_content = text_file.read()
            splitted_content = text_content.split("\n")
            for i in splitted_content:
                if len(i) < 1:
                    continue
                if (":" in i):
                    name = i.split(": ")[0]
                    if name.upper() == "SWAP":
                        frame_count, swap_id_1, swap_id_2 = i.split(": ")[1].split(" ")
                        if (swap_id_1 in self.swaps.keys()):
                            self.swaps[swap_id_1].append((frame_count, swap_id_2))
                        else:
                            self.swaps[swap_id_1] = [(frame_count, swap_id_2)]
                        if (swap_id_2 in self.swaps.keys()):
                            self.swaps[swap_id_2].append((frame_count, swap_id_1))
                        else:
                            self.swaps[swap_id_2] = [(frame_count, swap_id_1)]
                        continue
                    parsed_content.append([name, i.split(": ")[1].split(" ")])
                else:
                    name = f"UNK_{unknown_id_index}"
                    parsed_content.append([name, i.split(" ")])
                    unknown_id_index += 1
                    
            text_file.close()

        content = ""
        for name, numbers in parsed_content:
            content = f"{content}{name}: {' '.join(str(n) for n in numbers)}\n"
        
        with open(self.text_file_path, "w") as f:
            f.write(content)
        f.close()

        self.data = parsed_content

class data_writer:
    def __init__(self, output_text_file_path):
        self.out_path = output_text_file_path
        with open(self.out_path, "w") as temp:
            temp.close()
    
    def write(self, data):
        with open(self.out_path, "a") as output_file:
            for block in data:
                block_string = "#\n"
                for line in block:
                    block_string = f"{block_string}{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n"
                if block_string == "#\n":
                    block_string = f"{block_string}\n"
                output_file.write(block_string)
            output_file.close()

def edit_raw_output(RTD_reader, M_reader):
    modified_data = []
    for current_frame, block in enumerate(RTD_reader.data):
        new_block = []
        for line in block:
            class_id = line[0]
            label = ""
            keep = False

            if class_id in M_reader.swaps.keys():
                for frame_index, other_id in M_reader.swaps[class_id]:
                    if (current_frame >= int(frame_index)):
                        class_id = other_id
                        break

            for name, ids in M_reader.data:
                if class_id in ids:
                    keep = True
                    label = name

            new_line = [label] + line[1:]
            if len(new_line) != 5 or not keep:
                continue
            new_block.append(new_line)
        modified_data.append(new_block)
    return modified_data

def draw_bbox(image, color, bbox, label):
    x1, y1, x2, y2 = map(lambda v: int(float(v)), bbox)
    factor = 0.65 if label == "Face" else 0.3
    font_scale = max(0.65, ((x2 - x1 + y2 - y1) / 300) * factor)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    label_text = f"{label}"
    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, font_scale, 1)

    overlay = image.copy()
    cv2.rectangle(overlay, (x2 - w - 10, y2 - h - 10), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    cv2.putText(image, label_text, (x2 - w - 5, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255,255,255), 1)
    return image

def draw_triangle(image, color, bbox, label):
    x1, y1, x2, y2 = map(lambda v: int(float(v)), bbox)    
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.8, 1)

    triangle_height = 20
    triangle_width = 20

    tip_x = (x1 + x2) // 2
    tip_y = max(y1, text_height + triangle_height + 5)
    base_y = tip_y - triangle_height
    half_base = triangle_width // 2

    pt_tip = (tip_x, tip_y)
    pt_left = (tip_x - half_base, base_y)
    pt_right = (tip_x + half_base, base_y)

    triangle = np.array([pt_tip, pt_left, pt_right], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [triangle], color)

    text_x = tip_x - text_width // 2
    text_y = base_y + (triangle_height // 2) - triangle_height

    overlay = image.copy()
    cv2.rectangle(overlay, (text_x - 5, text_y - text_height - 5), (x + text_width + 5, y + 5), color, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
    cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)

    return image

def draw_bbox_from_file(file_path, input_video_path, output_video_path, annotation_type="bbox", draw_frame_count=False):
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    reader = raw_tracking_data_reader(file_path)
    frame_number = 0

    color_index = 0
    colors_used = {}
    seen_names = set()

    with tqdm(total=total_frames, desc=f"Drawing annotations ({os.path.basename(input_video_path)})") as pbar:
        while ret:
            if len(reader.data) <= frame_number:
                break
            bboxes = reader.data[frame_number]
            if len(bboxes) == 1 and bboxes[0] == '':
                frame_number += 1
                ret, frame = cap.read()
                pbar.update(1)
                continue

            for id_bbox in bboxes:
                if len(id_bbox) <= 1:
                    continue
                name, x1, y1, x2, y2 = id_bbox
                bbox = x1, y1, x2, y2
                if name not in seen_names:
                    seen_names.add(name)
                    colors_used[name] = colors[color_index]
                    color_index = (color_index + 1) % len(colors)

                if annotation_type == "bbox":
                    draw_bbox(frame, colors_used[name], bbox, name)
                elif annotation_type == "triangle":
                    draw_triangle(frame, colors_used[name], bbox, name)

            if draw_frame_count:
                label_text = f"{frame_number}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, 0.9, 1)
                x = frame_width - w - 10
                y = h + 10
                overlay = frame.copy()
                cv2.rectangle(overlay, (x - 5, y - h - 5), (x + w + 5, y + 5), (57, 46, 135), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, label_text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 1)

            out.write(frame)

            frame_number += 1
            ret, frame = cap.read()
            pbar.update(1)

    cap.release()
    out.release()

def build_strongsort(reid_weights=None, device=None, fp16=False,
                     tracker_config_path=None):
    """
    Build a StrongSORT tracker via boxmot.tracker_zoo.create_tracker.
    - reid_weights: str | Path to OSNet weights .pt (or None)
    - device: 'cuda' or 'cpu'
    - fp16: use half precision for the reid model
    - tracker_config_path: path to strongsort.yaml (optional)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Normalize reid_weights to Path or None
    if reid_weights:
        reid_weights = Path(reid_weights)
        if not reid_weights.exists():
            print(f"[StrongSORT] Warning: ReID weights not found at {reid_weights}. Falling back to defaults.")
            reid_weights = None
    else:
        reid_weights = None

    # If no config provided, try to auto-use the repo default
    if tracker_config_path is None and os.path.exists(STRONGSORT_CFG):
        tracker_config_path = STRONGSORT_CFG

    tracker = create_tracker(
        'strongsort',
        tracker_config=tracker_config_path if tracker_config_path and os.path.exists(tracker_config_path) else None,
        reid_weights=reid_weights,
        device=device,
        half=fp16
    )
    return tracker

def perform_tracking(input_video_path, output_text_file_path, detection_model, tracker, confidence_threshold, model_feature_extraction=None):
    """
    StrongSORT-based tracking with boxmot interface.
    - detection_model: ultralytics.YOLO instance (YOLOv8)
    - tracker: object from build_strongsort(...)
    Writes each frame as:
        #
        <track_id> <x1> <y1> <x2> <y2>
    """
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()

    with open(output_text_file_path, "w") as file_improve_tracking, tqdm(total=total_frames, desc=f"Tracking progress ({os.path.basename(input_video_path)})") as pbar:
        while ret:
            file_improve_tracking.write("#\n")

            preds = detection_model.predict(frame, verbose=False)[0]

            outputs = np.empty((0, 6))
            if preds.boxes is not None and len(preds.boxes) > 0:
                xyxy = preds.boxes.xyxy.detach().cpu().numpy()
                conf = preds.boxes.conf.detach().cpu().numpy().reshape(-1, 1)
                if hasattr(preds.boxes, "cls") and preds.boxes.cls is not None:
                    cls = preds.boxes.cls.detach().cpu().numpy().reshape(-1, 1)
                else:
                    cls = np.zeros_like(conf)

                keep = (conf.reshape(-1) >= confidence_threshold)
                xyxy = xyxy[keep]
                conf = conf[keep]
                cls = cls[keep]

                if xyxy.shape[0] > 0:
                    # dets: [x1, y1, x2, y2, conf, cls]
                    dets = np.hstack([xyxy, conf, cls])
                    outputs = tracker.update(dets, frame)
                else:
                    outputs = np.empty((0, 6))

            if outputs is not None and len(outputs) > 0:
                # outputs: [x1, y1, x2, y2, track_id, score, (cls? ...)]
                for det in outputs:
                    x1, y1, x2, y2 = det[:4]
                    track_id = det[4]
                    file_improve_tracking.write(
                        f"{int(track_id)} {float(x1)} {float(y1)} {float(x2)} {float(y2)}\n"
                    )
            else:
                file_improve_tracking.write("\n")

            ret, frame = cap.read()
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()