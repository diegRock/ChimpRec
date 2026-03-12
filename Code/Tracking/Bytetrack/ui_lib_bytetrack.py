import sys

# change the path below !!!
sys.path.append("/home/diego/Desktop/ChimpRec/ChimpRec/Code/chimplib/")
import tempfile
import sys
import os
import cv2
import subprocess
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import torchreid
from ultralytics import YOLO
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching

colors = [
    (120, 50, 99),
    (180, 25, 16),
    (73, 89, 176),
    (200, 158, 18),
    (199, 214, 152),
    (181, 37, 229),
    (118, 73, 165),
    (136, 3, 53),
    (40, 47, 142),
    (246, 26, 168),
    (33, 83, 190),
    (151, 220, 243),
    (156, 122, 217),
    (173, 0, 128),
    (61, 242, 230),
    (37, 10, 125),
    (64, 229, 201),
    (64, 137, 49),
    (136, 225, 85),
    (146, 80, 77),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 165, 0),
    (255, 255, 255),
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
    (255, 105, 180),
    (255, 69, 0),
    (34, 139, 34),
    (70, 130, 180),
    (255, 228, 225),
    (218, 165, 32)
]

class raw_tracking_data_reader():
    """
    This class reads the output file generated 
    based on the tracking step. And structures
    what has been read in a standardised format.
    """
    def __init__(self, text_file_path):
        self.text_file_path = text_file_path
        self.read()

    def read(self):
        parsed_content = []
        with open(self.text_file_path, 'r') as text_file:
            text_content = text_file.read()
            splitted_content = text_content.split("#\n")
            for i in splitted_content:
                if len(i) < 1: continue
                block = []
                for j in i.split("\n"):
                    block.append(j.split(" "))
                parsed_content.append(block)
            text_file.close()
        self.data = parsed_content
        # for i in self.data: print(i)

class modification_reader:
    """
    Reads manual annotation file.
    Supports:
      - Name: id1 id2 ...
      - Name: id*start-end (time-bounded assignment)
      - Name: id1, id2, id3*start-end (multi-id time-bounded assignment)
      - Name: id1*range, id2*range (comma-separated assignments)
      - SWAP: frame id1 id2
    """
    def __init__(self, text_file_path, rewrite=False):
        self.text_file_path = text_file_path
        self.swaps = {}
        self.rewrite = rewrite
        self.id_to_name_intervals = {}   # {id: [(name, start, end), ...]}
        self.id_to_name_global = {}      # {id: name} for full-video assignments
        self.read()

    def _parse_token(self, token):
        # token examples: "10", "10,11", "10*650-1000", "10,11*650-"
        
        # Check if this token has a time range (marked by *)
        if "*" not in token:
            # No range, just IDs (potentially comma separated like "1,2,3")
            ids = token.split(',')
            return ids, None, None
            
        id_part, rng = token.split("*", 1)
        
        # Parse the range part
        if "-" in rng:
            start_s, end_s = rng.split("-", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else float("inf")
        else:
            start = int(rng)
            end = float("inf")
            
        # Parse the ID part (split by commas)
        ids = id_part.split(',')
        
        return ids, start, end

    def read(self):
        parsed_content = []
        unknown_id_index = 0
        with open(self.text_file_path, 'r') as text_file:
            for line in text_file.read().split("\n"):
                if len(line) < 1:
                    continue
                if ":" in line:  # name present
                    name, rhs = line.split(": ", 1)
                    
                    # Handle SWAP separately first
                    if name.upper() == "SWAP":
                        frame_count, swap_id_1, swap_id_2 = rhs.split(" ")
                        for a, b in [(swap_id_1, swap_id_2), (swap_id_2, swap_id_1)]:
                            self.swaps.setdefault(a, []).append((int(frame_count), b))
                        continue
                    
                    # Normalize spaces around commas: "1, 2, 3" -> "1,2,3"
                    rhs = re.sub(r'\s*,\s*', ',', rhs)
                    
                    # Use Regex to parse tokens.
                    # We want to capture blocks like:
                    #   ID
                    #   ID*Range
                    #   ID,ID,ID
                    #   ID,ID*Range
                    # A comma acts as a grouper inside the ID part, but as a separator between blocks if it follows a range.
                    
                    # Regex logic:
                    # 1. Start with an ID component: [^,*\s]+ (anything not comma, *, space)
                    # 2. Optionally followed by more ID components starting with comma: (?:,[^,*\s]+)*
                    # 3. Optionally followed by a range component: (?:\*[\d\-]+)?
                    token_pattern = r'[^,*\s]+(?:,[^,*\s]+)*(?:\*[\d\-]+)?'
                    tokens = re.findall(token_pattern, rhs)
                    
                    for tok in tokens:
                        if not tok: continue
                        
                        id_list, start, end = self._parse_token(tok)
                        
                        for cid in id_list:
                            # Handle Global Assignment (no *)
                            if start is None:
                                self.id_to_name_global[cid] = name
                            # Handle Interval Assignment (has *)
                            else:
                                self.id_to_name_intervals.setdefault(cid, []).append((name, start, end))
                                
                    parsed_content.append([name, tokens])
                else:
                    name = f"UNK_{unknown_id_index}"
                    parsed_content.append([name, line.split(" ")])
                    unknown_id_index += 1
        
        if self.rewrite:
            content = ""
            for name, numbers in parsed_content:
                content += f"{name}: {' '.join(str(n) for n in numbers)}\n"
            with open(self.text_file_path, "w") as f:
                f.write(content)
        self.data = parsed_content
"""
class modification_reader:
    
    This class reads a modification file (stage 1).
    And structures what has been read in a standardised
    format usable for further modifications.
    
    def __init__(self, text_file_path, rewrite=False):  # default: do NOT rewrite
        self.text_file_path = text_file_path
        self.swaps = {}
        self.rewrite = rewrite
        self.read()

    def read(self):
        parsed_content = []
        unknown_id_index = 0
        with open(self.text_file_path, 'r') as text_file:
            text_content = text_file.read()
            splitted_content = text_content.split("\n")
            for i in splitted_content:
                if len(i) < 1: continue
                if (":" in i):  # name is present
                    name = i.split(": ")[0]
                    if name.upper() == "SWAP":  # storing the swap data
                        frame_count, swap_id_1, swap_id_2 = i.split(": ")[1].split(" ")
                        if (swap_id_1 in self.swaps.keys()): self.swaps[swap_id_1].append((frame_count, swap_id_2))
                        else: self.swaps[swap_id_1] = [(frame_count, swap_id_2)]
                        if (swap_id_2 in self.swaps.keys()): self.swaps[swap_id_2].append((frame_count, swap_id_1))
                        else: self.swaps[swap_id_2] = [(frame_count, swap_id_1)]
                        continue
                    parsed_content.append([name, i.split(": ")[1].split(" ")])
                else:
                    name = f"UNK_{unknown_id_index}"
                    parsed_content.append([name, i.split(" ")])
                    unknown_id_index += 1
                    
        # Only rewrite if explicitly requested
        if self.rewrite:
            content = ""
            for name, numbers in parsed_content:
                content = f"{content}{name}: {' '.join(str(n) for n in numbers)}\n"
            with open(self.text_file_path, "w") as f:
                f.write(content)

        self.data = parsed_content
"""
        
class data_writer:
    """
    writes the content of a block within 
    a destination file
    """
    def __init__(self, output_text_file_path):
        self.out_path = output_text_file_path
        # creates file if deosn't exist
        # erases file content if exists
        with open(self.out_path, "w") as temp:
            temp.close()
    
    def write(self, data):
        with open(self.out_path, "a") as output_file:
            for block in data:
                block_string = "#\n"
                for line in block:
                    block_string = f"{block_string}{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n"
                if block_string == "#\n": block_string = f"{block_string}\n"
                output_file.write(block_string)
            output_file.close()

def edit_raw_output(RTD_reader, M_reader):
    """
    returns the modified structure to be written in
    a new file. By taking into account the modifications
    made from the edit_stageX.txt file
    """
    modified_data = []

    for current_frame, block in enumerate(RTD_reader.data):
        new_block = []
        for line in block:
            class_id = line[0]
            label = ""
            keep = False

            # Apply swaps first
            if class_id in M_reader.swaps:
                for frame_index, other_id in M_reader.swaps[class_id]:
                    if current_frame >= int(frame_index):
                        class_id = other_id
                        break

            # Time-bounded assignment (highest priority)
            if class_id in M_reader.id_to_name_intervals:
                for name, start, end in M_reader.id_to_name_intervals[class_id]:
                    if start <= current_frame <= end:
                        label = name
                        keep = True
                        break  # first match wins

            # Global assignment (fallback)
            if not keep and class_id in M_reader.id_to_name_global:
                label = M_reader.id_to_name_global[class_id]
                keep = True

            # Legacy support: also honor parsed M_reader.data (for backward compatibility)
            if not keep:
                for name, ids in M_reader.data:
                    if class_id in ids:
                        keep = True
                        label = name
                        break

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
    label_text = f"{label}"#: {score:.2f}"
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

    # Triangle dimensions
    triangle_height = 20
    triangle_width = 20

    # Tip of triangle at the middle of the top edge of the bounding box
    tip_x = (x1 + x2) // 2
    tip_y = max(y1, text_height + triangle_height + 5)

    # Base is ABOVE the tip (Y decreases upward)
    base_y = tip_y - triangle_height
    half_base = triangle_width // 2

    pt_tip = (tip_x, tip_y)
    pt_left = (tip_x - half_base, base_y)
    pt_right = (tip_x + half_base, base_y)

    triangle = np.array([pt_tip, pt_left, pt_right], dtype=np.int32).reshape((-1, 1, 2))

    # Draw filled triangle
    cv2.fillPoly(image, [triangle], color)

    # Text position: centered horizontally, inside triangle
    text_x = tip_x - text_width // 2
    text_y = base_y + (triangle_height // 2) - triangle_height

    # Draw transparent rectangle using overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), color, -1)
    alpha = 0.6  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)

    # Draw label text on top
    cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)

    return image

def draw_bbox_from_file(file_path, input_video_path, output_video_path, annotation_type="bbox", draw_frame_count=False):
    """
    Draw the bounding boxes and class_id contained in <file_path>
    On the input video located at <input_video_path>
    The output video must be saved at <output_video_path>
    """
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

def extract_features_osnet(frame, bbox, model_feature_extraction):
    """Extrait les features d'un chimpanzé avec OSNet (permet à DeepSORT d'utiliser aussi l'apparence de l'objet en plus de la position, ...)"""
    transform = transforms.Compose([
        transforms.Resize((256, 128)),  # Taille attendue par OSNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x1, y1, x2, y2 = map(int, bbox)  # Convertir bbox en entiers
    chimp_img = frame[y1:y2, x1:x2]  # Extraire la région d'intérêt

    chimp_img = cv2.cvtColor(chimp_img, cv2.COLOR_BGR2RGB)  # Convertir en RGB
    chimp_img = transform(Image.fromarray(chimp_img)).unsqueeze(0)  # Appliquer transformations

    with torch.no_grad():
        feature = model_feature_extraction(chimp_img)  # Extraire les features

    # Convertir les features en vecteur 1D et s'assurer qu'ils ont une forme correcte
    feature = feature.cpu().numpy().flatten()  # Retourner un vecteur 512D
    return feature

# In your script (or ui_lib.py)

def perform_tracking(input_video_path, output_text_file_path, detection_model, confidence_threshold=0.5):
    """
    Performs tracking using YOLOv8's native ByteTrack.
    """
    
    # 1. Get Total Frames (needed for progress bar)
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    with open(output_text_file_path, "w") as f:
        
        results = detection_model.track(
            source=input_video_path,
            conf=confidence_threshold,
            iou=0.5,
            tracker="bytetrack.yaml",
            persist=True,
            stream=True,
            verbose=False
        )

        # 2. Add 'total=total_frames' to tqdm
        for frame_idx, result in enumerate(tqdm(results, total=total_frames, desc="ByteTrack Processing")):
            f.write("#\n")
            
            modified = False
            
            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    str_bbox = ' '.join(map(str, box))
                    f.write(f"{track_id} {str_bbox}\n")
                    modified = True
            
            if not modified:
                f.write("\n")

def mux_audio(original_video, processed_video, output_with_audio):
    """
    Combines processed video stream with the original audio stream.
    If output_with_audio equals processed_video, writes to a temp file
    and atomically replaces the processed file on success.
    """
    same_path = os.path.abspath(processed_video) == os.path.abspath(output_with_audio)
    temp_out = output_with_audio
    if same_path:
        suffix = ".mux.tmp.mp4"
        temp_fd, temp_out = tempfile.mkstemp(suffix=suffix, dir=os.path.dirname(output_with_audio))
        os.close(temp_fd)

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
    subprocess.run(cmd, check=True)

    if same_path:
        os.replace(temp_out, output_with_audio)