import cv2
import numpy as np
import os
from tqdm import tqdm


# --- 1. COLOR GENERATOR ---
def get_color(id_val):
    """
    Generates a consistent, unique color for each ID.
    """
    # Create a seed from the ID
    # 1. Convert to string
    val_str = str(id_val)
    # 2. Hash it to get a number
    val_hash = hash(val_str)
    # 3. Modulo to fit into 2^32 - 1 (NumPy requirement)
    seed_val = val_hash % (2**32 - 1)
    
    np.random.seed(seed_val)
    # Generate BGR color (0-255)
    color = np.random.randint(0, 255, size=3).tolist()
    return tuple(color)


# --- 2. ROBUST READER (FIXED SYNC) ---
def read_tracking_file(txt_path):
    """
    Parses the tracking file format:
    #
    id x1 y1 x2 y2
    ...
    
    CRITICAL FIX: 
    Does NOT skip empty blocks. It increments frame_idx for every '#' found.
    This prevents the "latency" and drift issue.
    """
    frame_data = {}
    
    with open(txt_path, 'r') as f:
        content = f.read()
        
    # Split by the frame delimiter '#'
    # If the file starts with '#', the first element is usually empty string
    blocks = content.split('#')
    
    # If the first block is empty (because file starts with #), discard it
    if len(blocks) > 0 and blocks[0].strip() == "":
        blocks = blocks[1:]
        
    frame_idx = 0
    
    for block in blocks:
        # DO NOT use "if not block.strip(): continue" -> This causes latency!
        
        detections = []
        if block.strip(): # Only parse lines if there is text
            lines = block.strip().split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5: 
                    tid = parts[0]
                    bbox = list(map(float, parts[1:5]))
                    detections.append({'id': tid, 'bbox': bbox})
        
        frame_data[frame_idx] = detections
        frame_idx += 1
        
    return frame_data


# --- 3. DRAWING FUNCTION (WITH COLORS) ---
def draw_tracks(video_path, txt_path, output_path, label_tag="Pipeline"):
    """
    Draws bounding boxes and IDs on the video.
    """
    # Load Data
    tracks = read_tracking_file(txt_path)
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=f"Rendering {label_tag}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Draw Label (Top Left)
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{label_tag} | Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw Tracks
        if frame_idx in tracks:
            for obj in tracks[frame_idx]:
                tid = obj['id']
                x1, y1, x2, y2 = map(int, obj['bbox'])
                
                # Get unique color for this ID
                color = get_color(tid)
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw ID Background
                id_text = f"ID: {tid}"
                (w, h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                
                # Draw ID Text (White or Black depending on contrast)
                cv2.putText(frame, id_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
        
    cap.release()
    out.release()
    pbar.close()
    return output_path


# --- 4. SIDE-BY-SIDE COMPARISON ---
def make_side_by_side(video1_path, video2_path, output_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Output width is doubled
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))
    
    print("Stitching videos together...")
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2: break
        
        # Concatenate horizontally
        combined = np.hstack((frame1, frame2))
        out.write(combined)
        
    cap1.release()
    cap2.release()
    out.release()
    print(f"Comparison saved to {output_path}")
