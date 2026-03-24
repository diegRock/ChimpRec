import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# --- 1. SETUP PATHS TO CHIMPUFE ---
# We add the ChimpUFE folder to system path so we can import its modules
CHIMP_REPO_PATH = os.path.join(os.getcwd(), "ChimpUFE")
sys.path.append(CHIMP_REPO_PATH)

try:
    from src.face_embedder.vision_transformer import vit_base
    USE_LOCAL_MODEL = True
except ImportError:
    USE_LOCAL_MODEL = False
    print("⚠️ Local import failed. Will try timm/hub.")

# --- 2. THE MODEL WRAPPER (FIXED) ---
class ChimpUFEWrapper:
    def __init__(self, weights_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🔄 Loading ChimpUFE from {weights_path}...")
        
        # 1. Instantiate Model
        if USE_LOCAL_MODEL:
            self.model = vit_base(patch_size=14)
        else:
            raise ImportError("Could not import vit_base from src/face_embedder/. Check paths.")
        
        # 2. Load Checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if "teacher" in checkpoint: state_dict = checkpoint["teacher"]
        elif "model" in checkpoint: state_dict = checkpoint["model"]
        else: state_dict = checkpoint

        # --- 3. BLIND ZIP MATCHING (With LayerScale Filtering) ---
        new_state_dict = {}
        
        # A. Separate "Block" keys
        ckpt_block_keys = []
        model_block_keys = []
        
        # Get all keys from the initialized model
        model_keys_all = list(self.model.state_dict().keys())
        for k in model_keys_all:
            if "blocks." in k:
                model_block_keys.append(k)

        # Get all keys from checkpoint
        for k in state_dict.keys():
            k_clean = k.replace("backbone.", "").replace("module.", "")
            if "blocks." in k_clean:
                # CRITICAL FIX: Filter out LayerScale keys (ls1, ls2) which don't exist in local model
                if "ls1" in k_clean or "ls2" in k_clean:
                    continue
                ckpt_block_keys.append(k) # Keep original key name

        # B. Sort them to align them
        model_block_keys.sort()
        ckpt_block_keys.sort()
        
        # C. Verify Counts
        print(f"DEBUG: Model Block Params: {len(model_block_keys)}")
        print(f"DEBUG: Checkpoint Block Params (Filtered): {len(ckpt_block_keys)}")
        
        if len(model_block_keys) != len(ckpt_block_keys):
            print("⚠️ WARNING: Parameter counts still differ. Mismatches likely.")
        
        # Map 1-to-1
        limit = min(len(model_block_keys), len(ckpt_block_keys))
        for i in range(limit):
            m_key = model_block_keys[i]
            c_key = ckpt_block_keys[i]
            new_state_dict[m_key] = state_dict[c_key]

        # D. Handle Non-Block Keys
        for k, v in state_dict.items():
            k_clean = k.replace("backbone.", "").replace("module.", "")
            if "blocks." not in k_clean:
                new_state_dict[k_clean] = v

        # 4. Load State Dict
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        
        real_missing = [k for k in msg.missing_keys if "head" not in k]
        print(f"✅ Model Loaded. Real missing keys: {len(real_missing)}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, cv2_image):
        try:
            if cv2_image is None or cv2_image.size == 0: return None
            img = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(input_tensor)
            return embedding.cpu().numpy().flatten()
        except Exception:
            return None

    def compare_tracks(self, crops_A, crops_B):
        if not crops_A or not crops_B: return 0.0

        step_A = max(1, len(crops_A)//5)
        step_B = max(1, len(crops_B)//5)
        samples_A = crops_A[::step_A]
        samples_B = crops_B[::step_B]

        embeddings_A = [self.get_embedding(img) for img in samples_A]
        embeddings_B = [self.get_embedding(img) for img in samples_B]
        embeddings_A = [e for e in embeddings_A if e is not None]
        embeddings_B = [e for e in embeddings_B if e is not None]

        if not embeddings_A or not embeddings_B: return 0.0

        best_sim = -1.0
        for emb_a in embeddings_A:
            for emb_b in embeddings_B:
                norm_a = np.linalg.norm(emb_a)
                norm_b = np.linalg.norm(emb_b)
                if norm_a > 0 and norm_b > 0:
                    sim = np.dot(emb_a, emb_b) / (norm_a * norm_b)
                    if sim > best_sim: best_sim = sim
        return best_sim

# --- 3. HELPER FUNCTIONS ---

def get_track_crops(video_path, track_data, num_samples=5):
    """
    Extracts 'num_samples' images of the chimp from the video.
    We prioritize the UPPER BODY (Face area).
    """
    cap = cv2.VideoCapture(video_path)
    crops = []
    
    # Pick indices uniformly distributed over the track duration
    indices = np.linspace(0, len(track_data['frames'])-1, num_samples, dtype=int)
    
    for idx in indices:
        frame_id = track_data['frames'][idx]
        bbox = track_data['boxes'][idx] # x1, y1, x2, y2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret: continue
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Heuristic: The face is usually in the top 30-50% of the body box
        # We crop the top half of the box to reduce background noise
        h = y2 - y1
        face_y2 = y1 + int(h * 0.5) 
        
        # Ensure bounds
        x1, y1 = max(0, x1), max(0, y1)
        face_y2 = min(frame.shape[0], face_y2)
        x2 = min(frame.shape[1], x2)
        
        if x2 > x1 and face_y2 > y1:
            crop = frame[y1:face_y2, x1:x2]
            crops.append(crop)
            
    cap.release()
    return crops

def parse_tracking_file(filepath):
    """Reads standard tracking txt."""
    tracks = {} 
    with open(filepath, 'r') as f:
        lines = f.readlines()
    frame_idx = 0
    lines = [l.strip() for l in lines if l.strip()]
    if lines[0] == '#': lines = lines[1:]

    for line in lines:
        if line == '#':
            frame_idx += 1
            continue
        parts = line.split()
        if len(parts) < 5: continue
        tid = int(parts[0])
        bbox = list(map(float, parts[1:5]))
        if tid not in tracks: tracks[tid] = {'frames': [], 'boxes': [], 'id': tid}
        tracks[tid]['frames'].append(frame_idx)
        tracks[tid]['boxes'].append(bbox)
    return tracks, frame_idx

# --- 4. MAIN SMART STITCHER ---

def smart_stitch(tracker_file, video_file, output_file, weights_path):
    
    # Init Model
    reid_model = ChimpUFEWrapper(weights_path)
    
    print("📂 Parsing Tracks...")
    tracks, total_frames = parse_tracking_file(tracker_file)
    sorted_tracks = sorted(tracks.values(), key=lambda x: x['frames'][0])
    
    id_map = {t['id']: t['id'] for t in sorted_tracks}
    
    print(f"🔍 Analyzing {len(sorted_tracks)} tracklets for potential merges...")
    
    # --- UPDATED THRESHOLDS (MORE AGGRESSIVE) ---
    TIME_THRESH = 30 * 45  # 60 seconds (at 30fps)
    DIST_THRESH = 500     # 1000 pixels (Cross-screen movement)
    SIM_THRESH = 0.5      # 0.35 Cosine Similarity (Lowered from 0.45)
    
    merges_count = 0
    
    for i in tqdm(range(len(sorted_tracks))):
        track_a = sorted_tracks[i]
        
        # If this track was already merged into something else, get its root ID
        # (Optional: implement Union-Find for strict correctness, but direct map works for linear pass)
        
        end_frame_a = track_a['frames'][-1]
        end_box_a = track_a['boxes'][-1]
        center_a = [(end_box_a[0]+end_box_a[2])/2, (end_box_a[1]+end_box_a[3])/2]
        
        best_match = None
        best_score = -1
        
        # Look ahead
        for j in range(i + 1, len(sorted_tracks)):
            track_b = sorted_tracks[j]
            start_frame_b = track_b['frames'][0]
            
            # Optimization: Stop looking if too far in time
            if start_frame_b > end_frame_a + TIME_THRESH:
                break
            
            # Must start after A ends
            if start_frame_b <= end_frame_a: continue
            
            # 1. SPATIAL CHECK (Quick Filter)
            start_box_b = track_b['boxes'][0]
            center_b = [(start_box_b[0]+start_box_b[2])/2, (start_box_b[1]+start_box_b[3])/2]
            dist = np.sqrt((center_a[0]-center_b[0])**2 + (center_a[1]-center_b[1])**2)
            
            if dist > DIST_THRESH:
                continue # Too far away, skip visual check
                
            # 2. VISUAL CHECK (Slow, ChimpUFE)
            # Only run this if spatial check passed
            crops_a = get_track_crops(video_file, track_a)
            crops_b = get_track_crops(video_file, track_b)
            
            sim_score = reid_model.compare_tracks(crops_a, crops_b)
            
            # DEBUG: See what scores we are getting
            if sim_score > 0.2:
                # print(f"👀 potential match: {track_a['id']} & {track_b['id']} -> Score: {sim_score:.3f}")
                pass 

            if sim_score > SIM_THRESH and sim_score > best_score:
                best_score = sim_score
                best_match = track_b
        
        if best_match:
            # Merge logic
            root_id = id_map[track_a['id']]
            id_map[best_match['id']] = root_id
            best_match['id'] = root_id # Propagate forward
            merges_count += 1
            print(f"🔗 Merged ID {track_a['id']} -> {best_match['id']} (Sim: {best_score:.2f})")

    print(f"✅ Smart Stitching Complete. Merged {merges_count} tracklets.")
    
    # Write Output
    frame_data = {}
    for track in sorted_tracks:
        final_id = id_map[track['id']]
        for f_idx, box in zip(track['frames'], track['boxes']):
            if f_idx not in frame_data: frame_data[f_idx] = []
            box_str = " ".join(map(str, box))
            frame_data[f_idx].append(f"{final_id} {box_str}")
            
    with open(output_file, 'w') as f:
        for i in range(total_frames + 1):
            f.write("#\n")
            if i in frame_data:
                for line in frame_data[i]:
                    f.write(line + "\n")
    print(f"💾 Saved to {output_file}")

# --- EXECUTION ---
if __name__ == "__main__":
    
    # ⚠️ UPDATE THESE PATHS
    VIDEO_PATH = "../Tracking/Manual Correction/input/20241019 - 14h29.MP4"
    TRACKER_INPUT = "../Tracking/Manual Correction/output/temp/raw_output/20241019 - 14h29.txt"
    TRACKER_OUTPUT = "../Tracking/Manual Correction/output/temp/raw_output/20241019 - 14h29_smart_stitched.txt"
    WEIGHTS_PATH = "assets/weights/25-08-29T11-49-28_340k.pth"
    
    smart_stitch(TRACKER_INPUT, VIDEO_PATH, TRACKER_OUTPUT, WEIGHTS_PATH)