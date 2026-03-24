import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score # NEW IMPORT FOR ML METRICS

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

def estimate_optimal_k(valid_tracks, dist_matrix, max_possible_k=20):
    """
    Automatically deduces the optimal number of chimps (K) in the video 
    using a mix of physical constraints and Silhouette clustering analysis.
    """
    n_tracks = len(dist_matrix)
    
    # 1. PHYSICAL LOWER BOUND (Concurrent Tracks)
    # Count the maximum number of tracks that exist in the exact same frame.
    frame_occupancy = {}
    for track in valid_tracks:
        for f in track['frames']:
            frame_occupancy[f] = frame_occupancy.get(f, 0) + 1
            
    min_k_physical = max(frame_occupancy.values()) if frame_occupancy else 1
    print(f"🐒 Physical constraint: Max concurrent tracks in a frame is {min_k_physical}. Therefore, K >= {min_k_physical}.")
    
    # If we only have a few tracks, handle edge cases
    max_search_k = min(max_possible_k, n_tracks - 1)
    if n_tracks <= 2: return n_tracks
    if min_k_physical >= max_search_k: return max_search_k
    
    # 2. SILHOUETTE ANALYSIS (Finding the most cohesive K)
    best_k = min_k_physical
    best_score = -1.0
    
    # Silhouette score requires at least 2 clusters
    start_k = max(2, min_k_physical)
    
    print(f"📈 Running Silhouette Analysis for K between {start_k} and {max_search_k}...")
    
    for k in range(start_k, max_search_k + 1):
        clusterer = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels = clusterer.fit_predict(dist_matrix)
        
        # Calculate how well separated the clusters are (Score from -1 to 1)
        # Using precomputed distance matrix
        score = silhouette_score(dist_matrix, labels, metric='precomputed')
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"🏆 Automated K Deduction Complete: Optimal K = {best_k} (Silhouette Score: {best_score:.3f})")
    return best_k


def smart_stitch_global_automated(tracker_file, video_file, output_file, weights_path):
    """
    The new main function that doesn't require hardcoding known_chimp_count.
    Includes 'Ghost Removal' to clean up stale tracks.
    """
    # Init Model
    reid_model = ChimpUFEWrapper(weights_path)
    
    print("📂 Parsing Tracks...")
    tracks, total_frames = parse_tracking_file(tracker_file)
    track_list = sorted(tracks.values(), key=lambda x: x['frames'][0])
    
    # --- CHANGED LOGIC START ---
    
    # 1. Define what is "Stitchable" (High Quality) vs "Keep As Is" (Low Quality)
    STITCH_THRESHOLD = 30  # Only attempt to merge tracks longer than this
    
    long_tracks = [t for t in track_list if len(t['frames']) > STITCH_THRESHOLD]
    short_tracks = [t for t in track_list if len(t['frames']) <= STITCH_THRESHOLD]
    
    print(f"🔍 Found {len(track_list)} total tracks.")
    print(f"   - Stitching candidates (Long): {len(long_tracks)}")
    print(f"   - Preserved noise/short segments: {len(short_tracks)}")
    
    # 1. Extract Representative Embeddings
    print("📸 Extracting embeddings for all valid tracks...")
    track_embeddings = []
    valid_track_indices = []
    
    for idx, track in enumerate(tqdm(long_tracks)):
        crops = get_track_crops(video_file, track, num_samples=5)
        embs = [reid_model.get_embedding(c) for c in crops]
        embs = [e for e in embs if e is not None]
        
        if len(embs) > 0:
            avg_emb = np.mean(embs, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb) # Normalize
            track_embeddings.append(avg_emb)
            valid_track_indices.append(idx)

    track_embeddings = np.array(track_embeddings)
    
    if len(track_embeddings) < 2:
        print("Not enough tracks with valid faces to cluster. Exiting.")
        return

    # 2. Compute Similarity & Distance Matrix
    sim_matrix = np.dot(track_embeddings, track_embeddings.T) 
    dist_matrix = 1.0 - sim_matrix
    dist_matrix[dist_matrix < 0] = 0 
    
    # 3. AUTOMATED K-ESTIMATION
    valid_long_tracks = [long_tracks[i] for i in valid_track_indices]
    optimal_k = estimate_optimal_k(valid_long_tracks, dist_matrix, max_possible_k=20)
    
    # 4. Clustering
    print(f"🧠 Applying Global Clustering with K={optimal_k}...")
    clusterer = AgglomerativeClustering(n_clusters=optimal_k, metric='precomputed', linkage='average')
    labels = clusterer.fit_predict(dist_matrix)

    # 5. Apply New IDs
    id_map = {}
    
    # Map the CLUSTERED tracks
    for i, list_idx in enumerate(valid_track_indices):
        track = long_tracks[list_idx]
        new_id = int(labels[i]) + 1
        id_map[track['id']] = new_id

    # Map the UNCLUSTERED (Long tracks that failed embedding)
    # Give them unique IDs starting after the clusters
    max_cluster_id = optimal_k + 1
    
    for idx, t in enumerate(long_tracks):
        if t['id'] not in id_map: 
            id_map[t['id']] = max_cluster_id
            max_cluster_id += 1

    # Map the SHORT TRACKS (Preserve them!)
    # CRITICAL: Do not drop them. Give them unique IDs.
    for t in short_tracks:
        id_map[t['id']] = max_cluster_id
        max_cluster_id += 1

    # 6. Write Output with FRAME SORTING
    # This ensures frames are written in order (0, 1, 2...)
    # The previous code wrote randomly if track order was mixed
    frame_data = {}
    all_tracks = long_tracks + short_tracks
    
    for track in all_tracks:
        final_id = id_map.get(track['id'], track['id'])
        for f_idx, box in zip(track['frames'], track['boxes']):
            if f_idx not in frame_data: frame_data[f_idx] = []
            box_str = " ".join(map(str, box))
            frame_data[f_idx].append(f"{final_id} {box_str}")
            
    with open(output_file, 'w') as f:
        # Loop exactly through 0 to total_frames
        # This prevents "skipping" or "out of order" frames causing visual jitter
        for i in range(total_frames + 1):
            f.write("#\n")
            if i in frame_data:
                for line in frame_data[i]:
                    f.write(line + "\n")
    print(f"💾 Saved dynamically clustered tracks to {output_file}")

# --- EXECUTION ---
if __name__ == "__main__":
    VIDEO_PATH = "../Tracking/Manual Correction/input/20241019 - 14h29.MP4"
    TRACKER_INPUT = "../Tracking/Manual Correction/output/temp/raw_output/20241019 - 14h29.txt"
    TRACKER_OUTPUT = "../Tracking/Manual Correction/output/temp/raw_output/20241019 - 14h29_v7_bytetrack_auto_clusteredv2.txt"
    WEIGHTS_PATH = "assets/weights/25-08-29T11-49-28_340k.pth"
    
    # We no longer pass KNOWN_COUNT!
    smart_stitch_global_automated(TRACKER_INPUT, VIDEO_PATH, TRACKER_OUTPUT, WEIGHTS_PATH)