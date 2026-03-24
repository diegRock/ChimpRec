import numpy as np

def parse_tracking_file(filepath):
    """Reads your custom format into a dictionary of tracks."""
    tracks = {} # {id: {'frames': [], 'boxes': []}}
    
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
        # box format: x1 y1 x2 y2
        bbox = list(map(float, parts[1:5]))
        
        if tid not in tracks:
            tracks[tid] = {'frames': [], 'boxes': [], 'id': tid}
        
        tracks[tid]['frames'].append(frame_idx)
        tracks[tid]['boxes'].append(bbox)
        
    return tracks, frame_idx

def stitch_tracks(input_file, output_file, time_thresh=30, dist_thresh=50):
    """
    time_thresh: Max frames to look ahead (e.g. 30 frames = 1 second)
    dist_thresh: Max pixels the chimp could have moved
    """
    tracks, total_frames = parse_tracking_file(input_file)
    
    # Sort tracks by their START frame
    sorted_tracks = sorted(tracks.values(), key=lambda x: x['frames'][0])
    
    # Stitching Logic
    # We map old_id -> new_id
    id_map = {t['id']: t['id'] for t in sorted_tracks}
    
    for i in range(len(sorted_tracks)):
        track_a = sorted_tracks[i]
        end_frame_a = track_a['frames'][-1]
        end_box_a = track_a['boxes'][-1] # [x1, y1, x2, y2]
        center_a = [(end_box_a[0]+end_box_a[2])/2, (end_box_a[1]+end_box_a[3])/2]
        
        # Look for the best matching future track
        best_match = None
        min_dist = float('inf')
        
        for j in range(i + 1, len(sorted_tracks)):
            track_b = sorted_tracks[j]
            start_frame_b = track_b['frames'][0]
            
            # If track B starts too late, stop looking (optimization)
            if start_frame_b > end_frame_a + time_thresh:
                break
                
            # Track B must start AFTER Track A ends
            if start_frame_b <= end_frame_a:
                continue
                
            # Check Distance
            start_box_b = track_b['boxes'][0]
            center_b = [(start_box_b[0]+start_box_b[2])/2, (start_box_b[1]+start_box_b[3])/2]
            
            dist = np.sqrt((center_a[0]-center_b[0])**2 + (center_a[1]-center_b[1])**2)
            
            if dist < dist_thresh and dist < min_dist:
                min_dist = dist
                best_match = track_b
        
        if best_match:
            # Merge!
            # Set the ID of track B to match track A (following the chain)
            root_id = id_map[track_a['id']]
            id_map[best_match['id']] = root_id
            
            # Update track B's ID in the list so it can continue the chain
            best_match['id'] = root_id

    # Rewrite the file
    print(f"Stitching Complete. Writing to {output_file}...")
    
    # We need to reconstruct the file frame by frame
    # Create a lookup: frame -> list of strings
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

if __name__ == "__main__":
    # RUN THIS on your ByteTrack output
    stitch_tracks(
        "output/temp/raw_output/20241019 - 14h29.txt", 
        "output/temp/raw_output/20241019 - 14h29_stitched.txt",
        #time_thresh=150, # 5 seconds (at 30fps)
        #dist_thresh=200  # Allow them to move 200 pixels
        time_thresh=900, # 30 seconds (at 30fps)
        dist_thresh=400  # Allow them to move 200 pixels
    )