import os
import re
import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict

# Define the mapping from YOLO class IDs to human-readable labels
label_map = {
    0: 'Astra -Ally-', 
    1: 'Breach -Ally-', 
    2: 'Breach -Enemy-Yellow-', 
    3: 'Brim -Ally-', 
    4: 'Brim -Enemy-Yellow-', 
    5: 'Chamber -Ally-', 
    6: 'Chamber -Enemy-Purple-', 
    7: 'Chamber -Enemy-Yellow-', 
    8: 'Cypher -Ally-', 
    9: 'Cypher -Enemy-Yellow-', 
    10: 'Deadlock -Ally-', 
    11: 'Deadlock -Enemy-Yellow-', 
    12: 'Enemy', 
    13: 'Fade -Enemy-Yellow-', 
    14: 'Gekko -Enemy-Yellow-', 
    15: 'Harbor -Enemy-Yellow-', 
    16: 'Headshot splash', 
    17: 'Iso -Enemy-Yellow-', 
    18: 'Jett -Ally-', 
    19: 'Jett -Enemy-Purple-', 
    20: 'Jett -Enemy-Red-', 
    21: 'Jett -Enemy-Yellow-', 
    22: 'Kayo -Ally-', 
    23: 'Kayo -Enemy-Yellow-', 
    24: 'Killjoy -Enemy-Yellow-', 
    25: 'Neon -Ally-', 
    26: 'Neon -Enemy-Yellow-', 
    27: 'Omen -Ally-', 
    28: 'Omen -Enemy-Yellow-', 
    29: 'Phoenix -Ally-', 
    30: 'Phoenix -Enemy-Yellow-', 
    31: 'Raze -Ally-', 
    32: 'Raze -Enemy-Yellow-', 
    33: 'Reyna -Ally-', 
    34: 'Reyna -Enemy-Purple-', 
    35: 'Reyna -Enemy-Red-', 
    36: 'Reyna -Enemy-Yellow-', 
    37: 'Sage -Ally-', 
    38: 'Sage -Enemy-Purple-', 
    39: 'Sage -Enemy-Red-', 
    40: 'Sage -Enemy-Yellow-', 
    41: 'Skye -Ally-', 
    42: 'Skye -Enemy-Yellow-', 
    43: 'Sova -Ally-', 
    44: 'Sova -Enemy-Purple-', 
    45: 'Sova -Enemy-Yellow-', 
    46: 'Viper -Ally-', 
    47: 'Viper -Enemy-Purple-', 
    48: 'Viper -Enemy-Red-', 
    49: 'Viper -Enemy-Yellow-', 
    50: 'Yoru -Enemy-Yellow-',
}

# Classes that are often less relevant for high-level commentary
IGNORE_CLASSES = {
    "gun",
    "friendly_outline",
    "friendly_util_outline",
    "friendly_corpse_outline",
}

class CV:
    def run_yolo_on_frames(self, model_path, frame_dir):
        """
        Runs YOLOv8 detection on frames in chunks to manage memory and saves results.

        Args:
            model_path (str): Path to the trained YOLOv8 model weights.
            frame_dir (str): Directory containing the extracted frames.

        Returns:
            str: Path to the directory where YOLOv8 saved the label files.
        """
        model = YOLO(model_path)
        
        # 1. Setup the necessary paths and device
        yolo_project = "runs/detect"
        yolo_run_name = "shoutcast"
        yolo_label_dir = os.path.join(yolo_project, yolo_run_name, "labels")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # 2. Get all frame file names, sorted numerically (critical for correct chunking)
        all_frames = sorted(
            [f for f in os.listdir(frame_dir) if f.endswith('.jpg')], 
            key=lambda x: int(re.search(r"(\d+)", x).group(1)) # Ensure numerical sort
        )
        
        CHUNK_SIZE = 1000 # Process 1000 frames at a time
        
        print(f"Starting YOLO prediction on {len(all_frames)} frames in chunks of {CHUNK_SIZE}...")

        # 3. Process frames in chunks
        for i in range(0, len(all_frames), CHUNK_SIZE):
            chunk_files = [os.path.join(frame_dir, f) for f in all_frames[i:i + CHUNK_SIZE]]
            
            print(f"-> Processing chunk: Frames {i} to {i + len(chunk_files) - 1}")
            
            # Run prediction on the list of file paths
            results = model.predict(
                chunk_files,
                save_conf=True, 
                save_txt=True, 
                project=yolo_project, # Use defined paths
                name=yolo_run_name,
                exist_ok=True, # Allow overwriting/continuing in the same folder
                batch=4, # Use a small batch size for VRAM stability
                verbose=False # Keep the output clean
            )
            
            # 4. Aggressive Memory Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # Clear unused cached memory
            
            # Explicitly delete results and free GPU memory before the next chunk
            del results
        
        print("✅ YOLO detection done.")
        return yolo_label_dir


    def parse_yolo_outputs(self, label_dir, frame_dir):
        """
        Parses YOLOv8 output text files to extract detected objects, including their
        labels and bounding box coordinates (converted to pixel values).

        Args:
            label_dir (str): Directory containing YOLOv8 output .txt files.
            frame_dir (str): Directory containing the original frames (to get image dimensions).

        Returns:
            tuple: A tuple containing:
                - list: A sorted list of dictionaries, where each dictionary represents a frame
                        and contains its ID and a list of detected objects with labels and pixel bboxes.
                        Example: [{"frame_id": 0, "detections": [{"label": "enemy", "bbox": (x1, y1, x2, y2), "conf": 0.9}]}, ...]
                - int: The width of the video frames.
                - int: The height of the video frames.
        """
        frame_data = defaultdict(lambda: {"frame_id": None, "detections": []})

        frame_files = sorted(os.listdir(frame_dir))
        if not frame_files:
            print(
                "No frames found in frame_dir. Cannot determine dimensions. Defaulting to 1920x1080."
            )
            img_width, img_height = 1920, 1080
        else:
            first_frame_path = os.path.join(frame_dir, frame_files[0])
            img = cv2.imread(first_frame_path)
            if img is None:
                print(
                    f"Could not read image at {first_frame_path}. Cannot determine dimensions. Defaulting to 1920x1080."
                )
                img_width, img_height = 1920, 1080
            else:
                img_height, img_width, _ = img.shape

        for file in sorted(os.listdir(label_dir)):
            match = re.search(r"(\d+)", file)
            if not match:
                continue
            frame_id = int(match.group(1))
            frame_data[frame_id]["frame_id"] = frame_id

            with open(os.path.join(label_dir, file)) as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        class_id = int(float(parts[0]))
                        x_center, y_center, width, height, confidence = map(
                            float, parts[1:]
                        )

                        label = label_map[class_id]
                        if label in IGNORE_CLASSES:
                            continue

                        x_min = int((x_center - width / 2) * img_width)
                        y_min = int((y_center - height / 2) * img_height)
                        x_max = int((x_center + width / 2) * img_width)
                        y_max = int((y_center + height / 2) * img_height)

                        frame_data[frame_id]["detections"].append(
                            {
                                "label": label,
                                "bbox": (x_min, y_min, x_max, y_max),
                                "confidence": confidence,
                            }
                        )
                    except ValueError as e:
                        print(
                            f"Error parsing line in {file}: {line.strip()} - {e}"
                        )
                        continue

        sorted_frame_data = sorted(
            frame_data.values(), key=lambda x: x["frame_id"]
        )
        return sorted_frame_data, img_width, img_height