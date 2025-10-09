# main.py (Updated Version)

import os
import argparse
import time
import tempfile
import re
import cv2
import torch
import threading
import numpy as np
from PIL import Image

# Import the necessary modules
from cv import CV
from llm import LLM 
from tts import TTS 

# Import LLaVA dependencies
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration 
)

# Import moviepy components needed for the final steps
from moviepy.editor import (
    AudioFileClip, 
    concatenate_audioclips, 
    ImageSequenceClip
)

# --- Configuration ---
# NOTE: Replace these with your actual paths/values
LLAVA_MODEL_PATH = r"models\llava-7b" 
YOLO_MODEL_PATH = r"runs\detect\train5\weights\best.pt" 
INPUT_VIDEO_PATH = r"videoplayback.mp4" 
OUTPUT_VIDEO_PATH = r"valorant_commentary_video.mp4"
FRAME_EXTRACTION_DIR = r"frames"
FPS_FOR_ANALYSIS = 1 # The value previously referred to as 'extraction_fps'
SEGMENT_DURATION_SECONDS = 100 

def extract_frames(video_path, output_dir, fps): # Use 'fps' argument now
    """
    Extracts frames from the video at a specified FPS.
    Returns the total number of frames extracted, width, and height.
    """
    print(f"🎬 Starting frame extraction from {video_path} at {fps} FPS...")
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Add error checking for robust code
        raise IOError(f"Cannot open video file: {video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if orig_fps == 0:
        orig_fps = 30 # Default safety value
          
    # Re-introduce the frame interval calculation:
    frame_interval = max(1, int(orig_fps / fps)) 
    
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only save frame when frame_count is a multiple of frame_interval
        if frame_count % frame_interval == 0: 
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"✅ Extracted {extracted_count} frames to {output_dir}")
    return extracted_count, frame_width, frame_height


def main():
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Input video not found at {INPUT_VIDEO_PATH}")
        return

    # --- 1. Frame Extraction ---
    try:
        total_extracted_frames, img_width, img_height = extract_frames(
            INPUT_VIDEO_PATH, FRAME_EXTRACTION_DIR, FPS_FOR_ANALYSIS
        )
        if total_extracted_frames == 0:
            print("Error: No frames were extracted.")
            return
    except IOError as e:
        print(f"Critical Error during frame extraction: {e}")
        return

    # --- 2. Initialize Models and Classes ---
    print("⏳ Initializing LLaVA Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    quantization_config = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16
    }

    try:
        llava_processor = AutoProcessor.from_pretrained(
            LLAVA_MODEL_PATH, 
            use_fast=True 
        )
        
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_PATH, 
            device_map="auto", 
            quantization_config=quantization_config,
        )
        
    except Exception as e:
        print(f"Error loading LLaVA model: {e}")
        print("Please ensure you have installed the necessary dependencies (e.g., bitsandbytes, accelerate).")
        return
        
    cv_processor = CV()
    tts_generator = TTS() # TTS instance created here

    # --- 3. Computer Vision (YOLO) Processing ---
    print("\n🔍 Running YOLO detection...")
    try:
        yolo_label_dir = cv_processor.run_yolo_on_frames(YOLO_MODEL_PATH, FRAME_EXTRACTION_DIR)
        
        # 🌟 NEW: Clear CUDA cache immediately after running YOLO (if using GPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Parse results
        parsed_data, img_width, img_height = cv_processor.parse_yolo_outputs(
            yolo_label_dir, FRAME_EXTRACTION_DIR
        )
        print(f"✅ Parsed {len(parsed_data)} frames of YOLO detections.")

    except Exception as e:
        print(f"Critical Error during CV processing: {e}")
        return

    # --- 4. LLM and TTS Segmentation Loop (Execution Logic from tts.py is now HERE) ---
    print("\n🎤 Generating Commentary and TTS Audio...")

    # The formerly 'undefined' variables are now defined in this scope!
    frames_per_segment = SEGMENT_DURATION_SECONDS * FPS_FOR_ANALYSIS
    all_segments_audio = []

    for seg_start in range(0, len(parsed_data), frames_per_segment):
        segment_frames = parsed_data[seg_start:seg_start + frames_per_segment]
        print(f"\nProcessing segment: Frames {seg_start} to {seg_start + len(segment_frames)-1}...")
        
        # Call the process_video_segment METHOD on the tts_generator INSTANCE
        segment_audio = tts_generator.process_video_segment(
            FRAME_EXTRACTION_DIR, 
            segment_frames, 
            llava_processor, 
            llava_model, 
            img_width, 
            img_height, 
            SEGMENT_DURATION_SECONDS
        )
        all_segments_audio.append(segment_audio)
        print(f"Segment audio clip generated with duration: {segment_audio.duration:.2f}s")


    # --- 5. Video Stitching (Execution Logic from tts.py is now HERE) ---
    print("\n🎥 Stitching video and audio...")
    try:
        # Get all frame paths, sorted numerically
        frame_paths_sorted = [
            os.path.join(FRAME_EXTRACTION_DIR, f) 
            for f in sorted(os.listdir(FRAME_EXTRACTION_DIR), key=lambda x: int(re.search(r"(\d+)", x).group(1)))
            if f.endswith(".jpg")
        ]
        
        # FPS_FOR_ANALYSIS is used here as the variable that was 'extraction_fps'
        video_clip = ImageSequenceClip(frame_paths_sorted, fps=FPS_FOR_ANALYSIS)
        
        if all_segments_audio:
            final_audio_clip = concatenate_audioclips(all_segments_audio)
        else:
            print("Warning: No audio segments generated. Creating silence for video.")
            final_audio_clip = tts_generator.make_silence(video_clip.duration)
        
        final_video = video_clip.set_audio(final_audio_clip)

        final_video.write_videofile(
            OUTPUT_VIDEO_PATH, 
            codec="libx264", 
            audio_codec="aac",
            fps=FPS_FOR_ANALYSIS 
        )
        print(f"\n🎉 Successfully created final commentary video: {OUTPUT_VIDEO_PATH}")

    except Exception as e:
        print(f"Critical Error during video stitching: {e}")
        
    # --- 6. Cleanup (Optional) ---

if __name__ == "__main__":
    main()