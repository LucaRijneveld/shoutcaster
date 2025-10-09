# tts.py (Cleaned version)
import os
import time
import tempfile
import numpy as np
from gtts import gTTS
from PIL import Image
from llm import LLM
from moviepy.editor import (
    AudioFileClip,
    concatenate_audioclips,
)
# Note: ImageSequenceClip and ColorClip are not needed here if only TTS/Audio logic lives here.
# Keep them if they are referenced elsewhere in the class, but for this correction, they are minimal.
from moviepy.audio.AudioClip import AudioArrayClip 


class TTS:
    def generate_tts_audio(self, text, output_filepath):
        """
        Generates TTS audio from text and saves it to a file.
        """
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(output_filepath)
            print(f"Generated TTS audio to {output_filepath}")
            return True
        except Exception as e:
            # gTTS errors often indicate network issues or rate limits
            print(f"Error generating TTS for '{text[:50]}...': {e}")
            return False
        
    def tts_with_retry(self, text, output_filepath, retries=5, wait=5):
        for i in range(retries):
            success = self.generate_tts_audio(text, output_filepath)
            if success:
                return True
            else:
                print(f"⚠️ TTS failed for segment. Retry {i+1}/{retries} in {wait}s...")
                time.sleep(wait)
                wait *= 2  # exponential backoff
        print("Too many retries for TTS, giving up.")
        return False

    def make_silence(self, duration, fps=44100):
        """Generate a silent AudioClip of a given duration in seconds."""
        # moviepy.audio.AudioClip.AudioArrayClip imported at top
        samples = np.zeros((int(duration * fps), 2), dtype=np.float32)
        return AudioArrayClip(samples, fps=fps)

    def process_video_segment(self, frame_dir, segment_frames, llava_processor, llava_model, img_width, img_height, segment_duration_s):
        """
        Generate LLM commentary + TTS for a segment and return audio clip.
        segment_frames: list of parsed frame dicts for this segment
        """
        if not segment_frames:
            return self.make_silence(segment_duration_s)  # nothing detected, silence

        # Pick last frame as context
        best_frame_id = segment_frames[-1]["frame_id"]
        best_frame_path = os.path.join(frame_dir, f"frame_{best_frame_id}.jpg")
        
        if not os.path.exists(best_frame_path):
            # fallback to blank image
            blank_image = Image.new("RGB", (img_width, img_height), color="black")
            temp_blank_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            blank_image.save(temp_blank_file.name)
            temp_blank_file.close()
            best_frame_path = temp_blank_file.name

        # Note: LLM.create_narrative_events and LLM.run_mllm are static or class methods 
        # that don't need a specific instance of LLM to be passed around.
        narrative_events, _ = LLM.create_narrative_events(segment_frames, img_width, img_height)
        
        # Pass the max duration to the prompt builder
        prompt = LLM.build_prompt(narrative_events, segment_duration_s) 
        commentary_text, _ = LLM.run_mllm(best_frame_path, prompt, llava_processor, llava_model)

        # Generate TTS immediately
        temp_audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        self.tts_with_retry(commentary_text, temp_audio_path)
        audio_clip = AudioFileClip(temp_audio_path)

        # Trim or pad audio to match segment duration
        if audio_clip.duration > segment_duration_s:
            audio_clip = audio_clip.subclip(0, segment_duration_s)
        elif audio_clip.duration < segment_duration_s:
            silence_padding = self.make_silence(segment_duration_s - audio_clip.duration)
            audio_clip = concatenate_audioclips([audio_clip, silence_padding])

        # Cleanup temp audio file immediately
        try:
            # Ensure the audio clip is closed to release the file handle
            audio_clip.close() 
            
            # Check if the path points to the temp file created earlier (important for robustness)
            if os.path.exists(temp_audio_path) and temp_audio_path.startswith(tempfile.gettempdir()): 
                os.remove(temp_audio_path)
            print(f"Cleaned up temporary audio file: {temp_audio_path}")

        except Exception as e:
            # Keep the print for debugging, but this error should now be rare
            print(f"Could not delete temporary file {temp_audio_path}: {e}")

        return audio_clip