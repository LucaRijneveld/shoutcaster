import torch
import os
from PIL import Image

IGNORE_CLASSES = {
    "gun",
    "friendly_outline",
    "friendly_util_outline",
    "friendly_corpse_outline",
}

class LLM:
    @staticmethod
    def create_narrative_events(parsed_frames_data, img_width, img_height, window_size=300, fps=1):
        """
        Summarizes detections across a window of frames into a compact narrative
        for the LLM. Instead of frame-by-frame logs, it merges all detections into
        a single tactical snapshot.
        """
        if not parsed_frames_data:
            return [], None

        # Focus on the last window_size frames
        window_data = parsed_frames_data[-window_size:]

        all_labels = []
        positions = []
        best_frame_id_for_image = window_data[-1]["frame_id"]  # last frame as context

        for frame_entry in window_data:
            for det in frame_entry["detections"]:
                label = det["label"]

                # Skip ignored classes
                if label in IGNORE_CLASSES:
                    continue

                # Map position (rough: left/center/right + top/middle/bottom)
                x1, y1, x2, y2 = det["bbox"]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                horizontal = (
                    "left" if center_x < img_width / 3
                    else "right" if center_x > 2 * img_width / 3
                    else "center"
                )
                vertical = (
                    "top" if center_y < img_height / 3
                    else "bottom" if center_y > 2 * img_height / 3
                    else "middle"
                )

                all_labels.append(label)
                positions.append(f"{label} at {horizontal}-{vertical}")

        # Deduplicate detections
        unique_labels = sorted(set(all_labels))
        unique_positions = sorted(set(positions))

        summary = []
        if unique_labels:
            summary.append("Agents and objects detected: " + ", ".join(unique_labels))
        if unique_positions:
            summary.append("Positions: " + "; ".join(unique_positions))

        # One compact event description for the full window
        narrative_events = [" | ".join(summary)] if summary else []

        return narrative_events, best_frame_id_for_image

    @staticmethod # assuming you applied the staticmethod fix from earlier
    def build_prompt(events_descriptions, max_duration_s, video_fps=1):
        """
        Constructs the prompt for the LLaVA model, optimized for concise Valorant commentary.
        """
        context = (
            "You are an expert Valorant esports analyst. "
            "You already know this is a Valorant match. "
            "Your task is to provide short, tactical commentary "
            f"about the events below. The entire commentary should last no longer than {max_duration_s} seconds. " # <-- NEW: Limit the length
            "Focus on agent actions, eliminations, spike events, and positioning. "
            "Keep the commentary concise (2–3 sentences maximum). "
            "Do not mention frames, images, screenshots, or the fact that you are analyzing visuals.\n\n"
            "Recent Game Events (aggregated across video frames at "
            f"{video_fps} FPS):\n"
        )
        if not events_descriptions:
            return (
                context
                + "In this segment, no agents, spike, or utility were detected. "
                + "Provide occasional commentary about what this absence might mean tactically "
                "(e.g., rotations, slow play, defaults), but keep it concise."
            )   
        else:
            return (
            context
            + "\n".join(events_descriptions)
            + "\n\nProvide a concise tactical summary of what happened and its significance:"
        )

    @staticmethod
    def get_frame_path_from_id(frame_id, frame_dir):
        """Constructs the full path to a frame image given its ID and directory."""
        return os.path.join(frame_dir, f"frame_{frame_id}.jpg")

    @staticmethod
    def run_mllm(image_path, prompt, processor, model):
        """
        Runs the LLaVA model with the given image and prompt to generate commentary.
        """
        print("Running LLaVA on selected frame + prompt...")

        image = Image.open(image_path).convert("RGB")

        # Make sure prompt includes the <image> token explicitly
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt  

        # Processor will insert image features aligned with <image> token
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        # Generate with limited tokens (keep it concise)
        output = model.generate(**inputs, max_new_tokens=60)

        result = processor.batch_decode(output, skip_special_tokens=True)[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result, 0.0  # keep your llm timing if needed