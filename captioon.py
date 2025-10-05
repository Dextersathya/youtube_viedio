# Install dependencies
!pip install torch torchvision transformers accelerate timm opencv-python pillow

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEFeatureExtractor,
    CLIPProcessor,
    CLIPModel,
    pipeline
)

# =========================
# 1. Extract frames from video
# =========================
video_path = "/content/This is why NFS Unbound online is so bad.mp4"
num_frames_to_extract = 16  # VideoMAE expects 16 frames
frames = []

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames == 0:
    raise ValueError("Could not read video file or video has no frames.")

frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)

for index in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Could not read frame at index {index}. Using last available frame.")
        if frames:
            frames.append(frames[-1])
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(Image.fromarray(frame))

cap.release()

if len(frames) < num_frames_to_extract:
    while len(frames) < num_frames_to_extract:
        frames.append(frames[-1])

# =========================
# 2. Use CLIP for scene description (zero-shot)
# =========================
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Example candidate scene descriptions (can expand)
candidate_scenes = ["racing in a car", "shooting in a game", "walking in game", "fighting enemies"]

inputs = clip_processor(
    text=candidate_scenes,
    images=frames[0],  # use first frame
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    scene_index = logits_per_image.argmax().item()
scene_description = candidate_scenes[scene_index]

print("Scene Description:", scene_description)

# =========================
# 3. Use VideoMAE for action recognition
# =========================
feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
video_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

processed_video = feature_extractor(frames, return_tensors="pt")["pixel_values"]  # (1,3,T,H,W)
print(f"Video tensor shape: {processed_video.shape}")

with torch.no_grad():
    video_outputs = video_model(processed_video)
    predicted_class = video_outputs.logits.argmax(-1).item()
action_label = video_model.config.id2label[predicted_class]

print("Action Label:", action_label)

# =========================
# 4. Generate caption using instruction-tuned model
# =========================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

text_model_name = "google/flan-t5-base"  # Instruction-tuned, better for captions
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSeq2SeqLM.from_pretrained(text_model_name)

# =========================
# 4. Generate a catchy/funny caption
# =========================
prompt = f"""
Write a short, fun, catchy caption for a gameplay video.
Make it exciting and social-media ready.
Examples:
1. Scene: player running, Action: sprinting -> "Catch me if you can! 🏃💨"
2. Scene: player jumping, Action: jumping -> "Jumping into the weekend like... 🚀"

Now write one for:
Scene: {scene_description}
Action: {action_label}
Caption:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = text_model.generate(**inputs, max_length=50, do_sample=True, top_p=0.9, top_k=50)
caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Caption:", caption)

