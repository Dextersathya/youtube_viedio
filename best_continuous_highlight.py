"""
best_continuous_highlight.py

Given an input video, finds the best continuous 50-second segment using a multimodal scoring
(audio energy + voice activity + motion + faces + scene stability) and extracts it using ffmpeg.

Usage:
    python best_continuous_highlight.py "/path/to/input.mp4" "/path/to/output_50s.mp4"

Dependencies:
    pip install numpy opencv-python soundfile librosa tqdm
    pip install webrtcvad facenet-pytorch scenedetect
"""

import os
import sys
import math
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2
import soundfile as sf
import librosa

# Optional dependencies
try:
    import webrtcvad
except Exception:
    webrtcvad = None

try:
    from facenet_pytorch import MTCNN
    mtcnn_available = True
except Exception:
    mtcnn_available = False

try:
    from scenedetect import detect, ContentDetector
    pyscenedetect_available = True
except Exception:
    pyscenedetect_available = False

# ------------------ Config ------------------
STEP_S = 0.5           # analysis step in seconds
WINDOW_S = 50.0        # length of the extracted highlight
SLIDE_STEP_S = 2.0     # slide step for searching windows

# Feature weights
WEIGHTS = {
    "audio_rms": 1.0,
    "vad": 1.2,
    "motion": 1.0,
    "faces": 0.8,
    "scene_cut_penalty": 0.6
}

AUDIO_SR = 16000

FFMPEG_EXTRACT_CMD = (
    "ffmpeg -y -ss {start:.3f} -i \"{infile}\" -t {dur:.3f} "
    "-c:v libx264 -preset veryfast -crf 23 -c:a aac -b:a 128k \"{outfile}\""
)

# ------------------------------------------------
def run_cmd(cmd):
    print("RUN:", cmd)
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print("ffmpeg/command error:", p.stderr.decode("utf8", errors="ignore"))
        raise RuntimeError("Command failed")
    return p

def extract_audio_to_wav(video_path, out_wav_path, sr=AUDIO_SR):
    cmd = f"ffmpeg -y -i \"{video_path}\" -ac 1 -ar {sr} -vn -loglevel error \"{out_wav_path}\""
    run_cmd(cmd)
    return out_wav_path

def load_audio(audio_path, sr=AUDIO_SR):
    wav, sr_read = sf.read(audio_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr_read != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr_read, target_sr=sr)
        sr_read = sr
    return wav.astype(np.float32), sr_read

def compute_audio_features(wav, sr, step_s=STEP_S, vad_enabled=True):
    duration = wav.shape[0] / sr
    n_steps = int(math.ceil(duration / step_s))
    rms_vals = np.zeros(n_steps, dtype=float)
    vad_frac = np.zeros(n_steps, dtype=float)

    samples_per_step = int(step_s * sr)
    for i in range(n_steps):
        start = i * samples_per_step
        end = min(len(wav), start + samples_per_step)
        block = wav[start:end]
        rms_vals[i] = float(np.sqrt(np.mean(block ** 2) + 1e-9)) if block.size > 0 else 0.0

    if vad_enabled and webrtcvad is not None:
        vad = webrtcvad.Vad(2)
        frame_ms = 30
        frame_n = int(sr * (frame_ms / 1000.0))
        pcm16 = (wav * 32767).astype(np.int16).tobytes()
        bytes_per_frame = frame_n * 2
        n_frames = len(pcm16) // bytes_per_frame
        voiced_flags = []
        for f_idx in range(n_frames):
            frame_bytes = pcm16[f_idx * bytes_per_frame:(f_idx + 1) * bytes_per_frame]
            try:
                is_voiced = vad.is_speech(frame_bytes, sample_rate=sr)
            except Exception:
                is_voiced = False
            voiced_flags.append(1 if is_voiced else 0)
        frames_per_step = int((step_s * 1000) / frame_ms)
        for i in range(n_steps):
            fstart = i * frames_per_step
            fend = fstart + frames_per_step
            if fstart >= len(voiced_flags):
                vad_frac[i] = 0.0
            else:
                vad_frac[i] = float(np.sum(voiced_flags[fstart:fend]) / max(1, min(len(voiced_flags) - fstart, frames_per_step)))
    else:
        S = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=32)
        band_energy = np.mean(S, axis=0)
        frames_per_step = max(1, int((step_s * sr) / 512))
        vad_frac = np.array([np.mean(band_energy[i * frames_per_step:(i + 1) * frames_per_step]) for i in range(n_steps)])
        vad_frac = vad_frac / np.max(vad_frac) if np.max(vad_frac) > 0 else np.zeros_like(vad_frac)

    return rms_vals, vad_frac

def compute_visual_features(video_path, step_s=STEP_S, sample_frame_scale=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video with OpenCV")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0
    n_steps = int(math.ceil(duration / step_s))

    motion_vals = np.zeros(n_steps, dtype=float)
    face_counts = np.zeros(n_steps, dtype=float)

    if mtcnn_available:
        mtcnn = MTCNN(keep_all=True, device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    prev_gray = None
    for i in tqdm(range(n_steps), desc="visual analysis"):
        t = i * step_s
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if sample_frame_scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * sample_frame_scale), int(h * sample_frame_scale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            motion_vals[i] = 0.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals[i] = float(np.mean(diff))
        prev_gray = gray

        if mtcnn_available:
            try:
                boxes, _ = mtcnn.detect(frame)
                face_counts[i] = 0.0 if boxes is None else float(len(boxes))
            except Exception:
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                face_counts[i] = float(len(faces))
        else:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_counts[i] = float(len(faces))

    cap.release()
    return motion_vals, face_counts, duration

def normalize_array(x):
    if np.nanmax(x) - np.nanmin(x) < 1e-9:
        return np.zeros_like(x)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)

def find_best_window(rms, vadf, motion, faces, step_s=STEP_S, window_s=WINDOW_S, slide_step_s=SLIDE_STEP_S):
    n = max(len(rms), len(vadf), len(motion), len(faces))
    def pad(a, n): return np.concatenate([a, np.zeros(n - len(a))]) if len(a) < n else a[:n]
    rms, vadf, motion, faces = map(lambda a: pad(np.array(a), n), [rms, vadf, motion, faces])
    nr, nv, nm, nf = map(normalize_array, [rms, vadf, motion, faces])
    combined = (WEIGHTS["audio_rms"] * nr + WEIGHTS["vad"] * nv + WEIGHTS["motion"] * nm + WEIGHTS["faces"] * nf)

    window_steps = int(round(window_s / step_s))
    slide_steps = int(round(slide_step_s / step_s))
    if window_steps >= n:
        return 0.0, combined.sum()

    scores, starts = [], []
    for start_idx in range(0, n - window_steps + 1, max(1, slide_steps)):
        end_idx = start_idx + window_steps
        score = float(np.mean(combined[start_idx:end_idx]))
        scores.append(score)
        starts.append(start_idx * step_s)
    best_idx = int(np.argmax(scores))
    return starts[best_idx], scores[best_idx]

def extract_clip_ffmpeg(input_video, start_s, dur_s, out_path):
    cmd = FFMPEG_EXTRACT_CMD.format(start=start_s, infile=input_video, dur=dur_s, outfile=out_path)
    run_cmd(cmd)
    return out_path

def main(infile, outfile):
    tmpdir = tempfile.mkdtemp()
    wav_path = os.path.join(tmpdir, "audio.wav")
    print("Extracting audio...")
    extract_audio_to_wav(infile, wav_path, sr=AUDIO_SR)
    print("Loading audio...")
    wav, sr = load_audio(wav_path, sr=AUDIO_SR)
    print("Computing audio features...")
    rms, vadf = compute_audio_features(wav, sr, step_s=STEP_S, vad_enabled=(webrtcvad is not None))
    print("Computing visual features...")
    motion, faces, duration = compute_visual_features(infile, step_s=STEP_S, sample_frame_scale=0.6)

    print("Searching best window...")
    best_start_s, best_score = find_best_window(rms, vadf, motion, faces, step_s=STEP_S, window_s=WINDOW_S, slide_step_s=SLIDE_STEP_S)
    print(f"Best window starts at {best_start_s:.2f}s (score {best_score:.4f})")

    try:
        probe = subprocess.run(
            f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{infile}\"",
            shell=True, capture_output=True
        )
        total_dur = float(probe.stdout.decode().strip())
    except Exception:
        total_dur = duration

    best_start_s = max(0.0, min(best_start_s, max(0.0, total_dur - WINDOW_S)))

    print("Extracting the 50s clip with ffmpeg...")
    extract_clip_ffmpeg(infile, best_start_s, WINDOW_S, outfile)
    print("DONE. Output:", outfile)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python best_continuous_highlight.py input.mp4 output_50s.mp4")
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    main(infile, outfile)
