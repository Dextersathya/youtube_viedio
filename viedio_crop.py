import subprocess
import torch

def is_gpu_available():
    if torch.cuda.is_available():
        print("✅ PyTorch CUDA available:", torch.cuda.is_available())
        print("✅ GPU detected:", torch.cuda.get_device_name(0))
        return True
    else:
        print("⚠️ No GPU detected. FFmpeg will run on CPU (slower).")
        return False

def convert_to_shorts_zoom(input_path, output_path):
    use_gpu = is_gpu_available()

    # Filter: scale video to fill 1080x1920, crop excess to avoid black bars
    vf_filter = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"

    if use_gpu:
        cmd = [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-i", input_path,
            "-vf", vf_filter,
            "-c:v", "h264_nvenc",
            "-preset", "fast",
            "-cq", "28",
            "-c:a", "copy",
            output_path
        ]
    else:
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-c:a", "copy",
            output_path
        ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Conversion complete (zoomed): {output_path}")
    except subprocess.CalledProcessError:
        print("❌ FFmpeg failed. Make sure FFmpeg is installed and NVENC is supported.")

# Example usage
if __name__ == "__main__":
    convert_to_shorts_zoom("videoplayback.mp4", "output_shorts_zoom.mp4")
