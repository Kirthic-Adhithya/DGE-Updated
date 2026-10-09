"""Pre-download the Hugging Face models DGE needs, with retry/resume (the connection resets occasionally).

Only safetensors + configs are fetched; diffusers 0.19 doesn't need the .bin / .ckpt duplicates.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so threestudio / gaussiansplatting import
import time

from huggingface_hub import snapshot_download

JOBS = [
    ("timbrooks/instruct-pix2pix", ["*.json", "*.txt", "*.safetensors"]),
    ("stable-diffusion-v1-5/stable-diffusion-v1-5", ["*.json", "*.txt", "text_encoder/*.safetensors"]),
    ("CompVis/stable-diffusion-v1-4", ["scheduler/*.json"]),
]

for repo, patterns in JOBS:
    for attempt in range(1, 21):
        try:
            path = snapshot_download(repo, allow_patterns=patterns, max_workers=2)
            print(f"OK {repo} -> {path}", flush=True)
            break
        except Exception as e:  # noqa: BLE001 - network flake, just retry
            print(f"retry {attempt} {repo}: {type(e).__name__}", flush=True)
            time.sleep(5)
    else:
        raise SystemExit(f"FAILED {repo}")
print("PREDOWNLOAD_DONE")
