"""
Download SmolLM2-135M model files directly from HuggingFace Hub to local directory.
"""
import os
import sys
from huggingface_hub import snapshot_download

def download_smollm2_135m(local_dir: str = "models/SmolLM2-135M"):
    repo_id = "HuggingFaceTB/SmolLM2-135M"
    print(f"[*] Downloading {repo_id} into {local_dir}...")
    
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors"
        ],
        local_dir_use_symlinks=False
    )
    print(f"[+] Download complete! Files saved in: {local_dir}")
    for f in sorted(os.listdir(local_dir)):
        size_mb = os.path.getsize(os.path.join(local_dir, f)) / (1024 * 1024)
        print(f"  - {f}: {size_mb:.2f} MB")

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "models/SmolLM2-135M"
    download_smollm2_135m(out_dir)
