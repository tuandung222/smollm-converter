"""
Convert SmolLM2 to LiteRT (TensorFlow Lite) format from scratch.
Uses clean Edge architecture and Google AI Edge Torch converter.
"""
from __future__ import annotations
import json
import os
import sys
import torch
from safetensors import safe_open
import litert_torch

# Ensure local dir is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_model import SmolLMEdgeModel


def convert_smollm_to_litert(
    model_dir: str = "models/SmolLM2-135M",
    output_tflite_path: str = "smollm2_135m.tflite",
    seq_len: int = 16
):
    print(f"[*] Starting SmolLM2 -> LiteRT (.tflite) conversion...")
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("[*] Initializing Edge PyTorch model architecture...")
    model = SmolLMEdgeModel(config)
    model.eval()

    # Load weights from safetensors
    print(f"[*] Loading weights from {weights_path}...")
    state_dict = {}
    with safe_open(weights_path, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            state_dict[k] = sf.get_tensor(k)

    # Remap HuggingFace names to EdgeModel names
    mapped_dict = {
        "embed_tokens.weight": state_dict["model.embed_tokens.weight"],
        "norm.weight": state_dict["model.norm.weight"],
    }
    if "lm_head.weight" in state_dict:
        mapped_dict["lm_head.weight"] = state_dict["lm_head.weight"]
    else:
        # Tied embeddings
        mapped_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    for i in range(config["num_hidden_layers"]):
        mapped_dict.update({
            f"layers.{i}.input_layernorm.weight": state_dict[f"model.layers.{i}.input_layernorm.weight"],
            f"layers.{i}.self_attn.q_proj.weight": state_dict[f"model.layers.{i}.self_attn.q_proj.weight"],
            f"layers.{i}.self_attn.k_proj.weight": state_dict[f"model.layers.{i}.self_attn.k_proj.weight"],
            f"layers.{i}.self_attn.v_proj.weight": state_dict[f"model.layers.{i}.self_attn.v_proj.weight"],
            f"layers.{i}.self_attn.o_proj.weight": state_dict[f"model.layers.{i}.self_attn.o_proj.weight"],
            f"layers.{i}.post_attention_layernorm.weight": state_dict[f"model.layers.{i}.post_attention_layernorm.weight"],
            f"layers.{i}.mlp.gate_proj.weight": state_dict[f"model.layers.{i}.mlp.gate_proj.weight"],
            f"layers.{i}.mlp.up_proj.weight": state_dict[f"model.layers.{i}.mlp.up_proj.weight"],
            f"layers.{i}.mlp.down_proj.weight": state_dict[f"model.layers.{i}.mlp.down_proj.weight"],
        })

    model.load_state_dict(mapped_dict, strict=False)
    print("[+] Model loaded successfully!")

    # Prepare sample input for conversion tracing
    sample_input = torch.zeros((1, seq_len), dtype=torch.long)
    print(f"[*] Tracing and converting to LiteRT with sample input shape {tuple(sample_input.shape)}...")

    edge_model = litert_torch.convert(model, (sample_input,))
    print(f"[*] Exporting LiteRT FlatBuffer to {output_tflite_path}...")
    edge_model.export(output_tflite_path)

    size_mb = os.path.getsize(output_tflite_path) / (1024 * 1024)
    print(f"[+] Successfully generated LiteRT model: {output_tflite_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "models/SmolLM2-135M"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "smollm2_135m.tflite"
    seq = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    convert_smollm_to_litert(src_dir, out_file, seq)
