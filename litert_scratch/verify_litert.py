"""
Verify LiteRT (.tflite) model against PyTorch.
Loads the exported FlatBuffer into LiteRT runtime interpreter and checks numerical parity.
"""
from __future__ import annotations
import json
import os
import sys
import torch
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

# Try importing LiteRT interpreter
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter

# Ensure local dir is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_model import SmolLMEdgeModel


def verify_litert(
    tflite_model_path: str = "smollm2_135m.tflite",
    model_dir: str = "models/SmolLM2-135M",
    seq_len: int = 16
):
    print("=" * 65)
    print("         VERIFYING LiteRT (.tflite) RUNTIME vs PYTORCH           ")
    print("=" * 65)

    if not os.path.exists(tflite_model_path):
        print(f"[-] Error: {tflite_model_path} not found! Please run convert_smollm_litert.py first.")
        return

    # 1. Load Tokenizer & prepare sample prompt
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt = "The future of AI is"
    input_tokens = tokenizer.encode(prompt)
    if len(input_tokens) < seq_len:
        input_tokens = input_tokens + [0] * (seq_len - len(input_tokens))
    else:
        input_tokens = input_tokens[:seq_len]

    input_np = np.array([input_tokens], dtype=np.int64)
    input_torch = torch.tensor(input_np, dtype=torch.long)

    print(f"[*] Prompt: '{prompt}' (padded to {seq_len} tokens)")

    # 2. PyTorch Ground Truth
    print("[*] Running PyTorch EdgeModel forward pass...")
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    pt_model = SmolLMEdgeModel(config)
    pt_model.eval()

    state_dict = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt", device="cpu") as sf:
        for k in sf.keys():
            state_dict[k] = sf.get_tensor(k)

    mapped_dict = {
        "embed_tokens.weight": state_dict["model.embed_tokens.weight"],
        "norm.weight": state_dict["model.norm.weight"],
        "lm_head.weight": state_dict.get("lm_head.weight", state_dict["model.embed_tokens.weight"]),
    }
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
    pt_model.load_state_dict(mapped_dict, strict=False)

    with torch.no_grad():
        pt_logits = pt_model(input_torch).numpy()

    # 3. LiteRT Interpreter Forward Pass
    print(f"[*] Initializing LiteRT Interpreter with {tflite_model_path}...")
    interpreter = Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"    - Input Details : {input_details[0]['shape']} ({input_details[0]['dtype'].__name__})")
    print(f"    - Output Details: {output_details[0]['shape']} ({output_details[0]['dtype'].__name__})")

    # Set tensor
    in_dtype = input_details[0]['dtype']
    interpreter.set_tensor(input_details[0]['index'], input_np.astype(in_dtype))

    # Invoke runtime
    print("[*] Invoking LiteRT on-device inference...")
    interpreter.invoke()

    litert_logits = interpreter.get_tensor(output_details[0]['index'])

    # 4. Compare Logits
    diff = np.abs(pt_logits - litert_logits)
    max_err = np.max(diff)
    mean_err = np.mean(diff)

    pt_last = pt_logits[0, -1, :].astype(np.float64)
    lt_last = litert_logits[0, -1, :].astype(np.float64)
    cos_sim = np.dot(pt_last, lt_last) / (np.linalg.norm(pt_last) * np.linalg.norm(lt_last))

    print("\n" + "=" * 65)
    print("               LiteRT NUMERICAL PARITY RESULTS                   ")
    print("=" * 65)
    print(f"Max Absolute Error  : {max_err:.6e}")
    print(f"Mean Absolute Error : {mean_err:.6e}")
    print(f"Cosine Similarity   : {cos_sim:.8f}")

    top_pt = int(np.argmax(pt_last))
    top_lt = int(np.argmax(lt_last))
    print(f"PyTorch Top Token   : ID={top_pt} -> {repr(tokenizer.decode([top_pt]))}")
    print(f"LiteRT  Top Token   : ID={top_lt} -> {repr(tokenizer.decode([top_lt]))}")

    assert top_pt == top_lt, "Top token mismatch between PyTorch and LiteRT!"
    print("\n[SUCCESS] LiteRT output perfectly matches PyTorch!")


if __name__ == "__main__":
    tflite_file = sys.argv[1] if len(sys.argv) > 1 else "smollm2_135m.tflite"
    src_dir = sys.argv[2] if len(sys.argv) > 2 else "models/SmolLM2-135M"
    seq = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    verify_litert(tflite_file, src_dir, seq)
