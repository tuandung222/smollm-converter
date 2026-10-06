"""
GGUF v3 Binary Writer implemented completely from scratch.
Zero dependency on llama.cpp internal classes.
Only relies on standard library (struct) and numpy.
"""
from __future__ import annotations
import enum
import os
import struct
from typing import Any, List, Tuple
import numpy as np


class GGUFValueType(enum.IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(enum.IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q8_0 = 8


def quantize_q8_0(tensor: np.ndarray) -> bytes:
    """
    Quantize a float32/float16 numpy array to GGML Q8_0 format from scratch.
    GGML Q8_0 structure per block of 32 values:
      - d: float16 (2 bytes) = max(abs(x)) / 127.0
      - qs: int8[32] (32 bytes) = round(x / d)
    Total: 34 bytes for 32 elements.
    """
    x = tensor.astype(np.float32).flatten()
    n_elements = x.size
    assert n_elements % 32 == 0, f"Tensor size {n_elements} must be divisible by 32 for Q8_0"

    blocks = x.reshape(-1, 32)
    # Calculate scale factor d
    max_val = np.max(np.abs(blocks), axis=-1)
    d = (max_val / 127.0).astype(np.float16)

    # Invert d safely for fast multiplication
    inv_d = np.where(d != 0, 1.0 / d.astype(np.float32), 0.0)
    qs = np.clip(np.round(blocks * inv_d[:, None]), -128, 127).astype(np.int8)

    # Interleave d (2 bytes) and qs (32 bytes)
    # d.tobytes() has len = n_blocks * 2
    # qs.tobytes() has len = n_blocks * 32
    # We pack them row by row efficiently using structured dtype
    dt = np.dtype([('d', np.float16), ('qs', np.int8, (32,))])
    packed = np.empty(blocks.shape[0], dtype=dt)
    packed['d'] = d
    packed['qs'] = qs
    return packed.tobytes()


class GGUFWriter:
    """
    Direct low-level GGUF v3 Binary File Writer.
    """
    GGUF_MAGIC = b"GGUF"
    GGUF_VERSION = 3

    def __init__(self, output_path: str, alignment: int = 32):
        self.output_path = output_path
        self.alignment = alignment
        self.kv_data: List[Tuple[str, GGUFValueType, bytes]] = []
        self.tensors: List[dict] = []

    def _pack_str(self, s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _pack_value(self, val_type: GGUFValueType, value: Any) -> bytes:
        if val_type == GGUFValueType.UINT8:
            return struct.pack("<B", value)
        elif val_type == GGUFValueType.INT8:
            return struct.pack("<b", value)
        elif val_type == GGUFValueType.UINT16:
            return struct.pack("<H", value)
        elif val_type == GGUFValueType.INT16:
            return struct.pack("<h", value)
        elif val_type == GGUFValueType.UINT32:
            return struct.pack("<I", value)
        elif val_type == GGUFValueType.INT32:
            return struct.pack("<i", value)
        elif val_type == GGUFValueType.FLOAT32:
            return struct.pack("<f", float(value))
        elif val_type == GGUFValueType.BOOL:
            return struct.pack("<B", 1 if value else 0)
        elif val_type == GGUFValueType.STRING:
            return self._pack_str(str(value))
        elif val_type == GGUFValueType.UINT64:
            return struct.pack("<Q", value)
        elif val_type == GGUFValueType.INT64:
            return struct.pack("<q", value)
        elif val_type == GGUFValueType.FLOAT64:
            return struct.pack("<d", float(value))
        else:
            raise ValueError(f"Unsupported direct value type: {val_type}")

    def add_key_value(self, key: str, val_type: GGUFValueType, value: Any):
        val_bytes = self._pack_value(val_type, value)
        self.kv_data.append((key, val_type, val_bytes))

    def add_string(self, key: str, value: str):
        self.add_key_value(key, GGUFValueType.STRING, value)

    def add_uint32(self, key: str, value: int):
        self.add_key_value(key, GGUFValueType.UINT32, int(value))

    def add_int32(self, key: str, value: int):
        self.add_key_value(key, GGUFValueType.INT32, int(value))

    def add_uint64(self, key: str, value: int):
        self.add_key_value(key, GGUFValueType.UINT64, int(value))

    def add_float32(self, key: str, value: float):
        self.add_key_value(key, GGUFValueType.FLOAT32, float(value))

    def add_bool(self, key: str, value: bool):
        self.add_key_value(key, GGUFValueType.BOOL, bool(value))

    def add_string_array(self, key: str, values: List[str]):
        elem_type = GGUFValueType.STRING
        payload = [struct.pack("<I", elem_type), struct.pack("<Q", len(values))]
        for v in values:
            payload.append(self._pack_str(v))
        self.kv_data.append((key, GGUFValueType.ARRAY, b"".join(payload)))

    def add_float32_array(self, key: str, values: List[float]):
        elem_type = GGUFValueType.FLOAT32
        payload = [struct.pack("<I", elem_type), struct.pack("<Q", len(values))]
        for v in values:
            payload.append(struct.pack("<f", float(v)))
        self.kv_data.append((key, GGUFValueType.ARRAY, b"".join(payload)))

    def add_int32_array(self, key: str, values: List[int]):
        elem_type = GGUFValueType.INT32
        payload = [struct.pack("<I", elem_type), struct.pack("<Q", len(values))]
        for v in values:
            payload.append(struct.pack("<i", int(v)))
        self.kv_data.append((key, GGUFValueType.ARRAY, b"".join(payload)))

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: GGMLType = GGMLType.F16):
        """
        Register a tensor for binary writing.
        data: numpy ndarray (float32, float16, etc.)
        """
        shape = list(data.shape)
        # Prepare binary bytes according to ggml_type
        if ggml_type == GGMLType.F32:
            raw_bytes = data.astype(np.float32).tobytes()
        elif ggml_type == GGMLType.F16:
            raw_bytes = data.astype(np.float16).tobytes()
        elif ggml_type == GGMLType.Q8_0:
            raw_bytes = quantize_q8_0(data)
        else:
            raise NotImplementedError(f"GGMLType {ggml_type} not implemented yet")

        self.tensors.append({
            "name": name,
            "shape": shape,
            "type": ggml_type,
            "raw_bytes": raw_bytes,
        })

    def write(self):
        """
        Write complete GGUF v3 file to disk.
        """
        print(f"[*] Serializing GGUF to {self.output_path}...")
        print(f"    - Metadata entries: {len(self.kv_data)}")
        print(f"    - Tensors: {len(self.tensors)}")

        with open(self.output_path, "wb") as f:
            # 1. Header
            f.write(self.GGUF_MAGIC)
            f.write(struct.pack("<I", self.GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.kv_data)))

            # 2. Metadata Key-Values
            for key, val_type, val_bytes in self.kv_data:
                f.write(self._pack_str(key))
                f.write(struct.pack("<I", int(val_type)))
                f.write(val_bytes)

            # 3. Calculate tensor offsets
            # In GGUF, tensor info contains the offset relative to the start of the tensor data section.
            current_tensor_offset = 0
            for t in self.tensors:
                # Align tensor offset to self.alignment
                pad = (self.alignment - (current_tensor_offset % self.alignment)) % self.alignment
                current_tensor_offset += pad
                t["offset"] = current_tensor_offset
                current_tensor_offset += len(t["raw_bytes"])

            # 4. Tensor Infos
            for t in self.tensors:
                f.write(self._pack_str(t["name"]))
                n_dims = len(t["shape"])
                f.write(struct.pack("<I", n_dims))
                # GGML convention: innermost dimension first (reverse of numpy)
                for d in reversed(t["shape"]):
                    f.write(struct.pack("<Q", int(d)))
                f.write(struct.pack("<I", int(t["type"])))
                f.write(struct.pack("<Q", int(t["offset"])))

            # 5. Alignment padding before tensor data body
            cur_pos = f.tell()
            pad_before_data = (self.alignment - (cur_pos % self.alignment)) % self.alignment
            if pad_before_data > 0:
                f.write(b"\x00" * pad_before_data)

            # 6. Tensor Data Body
            base_offset = f.tell()
            for t in self.tensors:
                target_pos = base_offset + t["offset"]
                gap = target_pos - f.tell()
                if gap > 0:
                    f.write(b"\x00" * gap)
                f.write(t["raw_bytes"])

        total_mb = os.path.getsize(self.output_path) / (1024 * 1024)
        print(f"[+] Successfully wrote GGUF file: {self.output_path} ({total_mb:.2f} MB)")
