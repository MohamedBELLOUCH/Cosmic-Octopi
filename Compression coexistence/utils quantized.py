"""Int8 uplink transport for the project's float32 policy/value Networks.

The bytes object is the complete model payload: header, scales, shapes, names,
architecture and int8 tensor values. Scalar fitness/episode messages and network
protocol framing are excluded, as in the original model-only overhead metric.
"""

import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

_CORE = Path(__file__).resolve().parents[1] / "Cosmic Octopi"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
from utils import Network

_MAGIC = b"COQ1"


def quantize_model(model):
    """Snapshot a float32 Network into per-tensor symmetric int8 wire bytes.

    Each tensor uses scale=max(abs(tensor))/127, or 1 for an all-zero tensor.
    No floating model or template is included in the upload. Reject payloads
    that would not actually save bytes rather than undercounting their cost.
    """
    architecture = [model.policy.dense1.in_features,
                    model.policy.mu_head.out_features,
                    model.policy.dense1.out_features]
    entries, chunks = [], []
    original_bytes = 0
    for name, tensor in model.state_dict().items():
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name}: the upload codec requires float32 tensors")
        values = tensor.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: cannot quantize non-finite weights")
        maximum = float(np.max(np.abs(values))) if values.size else 0.0
        scale = maximum / 127.0 if maximum else 1.0
        # Float64 arithmetic also handles very small float32 weights safely.
        quantized = np.clip(np.rint(values.astype(np.float64) / scale),
                            -127, 127).astype(np.int8)
        entries.append([name, list(values.shape), scale])
        chunks.append(quantized.tobytes(order="C"))
        original_bytes += values.nbytes
    header = json.dumps({"architecture": architecture, "tensors": entries},
                        separators=(",", ":"), allow_nan=False).encode("utf-8")
    payload = _MAGIC + struct.pack("<I", len(header)) + header + b"".join(chunks)
    if len(payload) >= original_bytes:
        raise ValueError("Quantized payload, including metadata, is not smaller "
                         "than the original float32 model")
    return payload


def dequantize_model(payload):
    """Rebuild a float32 CPU Network from transmitted bytes, before FedAvg.

    Meta-device construction avoids initializing throwaway random weights, so
    receiving a model does not advance the simulation's random-number state.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Expected a quantized bytes upload, not a floating model")
    if len(payload) < 8 or payload[:4] != _MAGIC:
        raise ValueError("Invalid quantized model header")
    header_size = struct.unpack("<I", payload[4:8])[0]
    body_start = 8 + header_size
    if body_start > len(payload):
        raise ValueError("Truncated quantized model header")
    header = json.loads(payload[8:body_start])
    with torch.device("meta"):
        model = Network(*header["architecture"])
    expected = model.state_dict()
    state = {}
    cursor = body_start
    for name, shape, scale in header["tensors"]:
        if name in state or name not in expected or list(expected[name].shape) != shape:
            raise ValueError(f"Unexpected or duplicate tensor: {name}")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid quantization scale for {name}")
        count = expected[name].numel()
        if cursor + count > len(payload):
            raise ValueError("Truncated quantized model weights")
        values = np.frombuffer(payload, dtype=np.int8, count=count, offset=cursor)
        restored = (values.astype(np.float64) * scale).astype(np.float32).reshape(shape)
        state[name] = torch.from_numpy(restored)
        cursor += count
    if cursor != len(payload) or state.keys() != expected.keys():
        raise ValueError("Quantized payload has extra bytes or missing tensors")
    model.load_state_dict(state, strict=True, assign=True)
    return model


def get_payload_size_in_kb(payload):
    """Full transmitted model payload size in KiB (matching original units)."""
    if not isinstance(payload, bytes):
        raise TypeError("Overhead must be measured on quantized upload bytes")
    return len(payload) / 1024.0
