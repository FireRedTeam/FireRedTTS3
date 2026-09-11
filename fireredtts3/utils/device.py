"""Device / dtype abstraction so the pipeline runs on CUDA, MPS or CPU.

The upstream code hard-codes CUDA (``torch.device('cuda')``, ``autocast('cuda')``,
``flash_attention_2``). This module resolves everything once at import time so the
model code stays free of ``if cuda ... else ...`` branches.

Environment overrides:

- ``FIRERED_DEVICE``  — force a device, e.g. ``cuda`` / ``mps`` / ``cpu``.
- ``FIRERED_DTYPE``   — autocast dtype: ``bfloat16`` / ``float16`` / ``float32``
  (``float32`` disables autocast). Default: bfloat16 on CUDA, float32 elsewhere.
- ``FIRERED_WEIGHT_DTYPE`` — dtype the checkpoints are *loaded* in. Autocast only
  casts activations, so this is the knob that actually halves resident memory.
  Unset means "whatever the checkpoint stores", which is what transformers does
  on its own — fp32 for the official weights.
- ``FIRERED_QUANT`` — ``int8`` applies weight-only quantization to the transformer
  backbones (requires ``optimum-quanto``). Default: ``none``.
"""

import os
import torch


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "none": torch.float32,
}


def _resolve_device() -> torch.device:
    override = os.environ.get("FIRERED_DEVICE", "").strip()
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_autocast_dtype(device_type: str) -> torch.dtype:
    override = os.environ.get("FIRERED_DTYPE", "").strip().lower()
    if override:
        if override not in _DTYPES:
            raise ValueError(f"invalid FIRERED_DTYPE={override!r}, expected one of {sorted(_DTYPES)}")
        return _DTYPES[override]
    # MPS/CPU: bf16 autocast is either unsupported or emulated (slow, and the
    # flow-matching head is numerically touchy) -> stay in fp32 by default.
    return torch.bfloat16 if device_type == "cuda" else torch.float32


def _resolve_weight_dtype():
    """Returns None when unset, so from_pretrained keeps the checkpoint's dtype."""
    override = os.environ.get("FIRERED_WEIGHT_DTYPE", "").strip().lower()
    if not override:
        return None
    if override not in _DTYPES:
        raise ValueError(f"invalid FIRERED_WEIGHT_DTYPE={override!r}, expected one of {sorted(_DTYPES)}")
    return _DTYPES[override]


def _resolve_quant() -> str:
    quant = os.environ.get("FIRERED_QUANT", "").strip().lower() or "none"
    if quant in ("int4", "qint4"):
        raise ValueError(
            "FIRERED_QUANT=int4 is not supported: measured 3x slower than bf16 on MPS "
            "(RTF 3.39 vs 1.47) with a clear speaker-similarity drop (0.828 vs 0.913), and "
            "quanto's int4 path conflicts with the @torch.inference_mode() decorators. "
            "Use FIRERED_QUANT=int8."
        )
    if quant not in ("none", "int8"):
        raise ValueError(f"invalid FIRERED_QUANT={quant!r}, expected 'none' or 'int8'")
    return quant


def _probe_fft(device_type: str) -> bool:
    """Whether complex tensors + inverse FFT work on this device (MPS: no)."""
    if device_type == "cpu":
        return True
    try:
        spec = torch.zeros(1, 3, 2, dtype=torch.complex64, device=device_type)
        torch.fft.irfft(spec, 4, dim=1)
        return True
    except Exception:
        return False


DEVICE: torch.device = _resolve_device()
DEVICE_TYPE: str = DEVICE.type
AUTOCAST_DTYPE: torch.dtype = _resolve_autocast_dtype(DEVICE_TYPE)
AUTOCAST_ENABLED: bool = AUTOCAST_DTYPE != torch.float32
WEIGHT_DTYPE = _resolve_weight_dtype()   # None => as stored in the checkpoint
QUANT: str = _resolve_quant()
# MPS has no FFT / complex kernels -> ISTFT and kaldi fbank must run on CPU.
FFT_ON_DEVICE: bool = _probe_fft(DEVICE_TYPE)


def get_device() -> torch.device:
    return DEVICE


def get_weight_dtype():
    """dtype to load checkpoints in; None means keep whatever they store."""
    return WEIGHT_DTYPE


def get_quant() -> str:
    return QUANT


def quantize_weights(*modules) -> None:
    """Apply int8 weight-only quantization in place; no-op unless FIRERED_QUANT=int8.

    Pass only the transformer backbones. The DiT flow head is deliberately left
    alone: it runs once per flow timestep with a CFG-doubled batch, so per-call
    dequantization dominates (measured RTF 2.41 vs 1.16 when it is included).
    """
    if QUANT == "none":
        return
    try:
        from optimum.quanto import quantize, freeze, qint8
    except ImportError as exc:  # optional dependency
        raise ImportError(
            "FIRERED_QUANT=int8 requires optimum-quanto: pip install optimum-quanto"
        ) from exc
    for module in modules:
        quantize(module, weights=qint8)
        freeze(module)


def get_attn_implementation() -> str:
    """flash-attn wheels are CUDA-only; SDPA covers MPS/CPU."""
    return "flash_attention_2" if DEVICE_TYPE == "cuda" else "sdpa"


def autocast(func):
    """Decorator replacing ``@torch.autocast(device_type='cuda', dtype=bfloat16)``."""
    if not AUTOCAST_ENABLED:
        return func
    return torch.autocast(device_type=DEVICE_TYPE, dtype=AUTOCAST_DTYPE)(func)


def disable_autocast(func):
    """Decorator replacing ``@autocast('cuda', enabled=False)``."""
    if not AUTOCAST_ENABLED:
        return func
    return torch.autocast(device_type=DEVICE_TYPE, enabled=False)(func)


def fft_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor to a device that can actually run FFT / complex math."""
    if FFT_ON_DEVICE:
        return tensor
    return tensor.cpu()


def describe() -> str:
    return (
        f"device={DEVICE}, weights={WEIGHT_DTYPE or 'as-stored'}, quant={QUANT}, "
        f"autocast={'off' if not AUTOCAST_ENABLED else AUTOCAST_DTYPE}, "
        f"attn={get_attn_implementation()}, fft_on_device={FFT_ON_DEVICE}"
    )
