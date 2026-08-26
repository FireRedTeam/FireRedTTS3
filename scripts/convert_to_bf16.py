"""Convert the FireRedTTS3 checkpoints in place from fp32 to bfloat16.

Halves them on disk (19 GB -> 9.7 GB) and speeds up loading, since the loader
then picks up bfloat16 from each config.json on its own. Measured speaker
similarity is unchanged; see the MPS section of the README.

Every tensor of the new file is checked against the fp32 original for exact
bfloat16 rounding BEFORE the original is replaced. The cast is lossy and
one-way: to get fp32 back, re-download the checkpoints.

Usage:
    python scripts/convert_to_bf16.py                      # all components
    python scripts/convert_to_bf16.py redae                # just one
    python scripts/convert_to_bf16.py --keep-fp32          # write <dir>.bf16, keep originals
"""

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open

from fireredtts3.llm.fireredtts3_base import FireRedTTS3BaseCore
from fireredtts3.llm.fireredtts3_instruct import FireRedTTS3InstructCore
from fireredtts3.redae.redae import RedAE

# smallest first, so disk is freed progressively on a nearly-full volume
COMPONENTS = {
    "redae": RedAE,
    "fireredtts3_base": FireRedTTS3BaseCore,
    "fireredtts3_instruct": FireRedTTS3InstructCore,
}


def dir_size(path: str) -> int:
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(path)
        for f in fs
    )


def verify(old_dir: str, new_dir: str) -> int:
    """Assert every tensor equals the fp32 original rounded to bf16."""
    old_f = os.path.join(old_dir, "model.safetensors")
    new_f = os.path.join(new_dir, "model.safetensors")
    with safe_open(old_f, framework="pt") as a, safe_open(new_f, framework="pt") as b:
        ka, kb = set(a.keys()), set(b.keys())
        if ka != kb:
            raise AssertionError(
                f"key mismatch: missing {sorted(ka - kb)[:5]}, extra {sorted(kb - ka)[:5]}"
            )
        for k in ka:
            t_old, t_new = a.get_tensor(k), b.get_tensor(k)
            if t_old.shape != t_new.shape:
                raise AssertionError(f"{k}: shape {t_old.shape} != {t_new.shape}")
            if t_new.dtype != torch.bfloat16:
                raise AssertionError(f"{k}: dtype is {t_new.dtype}, expected bfloat16")
            if not torch.equal(t_old.to(torch.bfloat16), t_new):
                raise AssertionError(f"{k}: does not match exact bf16 rounding")
        return len(ka)


def convert(root: str, comp: str, keep_fp32: bool) -> None:
    src = os.path.join(root, comp)
    if not os.path.isdir(src):
        print(f"[{comp}] SKIP: {src} not found")
        return

    cfg_path = os.path.join(src, "config.json")
    if json.load(open(cfg_path)).get("dtype") == "bfloat16":
        print(f"[{comp}] SKIP: already bfloat16")
        return

    tmp = src + ".bf16"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)

    before = dir_size(src)
    print(f"[{comp}] fp32 on disk: {before / 1024**3:.2f} GiB", flush=True)

    model = COMPONENTS[comp].from_pretrained(src, dtype=torch.bfloat16)
    model.save_pretrained(tmp)
    del model
    after = dir_size(tmp)
    print(f"[{comp}] bf16 written: {after / 1024**3:.2f} GiB "
          f"({before / after:.2f}x smaller)", flush=True)

    n = verify(src, tmp)
    print(f"[{comp}] verified {n} tensors: exact bf16 rounding")

    if keep_fp32:
        print(f"[{comp}] kept fp32; bf16 left in {tmp}")
        return
    shutil.rmtree(src)
    os.rename(tmp, src)
    print(f"[{comp}] replaced in place")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("components", nargs="*", choices=list(COMPONENTS), default=None,
                    help="components to convert (default: all)")
    ap.add_argument("--models", default="pretrained_models")
    ap.add_argument("--keep-fp32", action="store_true",
                    help="write <dir>.bf16 and leave the originals alone "
                         "(needs room for both)")
    args = ap.parse_args()

    for comp in (args.components or list(COMPONENTS)):
        convert(args.models, comp, args.keep_fp32)
    print("[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
