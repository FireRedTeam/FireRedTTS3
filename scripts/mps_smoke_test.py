"""Smoke test for FireRedTTS3 on Apple Silicon (MPS).

Runs the Base zero-shot cloning path and, optionally, the Instruct paths
(cloning / voice design / semantic edit / acoustic edit), timing each call and
writing wavs to an output directory.

Usage:
    python scripts/mps_smoke_test.py \
        --refs /path/to/persons_refs \
        --out outputs/mps_test \
        --language Russian \
        --tasks base,instruct_tts,voice_design,semantic_edit,acoustic_edit
"""

import argparse
import os
import time

import torch
import torchaudio

from fireredtts3.utils.device import describe, get_device


TEXTS = {
    "Russian": "Он поднял голову и посмотрел на далёкие пики, скрытые в облаках. "
               "До входа в долину оставалось три дня пути, и никто из них не знал, "
               "чем закончится это путешествие.",
    "English": "He raised his head and looked at the distant peaks hidden in the clouds. "
               "Three days of travel remained before the valley.",
    "Chinese": "他抬起头，望着远处隐没在云雾中的山峰，谁也不知道这次旅程会如何结束。",
}


def load_refs(refs_dir: str, max_seconds: float):
    """Load (name, wav, sr, transcript) for every audio file with a .txt sibling."""
    items = []
    for fn in sorted(os.listdir(refs_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
            continue
        txt_path = os.path.join(refs_dir, stem + ".txt")
        if not os.path.exists(txt_path):
            print(f"[SKIP] {fn}: no transcript")
            continue
        with open(txt_path, encoding="utf-8") as f:
            transcript = f.read().strip()
        audio, sr = torchaudio.load(os.path.join(refs_dir, fn))
        audio = audio[:1]  # mono
        if max_seconds > 0:
            audio = audio[:, : int(max_seconds * sr)]
        items.append((stem, audio, sr, transcript))
    return items


def timed(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    return out, dt, label


def report(label, dt, audio, sr):
    dur = audio.shape[-1] / sr
    rtf = dt / dur if dur > 0 else float("nan")
    print(f"[OK] {label}: {dur:.2f}s audio in {dt:.1f}s wall (RTF {rtf:.2f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="directory with <name>.mp3 + <name>.txt")
    ap.add_argument("--out", default="outputs/mps_test")
    ap.add_argument("--models", default="pretrained_models")
    ap.add_argument("--language", default="Russian")
    ap.add_argument("--text", default=None)
    ap.add_argument("--max-ref-seconds", type=float, default=10.0,
                    help="trim reference audio (0 = keep full length)")
    ap.add_argument("--max-voices", type=int, default=0, help="0 = all voices")
    ap.add_argument("--acoustic-instruction", default="adjust the speed to 0.8x",
                    help="templates: 'adjust the speed to X' / 'shift the pitch by N steps' / 'adjust the volume to X'")
    ap.add_argument("--semantic-instruction", default="Replace 'Возможно' with 'Вероятно'.")
    ap.add_argument("--tasks", default="base",
                    help="comma list: base,instruct_tts,voice_design,semantic_edit,acoustic_edit")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    os.makedirs(args.out, exist_ok=True)
    print(f"[INFO] {describe()}")
    print(f"[INFO] torch {torch.__version__}, tasks={tasks}")

    refs = load_refs(args.refs, args.max_ref_seconds)
    if args.max_voices:
        refs = refs[: args.max_voices]
    print(f"[INFO] {len(refs)} reference voice(s): {[r[0] for r in refs]}")
    text = args.text or TEXTS.get(args.language, TEXTS["English"])

    # ---------------- Base: zero-shot cloning ----------------
    if "base" in tasks:
        from fireredtts3.core import FireRedTTS3

        (tts, load_dt, _) = timed("load base", lambda: FireRedTTS3(
            args.models, use_fasttext=False, use_wetext=True, use_llm_tn=False))
        print(f"[OK] base model loaded in {load_dt:.1f}s", flush=True)

        for name, audio, sr, transcript in refs:
            (out, dt, _) = timed(name, lambda: tts.generate(
                language=args.language,
                prompt_text=transcript,
                prompt_audio=audio,
                prompt_audio_sr=sr,
                text=text,
            ))
            gen_audio, gen_sr = out
            path = os.path.join(args.out, f"base_{name}.wav")
            torchaudio.save(path, gen_audio.cpu(), gen_sr)
            report(f"base clone {name} -> {path}", dt, gen_audio, gen_sr)

        del tts
        if get_device().type == "mps":
            torch.mps.empty_cache()

    # ---------------- Instruct ----------------
    instruct_tasks = [t for t in tasks if t != "base"]
    if instruct_tasks:
        from fireredtts3.core import FireRedTTS3Instruct

        (ins, load_dt, _) = timed("load instruct", lambda: FireRedTTS3Instruct(
            args.models, use_fasttext=False, use_wetext=True, use_llm_tn=False))
        print(f"[OK] instruct model loaded in {load_dt:.1f}s", flush=True)

        if "instruct_tts" in instruct_tasks:
            name, audio, sr, transcript = refs[0]
            (out, dt, _) = timed("instruct_tts", lambda: ins.generate_tts(
                prompt_text=transcript, prompt_audio=audio, prompt_audio_sr=sr,
                text=text, language=args.language))
            gen_audio, gen_sr = out
            path = os.path.join(args.out, f"instruct_clone_{name}.wav")
            torchaudio.save(path, gen_audio.cpu(), gen_sr)
            report(f"instruct clone {name} -> {path}", dt, gen_audio, gen_sr)

        if "voice_design" in instruct_tasks:
            instruction = ("A calm middle-aged male narrator with a deep, warm voice, "
                           "speaking slowly and clearly, like an audiobook reader.")
            (out, dt, _) = timed("voice_design", lambda: ins.generate_voice_design(
                instruction=instruction, text=text, language=args.language))
            gen_audio, gen_sr, gen_text = out
            path = os.path.join(args.out, "instruct_voice_design.wav")
            torchaudio.save(path, gen_audio.cpu(), gen_sr)
            report(f"voice design -> {path}", dt, gen_audio, gen_sr)
            print(f"       voice plan: {gen_text}")

        if "semantic_edit" in instruct_tasks or "acoustic_edit" in instruct_tasks:
            # edits re-render the whole utterance -> use the shortest reference
            name, audio, sr, transcript = min(refs, key=lambda r: r[1].shape[-1])
            print(f"[INFO] editing {name} ({audio.shape[-1] / sr:.1f}s): {transcript[:60]}...")

        if "acoustic_edit" in instruct_tasks:
            (out, dt, _) = timed("acoustic_edit", lambda: ins.generate_acoustic_edit(
                instruction=args.acoustic_instruction, audio_in=audio, audio_in_sr=sr))
            gen_audio, gen_sr = out[0], out[1]
            path = os.path.join(args.out, f"acoustic_edit_{name}.wav")
            torchaudio.save(path, gen_audio.cpu(), gen_sr)
            report(f"acoustic edit -> {path}", dt, gen_audio, gen_sr)

        if "semantic_edit" in instruct_tasks:
            (out, dt, _) = timed("semantic_edit", lambda: ins.generate_semantic_edit(
                instruction=args.semantic_instruction, audio_in=audio, audio_in_sr=sr))
            gen_audio, gen_sr, gen_text = out
            path = os.path.join(args.out, f"semantic_edit_{name}.wav")
            torchaudio.save(path, gen_audio.cpu(), gen_sr)
            report(f"semantic edit -> {path}", dt, gen_audio, gen_sr)
            print(f"       edited text: {gen_text}")

    print("[DONE]")


if __name__ == "__main__":
    main()
