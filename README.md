<div align="center">
    <h1>
    FireRedTTS3
    </h1>
    <p>
    Official PyTorch code for <br>
    <b><em>FireRedTTS3: Unified Speech Generation and Editing with Semantically Enriched Speech Representations</em></b>
    </p>
    <p>
    <img src="assets/firered_logo.png" alt="FireRedTTS Logo" width="248" height="68">
    </p>
    <p>
    </p>
    <a href="https://arxiv.org/abs/2608.17492"><img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="technical report"></a>
    <a href="https://fireredteam.github.io/demos/firered_tts_3/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
    <a href="https://huggingface.co/FireRedTeam/FireRedTTS3"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Page-yellow" alt="HF-model"></a>
    <a href="https://www.modelscope.cn/models/FireRedTeam/FireRedTTS3"><img src="https://img.shields.io/badge/ModelScope-Model%20Page-624AFF?logo=modelscope&logoColor=white" alt="ModelScope-model"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0"></a>
</div>


## Overview

<!-- 
<div align="center">
  <img src="assets/fireredtts3_arch.png" alt="FireRedTTS3 Architecture" width="95%">
  <br>
</div> 
-->

<div align="center">

<https://github.com/user-attachments/assets/1ff882ce-828d-48d5-bd2a-4ad12ee912ac>

</div>

**FireRedTTS3** is a unified speech generation and editing system built on **semantically enriched continuous speech representations**. It comes in two variants:

- **FireRedTTS3-Base** — zero-shot voice cloning across **24 languages** and **21 Chinese dialects**
- **FireRedTTS3-Instruct** — natural-language **voice design** and **speech editing** (semantic + acoustic) in one unified model


## Highlights ✨ 

* 🌍 **Multilingual — 24 Languages** — Best average WER/CER (avg 3.754%) and best average speaker similarity on MiniMax-MLS-Test (avg 84.8%), plus best-in-class cloning WER/CER (avg 3.04%) and similarity on Seed-TTS-eval (avg 78.8%). Supported languages:
 `Arabic` · `Cantonese` · `Chinese` · `Czech` · `Dutch` · `English` · `Finnish` · `French` · `German` · `Greek` · `Hindi` · `Indonesian` · `Italian` · `Japanese` · `Korean` · `Polish` · `Portuguese` · `Romanian` · `Russian` · `Spanish` · `Thai` · `Turkish` · `Ukrainian` · `Vietnamese`
* 🗣️ **Multi-Dialect — 21 Chinese Dialects** — Zero-shot voice cloning across major Chinese dialect groups. Supported dialects: 
`Anhui` · `Fujian` · `Gansu` · `Guizhou` · `Hebei` · `Henan` · `Hubei` · `Hunan` · `Jiangxi` · `Liaoning` · `Minnan` · `Ningxia` · `Shaanxi` · `Shandong` · `Shanghai` · `Shanxi` · `Sichuan` · `Tianjin` · `Wenzhou` · `Wu` · `Yunnan`
* 🎨 **Instruction-Controlled Voice Design** — Generate a brand-new voice from a natural-language description (gender, age, timbre, emotion, pace, accent…) with no reference audio, guided by an explicit textual plainning step before synthesis.
* ✂️ **Free-Form Speech Editing** — Semantic editing (insertion / deletion / substitution) and acoustic editing (speed / pitch / volume) driven by free-form instructions.


## News
- [2026.08.13] We release the **FireRedTTS3**


## Roadmap

- [x] Release the FireRedTTS3-Base model
- [x] Release the FireRedTTS3-Instruct model
- [x] Release the technical report


## Contents

- [Quick Start](#-quick-start)
- [Model](#-model)
- [Performance](#-performance)
- [Usage Disclaimer](#-usage-disclaimer-)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

## Quick Start 🚀

### Installation with pip

```sh
pip install -r requirements.txt
```

### Model Download

Download the pretrained model from Hugging Face with the `hf` CLI:

```sh
pip install "huggingface_hub[cli]"
hf download FireRedTeam/FireRedTTS3 --local-dir pretrained_models/
```

Alternatively, you can download model from ModelScope
```sh
pip install modelscope
modelscope download --model FireRedTeam/FireRedTTS3 --local_dir pretrained_models/
```

### Configure Text Frontend

#### Language Recognition (Optional)
FireRedTTS3-Base relies on explicit language tags for best performance. However, if you don't know the exact language of the text, you can download Meta's `FastText` language-id model and let it detect the language automatically.

```sh
# Download FastText language-id model (lid.176) with:
curl -L -o fireredtts3/utils/llm_tn/models/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

#### Text Normalization (TN)
TN converts written numbers, dates, units, currencies, acronyms, etc. into their spoken form (e.g. 19:30 → nineteen thirty). By default, FireRedTTS3 uses the `wetext` TN tool, which supports Chinese and English, other languages (e.g. Japanese, Russian) undergo only basic cleaning. For full language TN support, enable the LLM-based TN by passing `use_llm_tn=True` when initializing FireRedTTS3. It reads its config from a .env file:

```sh
cp .env.example .env

# Then fill in your values
LLM_TN_API_URL=https://api.deepseek.com/chat/completions   # any OpenAI-compatible endpoint
LLM_TN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_TN_MODEL=deepseek-v4-flash                             # or any model >= 30B
```


### Python API

For the best voice cloning performance, use a prompt in the desired language or dialect, since the output inherits the speaking style of the reference. For example, provide a Japanese prompt when synthesizing Japanese and a Sichuanese prompt when synthesizing Sichuanese.

```python
import torch
import torchaudio
from fireredtts3.core import FireRedTTS3

# Init model: choose the text-normalization frontend here.
#   use_wetext=True  -> local weText TN (zh/en only)
#   use_llm_tn=True  -> LLM-based TN (all languages, needs .env / API creds)
#   both False       -> no TN frontend built
tts = FireRedTTS3(
    "pretrained_models",
    use_wetext=True,
    use_llm_tn=False,
)

language = None     # Automatic detection if pass None
prompt_text = "<prompt audio text>"
prompt_audio, prompt_audio_sr = torchaudio.load('prompt.wav')
text = "今天天气很好，我们一起去公园散步吧。"

gen_audio, gen_audio_sr = tts.generate(
    language=language,
    prompt_text=prompt_text,
    prompt_audio=prompt_audio,
    prompt_audio_sr=prompt_audio_sr,
    text=text,
    do_tn=True,       # whether to run the frontend TN on this call
)
torchaudio.save("gen.wav", gen_audio.cpu(), gen_audio_sr)

# Supported languages and dialects

# Multilingual languages:
# Arabic, Cantonese, Chinese, Czech, Dutch, English, Finnish,
# French, German, Greek, Hindi, Indonesian, Italian, Japanese,
# Korean, Polish, Portuguese, Romanian, Russian, Spanish, Thai,
# Turkish, Ukrainian, Vietnamese

# Multi-dialect:
# ZH_Anhui, ZH_Fujian, ZH_Gansu, ZH_Guizhou, ZH_Hebei, ZH_Henan,
# ZH_Hubei, ZH_Hunan, ZH_Jiangxi, ZH_Liaoning, ZH_Minnan, ZH_Ningxia,
# ZH_Shaanxi, ZH_Shandong, ZH_Shanghai, ZH_Shanxi, ZH_Sichuan,
# ZH_Tianjin, ZH_Wenzhou, ZH_Wu, ZH_Yunnan
```

### Instruct API — Voice Design & Speech Editing

**FireRedTTS3-Instruct** is a unified instruction-driven model. On top of
zero-shot voice cloning, it also supports **Voice Design**, **Semantic Edit**
and **Acoustic Edit** through a single entry point:
`fireredtts3.core.FireRedTTS3Instruct`.

```python
import torch
import torchaudio
from fireredtts3.core import FireRedTTS3Instruct

# Init the Instruct model (same text-frontend options as FireRedTTS3)
instruct = FireRedTTS3Instruct(
    "pretrained_models",
    use_wetext=True,
    use_llm_tn=False,   # set True to enable LLM-based TN (all languages)
)

# ---- 1) Voice Design Inference ---------------
# Generate a brand-new voice from a natural-language description only;
# no reference audio is needed. The model first writes a voice-attribute
# plan (returned as gen_text), then renders the audio.
instruction = "一个年轻女性的温柔嗓音，语速稍慢，带一点俏皮。"
text = "今天天气很好，我们一起去公园散步吧。"
gen_audio, gen_audio_sr, gen_text = instruct.generate_voice_design(
    instruction=instruction,
    text=text,
)
torchaudio.save("design.wav", gen_audio.cpu(), gen_audio_sr)
print("Voice plan:", gen_text)

# ---- 2) Semantic Edit ------------------------
# Content-level editing: insertion / deletion / substitution by instruction.
# Returns the edited audio and the model's rewritten text with edit mask.
audio_in, audio_in_sr = torchaudio.load("input.wav")
gen_audio, gen_audio_sr, gen_text = instruct.generate_semantic_edit(
    instruction="Replace 'cats' with 'dogs'.",
    audio_in=audio_in,
    audio_in_sr=audio_in_sr,
)
torchaudio.save("edit_semantic.wav", gen_audio.cpu(), gen_audio_sr)
print("Edited text:", gen_text)

# ---- 3) Acoustic Edit ------------------------
# Acoustic-attribute editing: speed / pitch / volume. The instruction must
# follow the trained templates below (free-form phrasing is not supported):
#   speed   ->  "adjust the speed to X"       X in [0.5, 2.0], step 0.1
#   pitch   ->  "shift the pitch by N step(s)"  N in {-6,...,-1,1,...,+6}
#   volume  ->  "adjust the volume to X"      X in [0.3, 2.0], step 0.1
gen_audio, gen_audio_sr = instruct.generate_acoustic_edit(
    instruction="adjust the speed to 0.5x",
    audio_in=audio_in,
    audio_in_sr=audio_in_sr,
)
torchaudio.save("edit_acoustic.wav", gen_audio.cpu(), gen_audio_sr)

# ---- 4) ICL zero-shot voice cloning using the Instruct model ----
gen_audio, gen_audio_sr = instruct.generate_tts(
    prompt_text="<prompt audio text>",
    prompt_audio=prompt_audio,
    prompt_audio_sr=prompt_audio_sr,
    text="<text to be synthesized>",
)
torchaudio.save("gen_instruct.wav", gen_audio.cpu(), gen_audio_sr)
```


## Performance

### Zero-Shot Voice Cloning — Seed-TTS-eval

Best in **bold**, second best in <ins>underline</ins>. Evaluation scripts: [Seed-TTS-eval](https://github.com/BytedanceSpeech/seed-tts-eval).

| Model | Test-EN<br>WER/SIM | Test-ZH<br>CER/SIM | Test-Hard<br>CER/SIM | Avg<br>WER/SIM |
| --- | --- | --- | --- | --- |
| CosyVoice3-1.5B | 2.22 / 72.0 | 1.12 / 78.1 | **5.83** / 75.8 | <ins>3.06</ins> / 75.3 |
| DiTAR | 1.69 / 73.5 | 1.02 / 75.3 | – / – | – / – |
| F5-TTS | 2.00 / 67.0 | 1.53 / 76.0 | 8.67 / 71.3 | 4.10 / 71.4 |
| FireRedTTS-2 | 1.95 / 66.5 | 1.14 / 73.6 | 8.98 / 70.3 | 4.02 / 70.1 |
| IndexTTS2 | 2.23 / 70.6 | 1.03 / 76.5 | 7.12 / 75.5 | 3.46 / 74.2 |
| MegaTTS3 | 2.79 / <ins>77.1</ins> | 1.52 / 79.0 | – / – | – / – |
| MiniMax-Speech | 1.65 / 69.2 | **0.83** / 78.3 | – / – | – / – |
| Qwen3-TTS | **1.23** / 71.7 | 1.22 / 77.0 | 6.76 / 74.8 | 3.07 / 74.5 |
| Seed-TTS | 2.25 / 76.2 | 1.12 / 79.6 | 7.59 / 77.6 | 3.65 / 77.8 |
| VibeVoice | 3.04 / 68.9 | 1.16 / 74.4 | – / – | – / – |
| VoxCPM2 | 1.84 / 75.3 | <ins>0.97</ins> / 79.5 | 8.13 / 75.3 | 3.65 / 76.7 |
| dots.tts (Pretrain) | 1.80 / 77.0 | <ins>0.97</ins> / <ins>80.4</ins> | 6.65 / **78.8** | 3.14 / <ins>78.7</ins> |
| **FireRedTTS3-Base** | <ins>1.64</ins> / **77.2** | 1.01 / **80.9** | <ins>6.50</ins> / <ins>78.4</ins> | **3.04** / **78.8** |


### Multilingual Zero-Shot Cloning — MiniMax-MLS-Test

Best in **bold**, second best in <ins>underline</ins>. CER reported for Chinese, Cantonese, Japanese, Korean, Arabic, Vietnamese, Hindi, Thai, and Greek; WER for the rest.

<details>
<summary><b>WER / CER (↓) (click to expand)</b></summary>

| Language | Minimax | ElevenLabs | VoxCPM2 | FishAudio S2 | dots.tts (Pretrain) | **FireRedTTS3** |
| --- | --- | --- | --- | --- | --- | --- |
| Arabic | **1.67** | **1.67** | 13.05 | 3.50 | 37.91 | <ins>1.75</ins> |
| Cantonese | <ins>34.11</ins> | 51.51 | 38.58 | **30.67** | 37.91 | 40.32 |
| Chinese | 2.25 | 16.03 | 1.14 | **0.73** | 1.08 | <ins>0.91</ins> |
| Czech | 3.88 | **2.11** | 24.13 | <ins>2.84</ins> | 5.05 | 3.17 |
| Dutch | 1.14 | **0.80** | <ins>0.91</ins> | 0.99 | 1.20 | 1.15 |
| English | 2.16 | 2.34 | 2.29 | 1.62 | **1.06** | <ins>2.12</ins> |
| Finnish | 4.67 | <ins>2.96</ins> | **2.63** | 3.33 | 3.44 | 3.10 |
| French | 4.10 | 5.22 | 4.53 | **3.05** | <ins>3.82</ins> | 5.28 |
| German | 1.91 | <ins>0.57</ins> | 0.68 | **0.55** | 1.03 | 0.69 |
| Greek | 2.02 | **0.99** | 2.84 | 5.74 | 2.97 | <ins>1.24</ins> |
| Hindi | <ins>6.96</ins> | **5.83** | 19.70 | 14.64 | 14.32 | 7.02 |
| Indonesian | 1.24 | **1.06** | <ins>1.08</ins> | 1.46 | 2.71 | 1.42 |
| Italian | <ins>1.54</ins> | 1.74 | 1.56 | **1.27** | 3.16 | 2.28 |
| Japanese | <ins>3.52</ins> | 10.65 | 4.63 | **2.76** | 7.16 | 3.60 |
| Korean | <ins>1.75</ins> | 1.87 | 1.96 | **1.18** | 5.30 | 2.42 |
| Polish | 1.42 | **0.77** | <ins>1.14</ins> | 1.26 | 2.72 | 1.22 |
| Portuguese | 1.88 | <ins>1.33</ins> | 1.94 | **1.14** | 1.64 | 1.79 |
| Romanian | 2.88 | **1.35** | 21.58 | 10.74 | 3.36 | <ins>1.93</ins> |
| Russian | 4.28 | 3.88 | 3.63 | **2.40** | 3.64 | <ins>3.28</ins> |
| Spanish | 1.03 | 1.08 | 1.44 | **0.91** | <ins>0.96</ins> | 1.21 |
| Thai | <ins>2.70</ins> | 73.94 | 2.96 | 4.23 | 7.45 | **1.87** |
| Turkish | 1.52 | **0.70** | <ins>0.82</ins> | 0.87 | 5.45 | 0.92 |
| Ukrainian | 1.08 | <ins>1.00</ins> | 6.32 | 2.30 | 1.61 | **0.55** |
| Vietnamese | <ins>0.88</ins> | 73.42 | 3.31 | 7.41 | 3.85 | **0.86** |
| **Average** | <ins>3.77</ins> | 10.95 | 6.79 | 4.40 | 6.60 | **3.75** |

</details>

<details>
<summary><b>SIM (↑) (click to expand)</b></summary>

| Language | Minimax | ElevenLabs | VoxCPM2 | FishAudio S2 | dots.tts (Pretrain) | **FireRedTTS3** |
| --- | --- | --- | --- | --- | --- | --- |
| Arabic | 73.6 | 70.6 | **79.1** | 75.0 | 77.5 | <ins>78.9</ins> |
| Cantonese | 77.8 | 67.0 | 83.5 | 80.5 | **84.7** | <ins>83.9</ins> |
| Chinese | 78.0 | 67.7 | <ins>82.5</ins> | 81.6 | 82.3 | **84.2** |
| Czech | 79.6 | 68.5 | 78.3 | 79.8 | <ins>83.8</ins> | **86.1** |
| Dutch | 73.8 | 68.0 | 80.8 | 73.0 | <ins>81.4</ins> | **84.3** |
| English | 75.6 | 61.3 | 85.4 | 79.7 | **86.9** | <ins>86.8</ins> |
| Finnish | 83.5 | 75.9 | <ins>89.0</ins> | 81.9 | 88.0 | **89.9** |
| French | 62.8 | 53.5 | 73.5 | 69.8 | <ins>78.2</ins> | **81.0** |
| German | 73.3 | 61.4 | <ins>80.3</ins> | 76.7 | 79.5 | **83.3** |
| Greek | 82.6 | 73.3 | 86.0 | 79.5 | <ins>87.6</ins> | **89.3** |
| Hindi | 81.8 | 73.0 | <ins>85.6</ins> | 82.1 | 84.5 | **87.2** |
| Indonesian | 72.9 | 66.0 | 80.0 | 76.3 | <ins>80.8</ins> | **83.3** |
| Italian | 69.9 | 57.9 | 78.0 | 74.7 | **84.5** | <ins>83.6</ins> |
| Japanese | 77.6 | 73.8 | <ins>82.8</ins> | 79.6 | **83.1** | <ins>82.8</ins> |
| Korean | 77.6 | 70.0 | 83.3 | 81.7 | <ins>84.3</ins> | **86.6** |
| Polish | 80.2 | 72.9 | <ins>88.4</ins> | 81.9 | 87.3 | **89.8** |
| Portuguese | 80.5 | 71.1 | <ins>83.7</ins> | 78.1 | 83.1 | **86.3** |
| Romanian | <ins>80.9</ins> | 69.9 | 79.7 | 73.3 | **86.2** | **86.2** |
| Russian | 76.1 | 67.6 | 81.1 | 79.0 | <ins>83.0</ins> | **84.7** |
| Spanish | 76.2 | 61.5 | 83.1 | 77.6 | <ins>83.9</ins> | **86.3** |
| Thai | 80.0 | 58.8 | **84.0** | 78.6 | <ins>83.8</ins> | 83.3 |
| Turkish | 77.9 | 59.6 | <ins>87.1</ins> | 83.5 | **87.4** | 86.6 |
| Ukrainian | 73.0 | 64.7 | <ins>79.8</ins> | 74.7 | **80.5** | <ins>79.8</ins> |
| Vietnamese | 74.3 | 36.9 | 80.6 | 74.0 | <ins>80.7</ins> | **81.3** |
| **Average** | 76.6 | 65.5 | 82.3 | 78.0 | <ins>83.5</ins> | **84.8** |

</details>

### Instruct TTS

Since Gemini-2.5-pro-preview is inaccessible, Gemini-2.5-pro is used to score all systems.

<div align="center">

<table style="border-collapse:collapse; margin:0 auto; text-align:center; white-space:nowrap;">
  <thead>
    <tr>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">Model</th>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">ZH<br>APS↑&nbsp;|&nbsp;DSD↑&nbsp;|&nbsp;RP↑</th>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">EN<br>APS↑&nbsp;|&nbsp;DSD↑&nbsp;|&nbsp;RP↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">MOSS-VoiceGenerator</td><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">71.6&nbsp;|&nbsp;72.5&nbsp;|&nbsp;61.3</td><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">58.8&nbsp;|&nbsp;71.8&nbsp;|&nbsp;61.6</td></tr>
    <tr><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">VoiceSculptor-VD</td><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">74.6&nbsp;|&nbsp;63.5&nbsp;|&nbsp;62.0</td><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">–&nbsp;|&nbsp;–&nbsp;|&nbsp;–</td></tr>
    <tr><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">Ming-Omni-TTS-16B-A3B</td><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">84.6&nbsp;|&nbsp;70.7&nbsp;|&nbsp;56.0</td><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">–&nbsp;|&nbsp;–&nbsp;|&nbsp;–</td></tr>
    <tr><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">Qwen3-TTS-VD</td><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">83.7&nbsp;|&nbsp;81.7&nbsp;|&nbsp;65.8</td><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;">76.4&nbsp;|&nbsp;81.4&nbsp;|&nbsp;64.2</td></tr>
    <tr><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;"><b>FireRedTTS3-Instruct</b></td><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;"><b>85.8</b>&nbsp;|&nbsp;<b>82.0</b>&nbsp;|&nbsp;<b>69.7</b></td><td style="text-align:center; padding:6px 14px; border:1px solid #ddd;"><b>80.7</b>&nbsp;|&nbsp;<b>82.3</b>&nbsp;|&nbsp;<b>72.0</b></td></tr>
  </tbody>
</table>

</div>


### Speech Editing

<details>
<summary><b>Semantic Editing (click to expand)</b></summary>

<div align="center">

<table style="border-collapse:collapse; margin:0 auto; text-align:center; white-space:nowrap;">
  <thead>
    <tr>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">Task</th>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">Setting</th>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">Metric</th>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">Ming-UniAudio-Edit<br>zh | en</th>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">FireRedTTS3-Instruct<br>zh | en</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="8"><b>Deletion</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="4"><b>basic</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">WER (%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">11.89 | 14.85</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>10.51</b> | <b>14.46</b></td>
    </tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">SIM↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.78</b> | 0.76</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.78</b> | <b>0.79</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">ACC (%)↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>100.00</b> | 82.22</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>100.00</b> | <b>97.78</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">no-edit WER (%)↓</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">11.49 | 24.26</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>10.30</b> | <b>23.97</b></td></tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="4"><b>open</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">WER (%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">22.92 | 27.60</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>16.31</b> | <b>18.62</b></td>
    </tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">SIM↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.81</b> | 0.74</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.81</b> | <b>0.78</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">ACC (%)↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">82.92 | 85.00</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>89.32</b> | <b>89.50</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">no-edit WER (%)↓</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">17.50 | 35.21</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>11.69</b> | <b>27.08</b></td></tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="8"><b>Insertion</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="4"><b>basic</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">WER (%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>3.42</b> | <b>6.63</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">3.62 | 6.84</td>
    </tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">SIM↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.83</b> | 0.79</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.83</b> | <b>0.83</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">ACC (%)↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">80.00 | 71.43</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>81.18</b> | <b>76.40</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">no-edit WER (%)↓</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>3.52</b> | <b>17.70</b></td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">3.80 | 18.23</td></tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="4"><b>open</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">WER (%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>3.89</b> | <b>7.59</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">4.79 | 9.05</td>
    </tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">SIM↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">0.83 | 0.79</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.84</b> | <b>0.83</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">ACC (%)↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>79.31</b> | 62.31</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>79.31</b> | <b>65.83</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">no-edit WER (%)↓</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>4.10</b> | <b>18.84</b></td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">5.22 | 20.22</td></tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="8"><b>Substitution</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="4"><b>basic</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">WER (%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">4.52 | 8.99</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>2.92</b> | <b>5.63</b></td>
    </tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">SIM↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">0.82 | 0.78</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.83</b> | <b>0.80</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">ACC (%)↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">78.62 | 59.78</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>87.42</b> | <b>75.42</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">no-edit WER (%)↓</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">4.63 | 19.28</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>3.19</b> | <b>17.05</b></td></tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="4"><b>open</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">WER (%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">4.56 | 7.64</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>3.52</b> | <b>6.54</b></td>
    </tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">SIM↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.83</b> | 0.77</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.83</b> | <b>0.80</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">ACC (%)↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">76.62 | 65.62</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>86.15</b> | <b>71.48</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">no-edit WER (%)↓</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">4.75 | <b>18.39</b></td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>3.85</b> | 18.42</td></tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="4"><b>Average</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="4"><b>basic+open</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">WER (%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">8.53 | 12.22</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>6.97</b> | <b>10.22</b></td>
    </tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">SIM↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.82</b> | 0.77</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.82</b> | <b>0.80</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">ACC (%)↑</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">82.91 | 71.06</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>87.27</b> | <b>78.91</b></td></tr>
    <tr><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">no-edit WER (%)↓</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">7.67 | 22.28</td><td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>6.49</b> | <b>20.90</b></td></tr>
  </tbody>
</table>

</div>

</details>

<details>
<summary><b>Acoustic Editing (click to expand)</b></summary>

<div align="center">

<table style="border-collapse:collapse; margin:0 auto; text-align:center; white-space:nowrap;">
  <thead>
    <tr>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">Task</th>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">Metric</th>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">Ming-UniAudio-Edit<br>ZH | EN</th>
      <th style="text-align:center; padding:6px 14px; border:1px solid #ddd; background-color:#f5f5f5;">FireRedTTS3-Instruct<br>ZH | EN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="3"><b>Speed Alteration</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">WER(%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">5.88 | 17.53</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>2.27</b> | <b>4.75</b></td>
    </tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">SIM↑</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">0.66 | 0.57</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.80</b> | <b>0.71</b></td>
    </tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">RDE(%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">6.36 | 5.92</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>4.35</b> | <b>4.29</b></td>
    </tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="2"><b>Pitch Alteration</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">WER(%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">7.45 | 13.37</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>2.34</b> | <b>2.94</b></td>
    </tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">SIM↑</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">0.36 | 0.24</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.51</b> | <b>0.44</b></td>
    </tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;" rowspan="3"><b>Volume Alteration</b></td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">WER(%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">1.71 | 1.35</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>1.69</b> | <b>1.26</b></td>
    </tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">SIM↑</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">0.86 | 0.80</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>0.92</b> | <b>0.90</b></td>
    </tr>
    <tr>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">RAE(%)↓</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;">14.9 | 11.7</td>
      <td style="text-align:center; padding:4px 12px; border:1px solid #ddd;"><b>3.58</b> | <b>4.44</b></td>
    </tr>
  </tbody>
</table>

</div>

</details>

## Usage Disclaimer

- The project incorporates zero-shot voice cloning functionality; Please note that this capability is intended **solely for academic research purposes**.
- **DO NOT** use this model for **ANY illegal activities**❗️❗️
- The developers assume no liability for any misuse of this model.
- If you identify any instances of **abuse**, **misuse**, or **fraudulent** activities related to this project, **please report them to our team immediately.**


## Citation

```bib
@article{fireredtts3,
  title   = {FireRedTTS3: Unified Speech Generation and Editing with Semantically Enriched Speech Representations},
  author  = {Shen, Feiyu and Xie, Kun and Wu, Yichen and Dai, Ziqi and Han, Yichen and Li, Junjie and Geng, Xuelong and Xie, Fenglong and Xie, Lei and Tang, Xu and others},
  journal = {arXiv preprint arXiv:2608.17492},
  year    = {2026},
}
```


## Acknowledgements

- [Qwen3](https://github.com/QwenLM/Qwen3) and [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) for the language model and audio understanding foundations
- [DiTAR](https://arxiv.org/abs/2502.03930) for the patch-level diffusion autoregressive formulation
- [X-Codec](https://github.com/zhenye234/xcodec) for the discriminator design used in RedAE training
- [CAM++](https://modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k) for speaker embedding extraction
- [fastText](https://fasttext.cc/docs/en/language-identification.html) for automatic language identification
- [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) (wetext) for the Chinese / English text normalization front-end


## License

Released under the [Apache-2.0](LICENSE) license.
