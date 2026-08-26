"""FireRedTTS3 合成管线：在 FireRedTTS3Base / FireRedTTS3Instruct 上增加完整的文本前端处理。

用法示例::

    from fireredtts3.core import FireRedTTS3, FireRedTTS3Instruct

    # ---- Base ----
    tts = FireRedTTS3('pretrained_models')
    gen_audio, gen_audio_sr = tts.generate(
        language=None,          # None 表示自动判定语种
        prompt_text='...',
        prompt_audio=prompt_audio,
        prompt_audio_sr=16000,
        text='这是一段很长的文本，需要被自动切句。第二句。第三句。',
    )
    torchaudio.save('gen.wav', gen_audio.cpu(), gen_audio_sr)

    # ---- Instruct ----
    instruct = FireRedTTS3Instruct('pretrained_models')
    gen_audio, gen_audio_sr, gen_text = instruct.generate_voice_design(
        instruction='一个年轻女性的温柔嗓音，语速稍慢。',
        text='今天天气很好，我们一起去公园散步吧。',
    )
    torchaudio.save('design.wav', gen_audio.cpu(), gen_audio_sr)
"""

import os
import torch
import torchaudio
import numpy as np
from typing import List, Optional, Callable

from fireredtts3.llm.fireredtts3_base import FireRedTTS3Base
from fireredtts3.llm.fireredtts3_instruct import FireRedTTS3Instruct as FireRedTTS3InstructBackend
from fireredtts3.utils.text_normalize import (
    clean_text,
    clean_tn_spaces,
    split_paragraph,
    detect_language,
    lang_tag_to_locale,
    build_llm_normalizer,
    build_wetext_normalizer,
)
from fireredtts3.utils.llm_tn.text_normalizer import TextNormalizer as LlmTextNormalizer


def cross_fade(
    seg_a: torch.Tensor,
    seg_b: torch.Tensor,
    fade_len: int,
) -> torch.Tensor:
    """将两段波形线性 cross-fading 拼接。"""
    if fade_len <= 0:
        return torch.cat([seg_a, seg_b], dim=1)
    fade_len = int(min(fade_len, seg_a.shape[1], seg_b.shape[1]))
    if fade_len <= 0:
        return torch.cat([seg_a, seg_b], dim=1)

    # 前段末尾 fade_len 个点做淡出，后段开头 fade_len 个点做淡入
    ramp = torch.linspace(0.0, 1.0, fade_len, device=seg_a.device, dtype=seg_a.dtype).view(1, -1)

    a_tail = seg_a[:, -fade_len:] * (1.0 - ramp)
    b_head = seg_b[:, :fade_len] * ramp

    overlap = a_tail + b_head

    head = seg_a[:, :-fade_len]
    tail = seg_b[:, fade_len:]
    return torch.cat([head, overlap, tail], dim=1)


class TextFrontendMixin:
    """文本前端（text front-end）能力：清洗 + 语种判定 + TN + 拆句。

    被 :class:`FireRedTTS3` 与 :class:`FireRedTTS3Instruct` 复用，保证
    克隆 / 音色设计 / 编辑等任务共享同一套文本规整逻辑。
    """

    _WETEXT_LANGS = {"Chinese", "English"}  # wetext 仅支持中/英

    def _init_frontend(
        self,
        use_fasttext: bool = True,
        use_llm_tn: bool = False,
        use_wetext: bool = True,
        tn_api_url: Optional[str] = None,
        tn_api_key: Optional[str] = None,
        tn_model: Optional[str] = None,
        tn_kwargs: Optional[dict] = None,
    ):
        """初始化文本前端组件：fasttext 语种检测 + llm_tn / wetext 归一化。"""
        self.use_fasttext = use_fasttext
        self._llm_tn = None
        if use_fasttext:
            try:
                self._llm_tn = LlmTextNormalizer(use_fasttext=True)
            except Exception as e:
                print(f"[WARN] Failed to init llm_tn TextNormalizer for lang detect: {e}", flush=True)
                self._llm_tn = None
        self.use_wetext = use_wetext
        self.wetext_normalizer = None
        self.llm_normalizer = None
        # 默认走 wetext（本地、无需 API key）。llm_tn 是可选增强：仅当显式
        # use_llm_tn=True 且成功构建（即有 .env / 环境变量）时才启用。
        if use_wetext:
            self.wetext_normalizer = build_wetext_normalizer()
        if use_llm_tn:
            self.llm_normalizer = build_llm_normalizer(
                api_url=tn_api_url,
                api_key=tn_api_key,
                model=tn_model,
                **(tn_kwargs or {}),
            )

    def _detect_lang(self, text: str) -> str:
        """判定文本语种，返回如 ``Chinese`` / ``English`` 的 lang tag。"""
        return detect_language(
            text,
            fasttext_detector=self._llm_tn.detect_locale if self._llm_tn is not None else None,
        )

    def _normalize_text(self, text: str, language: str) -> str:
        """TN 回退链：llm_tn → wetext（中/英/方言/粤语）→ 原文。"""
        if self.llm_normalizer is not None:
            locale = lang_tag_to_locale(language) if language else None
            try:
                return clean_tn_spaces(self.llm_normalizer(text, locale=locale))
            except Exception as e:
                print(f"[WARN] llm_tn normalization failed, fallback to raw text: {e}", flush=True)
                return clean_tn_spaces(text)
        _can_wetext = (
            language in self._WETEXT_LANGS
            or language == "Cantonese"
            or language.startswith("ZH_")
        )
        if _can_wetext and self.wetext_normalizer is not None:
            try:
                return clean_tn_spaces(self.wetext_normalizer(text))
            except Exception as e:
                print(f"[WARN] wetext normalization failed, fallback to raw text: {e}", flush=True)
                return clean_tn_spaces(text)
        # 其他语种且无 llm_tn => 仅基础清洗（返回原文）
        return clean_tn_spaces(text)

    def _apply_frontend(
        self,
        text: str,
        language: Optional[str] = None,
        do_clean: bool = True,
        do_tn: bool = True,
        do_split: bool = True,
        token_max_n: int = 80,
        token_min_n: int = 60,
        merge_len: int = 20,
    ) -> "tuple[str, str, List[str]]":
        """对文本执行清洗 / 语种判定 / 拆句 / TN，返回 ``(text, language, sentences)``。

        注：返回的 ``text`` 已按 ``do_split`` 结果合并（未开启拆句则原样返回）。
        """
        if do_clean:
            text = clean_text(text)
        if not text:
            raise ValueError("text is empty after cleaning")

        if language is None:
            language = self._detect_lang(text)

        # zh 按字符分句，非 zh 按 token 分句
        if do_split:
            split_lang = "zh" if language == "Chinese" else "en"
            tokenize = None
            if split_lang != "zh":
                tokenize = lambda s: self._tokenize_text(s).shape[1]
            sentences = split_paragraph(
                text,
                tokenize=tokenize,
                lang=split_lang,
                token_max_n=token_max_n,
                token_min_n=token_min_n,
                merge_len=merge_len,
            )
        else:
            sentences = [text]

        if do_tn:
            sentences = [self._normalize_text(s, language) for s in sentences]
        sentences = [s for s in sentences if s and s.strip()]
        if not sentences:
            raise ValueError("all sentences are empty after normalization")

        return "".join(sentences), language, sentences


class FireRedTTS3(TextFrontendMixin, FireRedTTS3Base):
    """带完整文本前端（清洗 + 语种判定 + 拆句 + 拼接）的零样本 TTS 管线。"""

    def __init__(
        self,
        pretrained_model_dir: str,
        use_fasttext: bool = True,
        use_llm_tn: bool = False,
        use_wetext: bool = True,
        tn_api_url: Optional[str] = None,
        tn_api_key: Optional[str] = None,
        tn_model: Optional[str] = None,
        tn_kwargs: Optional[dict] = None,
    ):
        super().__init__(pretrained_model_dir)
        self._init_frontend(
            use_fasttext=use_fasttext,
            use_llm_tn=use_llm_tn,
            use_wetext=use_wetext,
            tn_api_url=tn_api_url,
            tn_api_key=tn_api_key,
            tn_model=tn_model,
            tn_kwargs=tn_kwargs,
        )

    def _synthesize_one(
        self,
        text: str,
        language: str,
        prompt_text: str,
        prompt_audio: torch.Tensor,
        prompt_audio_sr: int,
        **kwargs,
    ):
        """合成单句文本，返回 ``(gen_audio, gen_audio_sr)``。"""
        return super().generate(
            language=language,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
            prompt_audio_sr=prompt_audio_sr,
            text=text,
            **kwargs,
        )

    def generate(
        self,
        # Input
        text: str,
        language: Optional[str] = None,
        prompt_text: str = "",
        prompt_audio: Optional[torch.Tensor] = None,
        prompt_audio_sr: Optional[int] = None,
        # Inference
        stop_threshold: float = 0.5,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
        seed: int = 1234,
        # 文本前端
        do_clean: bool = True,
        do_tn: bool = True,
        do_split: bool = True,
        token_max_n: int = 80,
        token_min_n: int = 60,
        merge_len: int = 20,
        # 拼接
        cross_fade_ms: float = 50.0,
        max_text_len: int = 300,
    ):
        """合成完整文本，自动清洗、切句、逐句生成并 cross-fading 拼接。"""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        text, language, sentences = self._apply_frontend(
            text=text,
            language=language,
            do_clean=do_clean,
            do_tn=do_tn,
            do_split=do_split,
            token_max_n=token_max_n,
            token_min_n=token_min_n,
            merge_len=merge_len,
        )

        gen_audio_sr = None
        segments: List[torch.Tensor] = []
        for i, sent in enumerate(sentences):
            seg, seg_sr = self._synthesize_one(
                text=sent,
                language=language,
                prompt_text=prompt_text,
                prompt_audio=prompt_audio,
                prompt_audio_sr=prompt_audio_sr,
                stop_threshold=stop_threshold,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                seed=seed,
            )
            gen_audio_sr = seg_sr
            segments.append(seg.cpu())

        gen_audio = segments[0]
        if len(segments) > 1:
            fade_len = int(cross_fade_ms / 1000.0 * gen_audio_sr)
            for s in segments[1:]:
                gen_audio = cross_fade(gen_audio, s, fade_len)
        return gen_audio, gen_audio_sr


class FireRedTTS3Instruct(TextFrontendMixin, FireRedTTS3InstructBackend):
    """FireRedTTS3-Instruct：指令驱动的语音生成与编辑（含文本前端）。

    在 :class:`fireredtts3.llm.fireredtts3_instruct.FireRedTTS3Instruct` 之上，
    叠加与 :class:`FireRedTTS3` 一致的文本前端能力（清洗 / 语种判定 / TN /
    拆句 + cross-fade 拼接），支持 4 类任务：

    - ``generate_tts``              —— ICL 零样本语音克隆（参考音频 + 参考文本）
    - ``generate_voice_design``     —— 音色设计：按自然语言音色描述生成新声音
    - ``generate_semantic_edit``    —— 语义编辑：改词 / 插入 / 删除等内容级编辑
    - ``generate_acoustic_edit``    —— 声学编辑：语速 / 音高 / 音量等声学属性编辑
    """

    def __init__(
        self,
        pretrained_model_dir: str,
        use_fasttext: bool = True,
        use_llm_tn: bool = False,
        use_wetext: bool = True,
        tn_api_url: Optional[str] = None,
        tn_api_key: Optional[str] = None,
        tn_model: Optional[str] = None,
        tn_kwargs: Optional[dict] = None,
    ):
        super().__init__(pretrained_model_dir)  # 先加载 RedAE + InstructCore + tokenizer
        self._init_frontend(
            use_fasttext=use_fasttext,
            use_llm_tn=use_llm_tn,
            use_wetext=use_wetext,
            tn_api_url=tn_api_url,
            tn_api_key=tn_api_key,
            tn_model=tn_model,
            tn_kwargs=tn_kwargs,
        )

    # ------------------------------------------------------------------ #
    # 任务 1: ICL 零样本语音克隆
    # ------------------------------------------------------------------ #
    def generate_tts(
        self,
        prompt_text: str,
        prompt_audio: torch.Tensor,
        prompt_audio_sr: int,
        text: str,
        language: Optional[str] = None,
        # Inference
        stop_threshold: float = 0.5,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
        seed: int = 1234,
        # 文本前端
        do_clean: bool = True,
        do_tn: bool = True,
        do_split: bool = True,
        token_max_n: int = 80,
        token_min_n: int = 60,
        merge_len: int = 20,
        cross_fade_ms: float = 50.0,
    ):
        """ICL 零样本语音克隆（Instruct 版）。

        Args:
            prompt_text: 参考音频对应的文本转写。
            prompt_audio: 参考音频波形 (1, T) 或 (C, T)。
            prompt_audio_sr: 参考音频采样率。
            text: 待合成文本；可包含多句（自动拆句 + cross-fade 拼接）。
            language: 可选语种 tag，为 None 时自动判定。
        """
        text, language, sentences = self._apply_frontend(
            text=text,
            language=language,
            do_clean=do_clean,
            do_tn=do_tn,
            do_split=do_split,
            token_max_n=token_max_n,
            token_min_n=token_min_n,
            merge_len=merge_len,
        )

        gen_audio_sr = None
        segments: List[torch.Tensor] = []
        for sent in sentences:
            seg, seg_sr = super().generate_tts(
                prompt_text=prompt_text,
                prompt_audio=prompt_audio,
                prompt_audio_sr=prompt_audio_sr,
                text=sent,
                stop_threshold=stop_threshold,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                seed=seed,
            )
            gen_audio_sr = seg_sr
            segments.append(seg.cpu())

        gen_audio = segments[0]
        if len(segments) > 1:
            fade_len = int(cross_fade_ms / 1000.0 * gen_audio_sr)
            for s in segments[1:]:
                gen_audio = cross_fade(gen_audio, s, fade_len)
        return gen_audio, gen_audio_sr

    # ------------------------------------------------------------------ #
    # 任务 2: Voice Design —— 按音色描述生成新声音（无需参考音频）
    # ------------------------------------------------------------------ #
    def generate_voice_design(
        self,
        instruction: str,
        text: str,
        language: Optional[str] = None,
        # Audio Inference Settings
        n_timesteps: int = 10,
        inference_cfg: float = 1.2,
        # Random
        seed: int = 2,
        # 文本前端
        do_clean: bool = True,
        do_tn: bool = True,
        do_split: bool = True,
        token_max_n: int = 80,
        token_min_n: int = 60,
        merge_len: int = 20,
        cross_fade_ms: float = 50.0,
    ):
        """Voice Design：从自然语言音色描述生成一段全新语音（无参考音频）。

        模型先输出一段 CoT 语音属性规划（返回 ``gen_text``），再据此合成音频。

        Args:
            instruction: 音色描述，如“一个年轻女性的温柔嗓音，语速稍慢”。
            text: 待合成文本；可包含多句（自动拆句 + 逐句生成 + cross-fade 拼接）。
            language: 可选语种 tag，为 None 时自动判定（用于 TN）。
        Returns:
            ``(gen_audio, gen_audio_sr, gen_text)``。gen_text 为模型输出的
            语音属性规划（CoT）。
        """
        text, language, sentences = self._apply_frontend(
            text=text,
            language=language,
            do_clean=do_clean,
            do_tn=do_tn,
            do_split=do_split,
            token_max_n=token_max_n,
            token_min_n=token_min_n,
            merge_len=merge_len,
        )

        gen_audio_sr = None
        segments: List[torch.Tensor] = []
        gen_text = None
        for i, sent in enumerate(sentences):
            seg, seg_sr, seg_text = super().generate_voice_design(
                instruction=instruction,
                text=sent,
                n_timesteps=n_timesteps,
                inference_cfg=inference_cfg,
                seed=seed,
            )
            gen_audio_sr = seg_sr
            segments.append(seg.cpu())
            if i == 0:
                gen_text = seg_text  # CoT 规划只需取第一次

        gen_audio = segments[0]
        if len(segments) > 1:
            fade_len = int(cross_fade_ms / 1000.0 * gen_audio_sr)
            for s in segments[1:]:
                gen_audio = cross_fade(gen_audio, s, fade_len)
        return gen_audio, gen_audio_sr, gen_text

    # ------------------------------------------------------------------ #
    # 任务 3: Semantic Edit —— 内容级编辑（改词 / 插入 / 删除）
    # ------------------------------------------------------------------ #
    def generate_semantic_edit(
        self,
        instruction: str,
        audio_in: torch.Tensor,
        audio_in_sr: torch.Tensor,
        n_timesteps: int = 10,
        inference_cfg: float = 1.2,
        seed: int = 1234,
    ):
        """Semantic Edit：按自然语言指令对音频进行内容级编辑。

        支持插入 / 删除 / 替换等语义编辑，返回编辑后音频与模型生成的
        编辑后文本（CoT，即 ``<|sot|>{rewritten text}<|eot|>``）。

        Args:
            instruction: 编辑指令，如 "insert '简直' after the character or word at index 8."。
            audio_in: 输入音频波形。
            audio_in_sr: 输入音频采样率。
        Returns:
            ``(gen_audio, gen_audio_sr, gen_text)``。
        """
        return super().generate_semantic_edit(
            instruction=instruction,
            audio_in=audio_in,
            audio_in_sr=audio_in_sr,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            seed=seed,
        )

    # ------------------------------------------------------------------ #
    # 任务 4: Acoustic Edit —— 声学属性编辑（语速 / 音高 / 音量）
    # ------------------------------------------------------------------ #
    def generate_acoustic_edit(
        self,
        instruction: str,
        audio_in: torch.Tensor,
        audio_in_sr: torch.Tensor,
        n_timesteps: int = 10,
        inference_cfg: float = 1.2,
        seed: int = 1234,
    ):
        """Acoustic Edit：按自然语言指令对音频进行声学属性编辑。

        支持调整语速 / 音高 / 音量（如 "adjust the speed to 0.5x"）。

        Args:
            instruction: 编辑指令，需使用模型训练模板，如
                "adjust the speed to 0.5x" / "shift the pitch by 3 steps"。
            audio_in: 输入音频波形。
            audio_in_sr: 输入音频采样率。
        Returns:
            ``(gen_audio, gen_audio_sr)``。
        """
        return super().generate_acoustic_edit(
            instruction=instruction,
            audio_in=audio_in,
            audio_in_sr=audio_in_sr,
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            seed=seed,
        )


if __name__ == '__main__':
    tts = FireRedTTS3('pretrained_models')
    print('[INFO] FireRedTTS3 (text front-end) loaded')

    prompt_audio_path = 'tests/prompts/default_prompt.wav'
    prompt_text = '在欧洲行走简直就是走进汽车博览馆博览会，'
    prompt_audio, prompt_audio_sr = torchaudio.load(prompt_audio_path)

    text = '法院与不动产登记部门加强沟通，并督促银行提前办理抵押预约登记。我们也需要关注后续的进展。'
    gen_audio, gen_audio_sr = tts.generate(
        language=None,          # 自动判定语种
        prompt_text=prompt_text,
        prompt_audio=prompt_audio,
        prompt_audio_sr=prompt_audio_sr,
        text=text,
    )
    torchaudio.save('gen.wav', gen_audio.cpu(), gen_audio_sr)