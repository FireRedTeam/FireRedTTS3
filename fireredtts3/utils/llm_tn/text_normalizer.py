"""LLM-based few-shot Text Normalizer (TN) for TTS front-ends."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_API_URL = os.environ.get("LLM_TN_API_URL")
DEFAULT_API_KEY = os.environ.get("LLM_TN_API_KEY")
DEFAULT_MODEL = os.environ.get("LLM_TN_MODEL")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

SUPPORTED_LOCALES = [
    "ar-SA", "cs-CZ", "de-DE", "el-GR", "en-US", "es-MX",
    "fi-FI", "fr-FR", "hi-IN", "id-ID", "it-IT", "ja-JP",
    "ko-KR", "lt-LT", "nl-NL", "pl-PL", "pt-BR", "ro-RO", "ru-RU",
    "th-TH", "tr-TR", "uk-UA", "vi-VN", "zh-CN",
]

# Meta FastText language-id model (lid.176).
# Download: curl -L -o .../models/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
FASTTEXT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "lid.176.ftz"

# Map FastText ISO-639 language codes to our locale codes.
_FT_LANG_TO_LOCALE = {
    "ar": "ar-SA", "cs": "cs-CZ", "de": "de-DE", "el": "el-GR",
    "en": "en-US", "es": "es-MX", "fi": "fi-FI", "fr": "fr-FR",
    "hi": "hi-IN", "id": "id-ID", "it": "it-IT", "ja": "ja-JP",
    "ko": "ko-KR", "lt": "lt-LT", "nl": "nl-NL", "pl": "pl-PL",
    "pt": "pt-BR", "ro": "ro-RO", "ru": "ru-RU", "th": "th-TH",
    "tr": "tr-TR", "uk": "uk-UA", "vi": "vi-VN", "zh": "zh-CN",
}

# Characters indicative of a specific Latin-script language.
_LT_CHARS = set("ąčęėįšųūž")   # Lithuanian
_DE_CHARS = set("äöüß")         # German
_FR_CHARS = set("àâæçéèêëîïôœùûÿ")  # French
_ES_CHARS = set("ñ¿¡")          # Spanish
_IT_CHARS = set("àèéìíîòóùú")   # Italian

# Stop-words to disambiguate Latin-script languages.
_STOPWORDS: Dict[str, set] = {
    "de-DE": {"und", "der", "die", "das", "ist", "ein", "nicht", "mit", "den",
              "von", "sie", "ich", "auf", "für", "auch", "wird", "im", "am"},
    "en-US": {"the", "and", "is", "are", "of", "to", "in", "for", "on", "with",
              "this", "that", "you", "it", "have", "was", "please", "at"},
    "es-MX": {"el", "la", "los", "las", "de", "que", "es", "un", "una", "por",
              "con", "para", "su", "se", "no", "más", "está", "del"},
    "fr-FR": {"le", "la", "les", "des", "est", "et", "un", "une", "vous", "que",
              "pour", "dans", "qui", "pas", "sur", "au", "du", "ce"},
    "it-IT": {"il", "la", "le", "di", "che", "è", "un", "una", "per", "con",
              "non", "sono", "del", "della", "gli", "nel", "ha", "si"},
}


class TextNormalizer:
    """Few-shot, LLM-driven multilingual text normalizer."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        templates_dir: Path = TEMPLATES_DIR,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 30,
        max_retries: int = 3,
        default_locale: str = "zh-CN",
        use_fasttext: bool = True,
        fasttext_path: Path = FASTTEXT_MODEL_PATH,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.api_url = api_url or DEFAULT_API_URL
        self.api_key = api_key or DEFAULT_API_KEY
        if not self.api_url:
            raise ValueError(
                "TextNormalizer requires an LLM API URL. Set LLM_TN_API_URL in "
                "your environment / .env file, or pass api_url explicitly."
            )
        if not self.api_key:
            raise ValueError(
                "TextNormalizer requires an LLM API key. Set LLM_TN_API_KEY in "
                "your environment / .env file, or pass api_key explicitly."
            )
        self.templates_dir = Path(templates_dir)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_locale = default_locale
        self.stop = []
        self.use_fasttext = use_fasttext
        self.fasttext_path = Path(fasttext_path)
        self._ft_model = None
        self._ft_loaded = False
        self._ft_lock = threading.Lock()
        self._templates: Dict[str, Dict] = {}
        self._load_templates()

    def _get_fasttext(self):
        """Lazily load FastText lid.176 model (thread-safe)."""
        if not self.use_fasttext:
            return None
        if self._ft_loaded:
            return self._ft_model
        with self._ft_lock:
            if self._ft_loaded:
                return self._ft_model
            self._ft_loaded = True
            try:
                import fasttext
                if self.fasttext_path.exists():
                    self._ft_model = fasttext.load_model(str(self.fasttext_path))
                else:
                    self._ft_model = None
            except Exception:
                self._ft_model = None
        return self._ft_model

    def _fasttext_locale(self, text: str):
        """Predict a supported locale via FastText, or None if unavailable."""
        model = self._get_fasttext()
        if model is None:
            return None
        snippet = text.replace("\n", " ").strip()
        if not snippet:
            return None
        try:
            # Use low-level predictor to avoid numpy-2.0 incompatibility in predict().
            with self._ft_lock:
                preds = model.f.predict(snippet, 1, 0.0, "strict")
        except Exception:
            return None
        if not preds:
            return None
        _prob, label = preds[0]
        lang = label.replace("__label__", "")
        return _FT_LANG_TO_LOCALE.get(lang)

    def _load_templates(self) -> None:
        for loc in SUPPORTED_LOCALES:
            path = self.templates_dir / f"{loc}.json"
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    self._templates[loc] = json.load(f)

    def available_locales(self) -> List[str]:
        return sorted(self._templates.keys())

    _TN_SYMBOLS = set("0123456789@#$£€¥%&<>=^`~/\\+*°²³×÷")

    def needs_normalization(self, text: str) -> bool:
        """Return False for plain prose (no digits/symbols needing TN)."""
        if not text or not text.strip():
            return False
        for ch in text:
            if ch in self._TN_SYMBOLS:
                return True
        if re.search(r"https?://|www\.|[\w.+-]+@[\w-]+\.", text):
            return True
        if re.search(r"\b[A-Z]{2,}\b", text):
            return True
        if re.search(r"\b[IVXLCDM]{2,}\b", text):
            return True
        if re.search(r"\b[A-Za-zÀ-ÿ]{1,4}\.", text):
            return True
        return False

    @staticmethod
    def _script_stats(text: str) -> Dict[str, int]:
        stats = {"han": 0, "kana": 0, "latin": 0, "hangul": 0, "cyrillic": 0,
                 "arabic": 0, "devanagari": 0, "thai": 0, "greek": 0}
        for ch in text:
            code = ord(ch)
            if 0x3040 <= code <= 0x30FF:  # kana
                stats["kana"] += 1
            elif (0x4E00 <= code <= 0x9FFF or
                  0x3400 <= code <= 0x4DBF or
                  0xF900 <= code <= 0xFAFF or
                  0x20000 <= code <= 0x2FA1F):
                stats["han"] += 1
            elif 0xAC00 <= code <= 0xD7AF:  # hangul
                stats["hangul"] += 1
            elif 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F:  # cyrillic
                stats["cyrillic"] += 1
            elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:  # arabic
                stats["arabic"] += 1
            elif 0x0900 <= code <= 0x097F:  # devanagari
                stats["devanagari"] += 1
            elif 0x0E00 <= code <= 0x0E7F:  # thai
                stats["thai"] += 1
            elif 0x0370 <= code <= 0x03FF:  # greek
                stats["greek"] += 1
            elif ch.isalpha() and code < 0x250:  # latin
                stats["latin"] += 1
        return stats

    def _detect_latin_locale(self, text: str) -> Optional[str]:
        """Disambiguate Latin text among de/en/es/fr/it/lt; None if ambiguous."""
        low = text.lower()
        chars = set(low)
        if chars & _LT_CHARS:
            return "lt-LT"
        if "ß" in low or (chars & _DE_CHARS):
            de_signal = "ß" in low
        else:
            de_signal = False
        if "ñ" in low or "¿" in low or "¡" in low:
            return "es-MX"
        words = re.findall(r"[a-zà-ÿ]+", low)
        if not words:
            return None
        wordset = set(words)
        scores = {loc: len(wordset & sw) for loc, sw in _STOPWORDS.items()}
        if chars & _DE_CHARS:
            scores["de-DE"] += 1
        if de_signal:
            scores["de-DE"] += 2
        if chars & _FR_CHARS:
            scores["fr-FR"] += 1
        if chars & _IT_CHARS:
            scores["it-IT"] += 1
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return None
        ranked = sorted(scores.values(), reverse=True)
        if len(ranked) > 1 and ranked[0] == ranked[1]:
            return None
        return best

    def detect_locale(self, text: str, use_llm_fallback: bool = True,
                      fallback_locale: Optional[str] = None) -> str:
        """Detect the locale of *text* among SUPPORTED_LOCALES."""
        fb = fallback_locale or self.default_locale
        stats = self._script_stats(text)
        if stats["kana"] > 0:
            return "ja-JP"
        if stats["han"] > 0:
            return "zh-CN"
        if re.search(r"[\u3000-\u303F\uFF01-\uFF0F\uFF1A-\uFF20]", text):
            return "zh-CN"
        if any(stats[k] > 0 for k in ("hangul", "cyrillic", "arabic",
                                      "devanagari", "thai", "greek")):
            ft_guess = self._fasttext_locale(text)
            if ft_guess is not None:
                return ft_guess
            if use_llm_fallback:
                llm_guess = self._detect_locale_llm(text)
                if llm_guess in SUPPORTED_LOCALES:
                    return llm_guess
            return fb
        if stats["latin"] == 0 or not re.search(r"[A-Za-zÀ-ÿ]{3,}", text):
            return fb
        ft_guess = self._fasttext_locale(text)
        if ft_guess is not None:
            return ft_guess
        guess = self._detect_latin_locale(text)
        if guess is not None:
            return guess
        if use_llm_fallback:
            llm_guess = self._detect_locale_llm(text)
            if llm_guess in SUPPORTED_LOCALES:
                return llm_guess
        return fb

    def _detect_locale_llm(self, text: str) -> Optional[str]:
        prompt = (
            "Identify the language of the text below and answer with EXACTLY one "
            "of these locale codes and nothing else: "
            + ", ".join(SUPPORTED_LOCALES)
            + f".\n\nText: {text}\nLocale:"
        )
        messages = [
            {"role": "system", "content": "You are a precise language identifier."},
            {"role": "user", "content": prompt},
        ]
        try:
            out = self._chat(messages, max_tokens=16)
        except Exception:
            return None
        out = out.strip()
        for loc in SUPPORTED_LOCALES:
            if loc.lower() in out.lower():
                return loc
        lang = out.lower()[:2]
        for loc in SUPPORTED_LOCALES:
            if loc.lower().startswith(lang):
                return loc
        return None

    def _build_messages(self, text: str, locale: str) -> List[Dict[str, str]]:
        tmpl = self._templates.get(locale)
        if tmpl is None:
            raise ValueError(f"No template available for locale '{locale}'")
        messages: List[Dict[str, str]] = [{"role": "system", "content": tmpl["system"]}]
        for ex in tmpl.get("examples", []):
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})
        messages.append({"role": "user", "content": text})
        return messages

    @staticmethod
    def _clean_output(text: str, raw_input: str) -> str:
        text = text.split("\n\n", 1)[0]
        out = text.strip()
        if len(out) >= 2 and out[0] in "\"'“”「" and out[-1] in "\"'“”」":
            out = out[1:-1].strip()
        return out or raw_input

    def normalize(
        self,
        text: str,
        locale: Optional[str] = None,
        auto_detect: bool = True,
        force: bool = False,
        fallback_locale: Optional[str] = None,
    ) -> str:
        """Normalize *text* into its spoken form."""
        if text is None:
            return text
        if not text.strip():
            return text
        if not force and not self.needs_normalization(text):
            return text
        if locale is None:
            if auto_detect:
                locale = self.detect_locale(text, fallback_locale=fallback_locale)
            else:
                locale = fallback_locale or self.default_locale
        if locale not in self._templates:
            locale = fallback_locale or self.default_locale
        messages = self._build_messages(text, locale)
        result = self._chat(messages)
        return self._clean_output(result, text)

    def _build_request(self, messages: List[Dict[str, str]], max_tokens: int):
        """Build OpenAI-compatible chat/completions request.

        CoT disabled based on model name:
          ``deepseek`` -> ``thinking: {"type": "disabled"}``
          ``qwen``     -> ``chat_template_kwargs: {"enable_thinking": False}``
        """
        model = (self.model or "").lower()
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if "deepseek" in model:
            payload["thinking"] = {"type": "disabled"}
        elif "qwen" in model:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if self.stop:
            payload["stop"] = self.stop
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        return payload, headers

    def _parse_response(self, obj: Dict) -> str:
        msg = obj["choices"][0]["message"]
        content = msg.get("content")
        if content:
            return content
        reasoning = msg.get("reasoning_content")
        if reasoning:
            return reasoning
        raise RuntimeError("empty completion content")

    def _chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        payload, headers = self._build_request(messages, max_tokens or self.max_tokens)
        data = json.dumps(payload).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    self.api_url, data=data, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                obj = json.loads(body)
                return self._parse_response(obj)
            except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError,
                    KeyError, IndexError, json.JSONDecodeError) as exc:
                last_err = exc
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM request failed after {self.max_retries} retries: {last_err}")


if __name__ == "__main__":
    import sys
    tn = TextNormalizer()
    if len(sys.argv) > 1:
        sample = " ".join(sys.argv[1:])
        loc = tn.detect_locale(sample)
        print(f"[locale={loc}] {tn.normalize(sample)}")
    else:
        demos = [
            "He bought 2.5 kg of apples for $5.99 on 10/12/2023.",
            "会议定于2023年5月20日举行，电话138-1234-5678。",
            "这是一句不需要归一化的普通中文。",
            "Le vol AF123 partira le 1er juin à 18h45.",
        ]
        for d in demos:
            loc = tn.detect_locale(d)
            need = tn.needs_normalization(d)
            print(f"\nINPUT : {d}")
            print(f"locale={loc}  needs_tn={need}")
            print(f"OUTPUT: {tn.normalize(d)}")