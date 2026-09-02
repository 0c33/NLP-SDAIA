import html
import re
import unicodedata
from dataclasses import dataclass
import spacy

PREPROCESSOR_VERSION = "1.0.0"

@dataclass
class TextRecord:
    raw_text: str
    model_text: str

class ArabicTextPreprocessor:
    ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
    TATWEEL = "\u0640"
    WHITESPACE = re.compile(r"\s+")
    HTML_TAG = re.compile(r"<[^>]+>")
    EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    SAUDI_MOBILE = re.compile(r"(?:(?:\+|00)?966[\s.-]?|0)?5\d{2}[\s.-]?\d{3}[\s.-]?\d{3}\b")
    COMMA = re.compile(r",")
    LETTER_HYPHEN = re.compile(r"(?<=[A-Za-z])-+(?=[A-Za-z])")
    SINGLE_LETTER_RUN = re.compile(r"\b(?:[A-Za-z]\s+){1,}[A-Za-z]\b")

    def __init__(self, remove_diacritics: bool = True, normalize_alef: bool = True):
        self.remove_diacritics = remove_diacritics
        self.normalize_alef = normalize_alef
        self.nlp = spacy.blank("xx")
        self.nlp.add_pipe("sentencizer")

    def _mask_pii(self, text: str) -> str:
        text = self.EMAIL.sub("[REDACTED_EMAIL]", text)
        text = self.SAUDI_MOBILE.sub("[REDACTED_PHONE]", text)
        return text

    def _join_single_letters(self, match: re.Match) -> str:
        return re.sub(r"\s+", "", match.group(0))

    def _strip_symbol_only_tokens(self, text: str) -> str:
        tokens = text.split()
        kept = [t for t in tokens if re.search(r"[^\W\d_]|\d", t)]
        return " ".join(kept)

    def collapse_repeated_phrases(self, text: str, min_repeats: int = 2) -> str:
        words = text.split()
        n = len(words)
        out = []
        i = 0
        while i < n:
            best_length, best_repeats, best_total = 0, 1, 1
            max_len = (n - i) // min_repeats
            for length in range(1, max_len + 1):
                phrase = words[i:i + length]
                repeats = 1
                j = i + length
                while j + length <= n and words[j:j + length] == phrase:
                    repeats += 1
                    j += length
                total = length * repeats
                if repeats >= min_repeats and total > best_total:
                    best_length, best_repeats, best_total = length, repeats, total
            if best_length and best_repeats >= min_repeats:
                out.extend(words[i:i + best_length])
                i += best_total
            else:
                out.append(words[i])
                i += 1
        return " ".join(out)

    def prepare_text(self, text: str) -> TextRecord:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        raw = text
        model = unicodedata.normalize("NFC", html.unescape(text))
        model = self.HTML_TAG.sub(" ", model).replace(self.TATWEEL, "")
        if self.remove_diacritics:
            model = self.ARABIC_DIACRITICS.sub("", model)
        if self.normalize_alef:
            model = re.sub(r"[إأآٱ]", "ا", model)
        model = self._mask_pii(model)
        model = self.COMMA.sub(" ", model)
        model = self.LETTER_HYPHEN.sub("", model)
        model = self.SINGLE_LETTER_RUN.sub(self._join_single_letters, model)
        model = self._strip_symbol_only_tokens(model)
        model = self.WHITESPACE.sub(" ", model).strip()
        model = self.collapse_repeated_phrases(model)
        return TextRecord(raw_text=raw, model_text=model)

    def split_sentences(self, model_text: str) -> list:
        doc = self.nlp(model_text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

def run_startup_canaries() -> dict:
    from .router import SmartRouter
    p = ArabicTextPreprocessor()
    canaries = {
        "version": PREPROCESSOR_VERSION,
        "pii_email_masked": "test@example.com" not in p.prepare_text("email me at test@example.com").model_text,
        "pii_phone_masked": "0551234567" not in p.prepare_text("call 0551234567").model_text,
        "two_copy_contract": p.prepare_text("test").raw_text == "test",
        "router_disambiguation": SmartRouter.route("الرد من الموظف") == "classification"
    }
    canaries["all_pass"] = all(v for k, v in canaries.items() if isinstance(v, bool))
    return canaries