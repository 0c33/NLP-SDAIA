# First Day at Sdaia Academy - 2026/08/30


# إعداد بيئة قابلة لإعادة التشغيل / Reproducible setup
import importlib.util
import subprocess
import sys
import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable
import spacy


import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing

# Conservative Arabic decision

SEED = 42
rng = np.random.default_rng(SEED)
samples = [
    "أهلاً وسهلاً بكم في برنامج بيان!",
    "مرحبــاً\u00a0بكم",
    "Contact us at learner@example.org",
    "للتجربة فقط: 0551234567",
    "Natural language processing connects text and models.",
    "Hi i'm Sami, an Ai Engineer",
    "Th--is, i,s @#$%^%$@#$%^&*)(*&^%$#@!) Sami L A B",
    "Sure! I can help you Sure! I can help you Sure! I can help you Sure! I can help you Sure! I can help you Sure! I can help you Sure! I can help you Sure! I can help you"
]

@dataclass
class TextRecord:
    raw_text: str
    model_text: str

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"
WHITESPACE = re.compile(r"\s+")
HTML_TAG = re.compile(r"<[^>]+>")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
SAUDI_MOBILE = re.compile(r"(?:(?:\+|00)?966[\s.-]?|0)?5\d{2}[\s.-]?\d{3}[\s.-]?\d{3}\b")
NOISE_CHARS = re.compile(r"[#&$*^~`|\\_@%]+")
REPEATED_PUNCT = re.compile(r"([-,.;:!?])\1+")
COMMA = re.compile(r",")
LETTER_HYPHEN = re.compile(r"(?<=[A-Za-z])-+(?=[A-Za-z])")
SINGLE_LETTER_RUN = re.compile(r"\b(?:[A-Za-z]\s+){1,}[A-Za-z]\b")

def collapse_repeated_phrases(text: str, min_repeats: int = 2) -> str:
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
            while words[j:j + length] == phrase:
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


def mask_pii(text: str) -> str: # remove sensitive data/private data from MOBILE NUMBER or EMAIL
    return SAUDI_MOBILE.sub("", EMAIL.sub("", text))


def join_single_letters(match: re.Match) -> str:
    return re.sub(r"\s+", "", match.group(0))


def strip_symbol_only_tokens(text: str) -> str:
    tokens = text.split()
    kept = [t for t in tokens if re.search(r"[^\W\d_]|\d", t)]
    return " ".join(kept)


def inspect_unicode(text: str) -> list[dict[str, str]]:
    return [
        {"char": ch, "code_point": f"U+{ord(ch):04X}", "name": unicodedata.name(ch, "UNKNOWN")}
        for ch in text
    ]

def prepare_text(text: str, *, remove_diacritics: bool = False, normalize_alef: bool = False) -> TextRecord:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    raw = text
    model = unicodedata.normalize("NFC", html.unescape(text))
    model = HTML_TAG.sub(" ", model).replace(TATWEEL, "")
    if remove_diacritics:
        model = ARABIC_DIACRITICS.sub("", model)
    if normalize_alef:
        model = re.sub(r"[إأآٱ]", "ا", model)
    model = mask_pii(model)
    model = COMMA.sub("", model)
    model = LETTER_HYPHEN.sub("", model)
    model = SINGLE_LETTER_RUN.sub(join_single_letters, model)
    model = strip_symbol_only_tokens(model)
    model = WHITESPACE.sub(" ", model).strip()
    model = collapse_repeated_phrases(model)
    return TextRecord(raw_text=raw, model_text=model)

# ls = {} # to store every words character unicode

# <ONLY TEST>

# for i in samples:
#     unicode_rows = inspect_unicode(i)

#     print(f"Sentence: {i}:\n\n")

#     for row in unicode_rows:
#         print(f'{row}\n\n\n')

#     print('================================\n\n\n')

# <END OF ONLY TEST>

records = [prepare_text(text) for text in samples]

print(records)

for record in records:
    print({"raw": record.raw_text, "model": record.model_text})
assert records[1].raw_text != records[1].model_text
assert "learner@example.org" not in records[2].model_text
assert "0551234567" not in records[3].model_text
assert records[2].raw_text == samples[2]
print("Two-copy preprocessing contract=PASS")

# A testable spaCy stage

sentence_nlp = spacy.blank("xx")
sentence_nlp.add_pipe("sentencizer")

def split_model_sentences(text: str) -> list[str]:
    protected = prepare_text(text).model_text
    return [sentence.text.strip() for sentence in sentence_nlp(protected).sents if sentence.text.strip()]

sentence_examples = {
    "ar": "الخدمة جيدة. لم يصل الرمز!",
    "en": "The portal stopped. Please retry.",
    # "en": "Server #02 has crashed, rebooting now!",
    # "en": "Server #05 has successfully started up, and all services are up!"
}
sentence_splits = {language: split_model_sentences(text) for language, text in sentence_examples.items()}
for language, sentences in sentence_splits.items():
    print(language, sentences)

abbreviation_probe = split_model_sentences("راجع د. أحمد. ثم أعد المحاولة.")
print("Known abbreviation probe / حالة تحتاج قاعدة إضافية:", abbreviation_probe)
assert sentence_splits["ar"] == ["الخدمة جيدة.", "لم يصل الرمز!"]
assert sentence_splits["en"] == ["The portal stopped.", "Please retry."]
print("SPACY_SENTENCE_PIPELINE=PASS")


# Subword tokenisation

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# [PAD] (0): Padding for equal-length sequences.

# [UNK] (1): Represents unknown out-of-vocabulary words.

# [CLS] (2): Placed at the start of every sequence (used for sequence-level tasks).

# [SEP] (3): Placed at the end or between two sentences as a delimiter.

# [MASK] (4): Used during masked language model pre-training.


known_tokens = [
    "أهلاً", "وسهلاً", "بكم", "في", "برنامج", "بيان", "!", "مرحبا",
    "Contact", "us", "at", "<", "EMAIL", ">", "Natural", "language",
    "processing", "connects", "text", "and", "models", ".",
]
vocab_tokens = special_tokens + known_tokens
vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
local_tokenizer = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
local_tokenizer.pre_tokenizer = BertPreTokenizer()
local_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
)
for record in records:
    encoded = local_tokenizer.encode(record.model_text)
    print(record.model_text)
    print("tokens:", encoded.tokens)
    print("ids:   ", encoded.ids)
assert local_tokenizer.encode("مرحبا بكم").tokens == ["[CLS]", "مرحبا", "بكم", "[SEP]"]
print("Local WordPiece demonstration=PASS")


# Two pre-selection metrics

def word_count(text: str) -> int:
    return max(1, len(text.split()))


def token_fertility(tokenizer: Tokenizer, text: str) -> float:
    tokens = tokenizer.encode(text).tokens
    content = [t for t in tokens if t not in {"[CLS]", "[SEP]", "[PAD]"}]
    return len(content) / word_count(text)


def truncation_rate(tokenizer: Tokenizer, texts: Iterable[str], max_length: int) -> float:
    texts = list(texts)
    if not texts:
        raise ValueError("texts must not be empty")
    return sum(len(tokenizer.encode(t).ids) > max_length for t in texts) / len(texts)

model_texts = [record.model_text for record in records]
fertilities = [token_fertility(local_tokenizer, text) for text in model_texts]
rate_at_10 = truncation_rate(local_tokenizer, model_texts, max_length=10)
print("fertility per sample:", [round(x, 2) for x in fertilities])
print("mean fertility:", round(float(np.mean(fertilities)), 2))
print("truncation rate @10:", f"{rate_at_10:.0%}")
assert all(x > 0 for x in fertilities)
assert 0.0 <= rate_at_10 <= 1.0
print("Tokenisation metrics=PASS")


MAX_LENGTH = 12
local_tokenizer.enable_truncation(max_length=MAX_LENGTH)
local_tokenizer.enable_padding(length=MAX_LENGTH, pad_id=vocab["[PAD]"], pad_token="[PAD]")
batch = [local_tokenizer.encode(text) for text in model_texts[:2]]
input_ids = np.array([item.ids for item in batch], dtype=np.int64)
attention_mask = np.array([item.attention_mask for item in batch], dtype=np.int64)
embedding_table = rng.normal(0.0, 0.02, size=(len(vocab), 8))
embeddings = embedding_table[input_ids]
print("input_ids shape:", input_ids.shape)
print("attention_mask shape:", attention_mask.shape)
print("embeddings shape:", embeddings.shape)
print("first attention mask:", attention_mask[0].tolist())
assert input_ids.shape == (2, MAX_LENGTH)
assert attention_mask.shape == input_ids.shape
assert embeddings.shape == (2, MAX_LENGTH, 8)
print("IDs-to-embeddings pipeline=PASS")


try:
    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased", use_fast=True)
    comparison_text = "معالجة اللغة الطبيعية مفيدة Natural language processing is useful"

    for i in 
    hf_tokens = hf_tokenizer.tokenize(comparison_text)
    print("mBERT tokens:", hf_tokens)
    print("mBERT token count:", len(hf_tokens))
    print("Fast tokenizer:", hf_tokenizer.is_fast)
except Exception as exc:
    print("Optional mBERT download unavailable; continue with Core.")
    print("Reason:", type(exc).__name__)