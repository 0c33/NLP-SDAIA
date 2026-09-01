"""
=============================================================================
PROJECT: BAYAN - Bilingual Citizen Feedback NLP System
AUTHOR:  Sami (AI Systems Engineer)
COURSE:  SDAIA Academy Applied NLP
=============================================================================
Architecture: Single-file modular pipeline.
Models: mDeBERTa-v3 (Zero-Shot), mBERT (NER/QA), MiniLM (Search + FAISS).
Hardware: Optimized for Intel Ultra 7 / CPU (16GB RAM).

Run:
    python main.py test        - run Day 1 preprocessing/tokenization tests
    python main.py benchmark   - run latency/memory benchmark across all tasks
    python main.py serve       - start the FastAPI server (http://127.0.0.1:8000/docs)

Setup:
    pip install spacy numpy faiss-cpu psutil pandas tabulate \
        transformers sentence-transformers fastapi uvicorn pydantic
    python -m spacy download xx_ent_wiki_sm   (only needed if you fall back to spaCy NER)
=============================================================================
"""

import sys
import html
import re
import unicodedata
import time
import os
import psutil
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict, Any

import spacy
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
from tabulate import tabulate

# ==========================================
# CONFIGURATION & SEEDS
# ==========================================
SEED = 42
np.random.seed(SEED)

CLASSIFICATION_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"  # covers Arabic - trained on ANERcorp
QA_MODEL = "deepset/bert-base-multilingual-cased-squad2"
SEARCH_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MBERT_TOKENIZER = "bert-base-multilingual-cased"

# FIX: one canonical (English) label set instead of duplicating each label
# in Arabic + English. mDeBERTa-v3-mnli-xnli is trained cross-lingually
# specifically so Arabic premises can match English hypothesis labels -
# passing both language versions as separate candidates just splits the
# model's confidence between two labels that mean the same thing.
CLASSIFICATION_LABELS = ["complaint", "inquiry", "suggestion", "praise"]

# ==========================================
# MODULE 1: DAY 1 PREPROCESSING ENGINE
# ==========================================
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
        # Used by split_sentences() below - kept from Day 1 and actually
        # wired in now (previously initialized but never called).
        self.nlp = spacy.blank("xx")
        self.nlp.add_pipe("sentencizer")

    def _mask_pii(self, text: str) -> str:
        return self.SAUDI_MOBILE.sub("[REDACTED_PHONE]", self.EMAIL.sub("[REDACTED_EMAIL]", text))

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

    def split_sentences(self, model_text: str) -> List[str]:
        """Split already-cleaned text into sentences.

        Known limitation (documented, not fixed): naive '.' splitting
        breaks on abbreviations, e.g. Arabic "د." (Dr.) is treated as a
        sentence end. Fixing this needs an abbreviation exception list.
        """
        doc = self.nlp(model_text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]


# ==========================================
# MODULE 2: BAYAN NLP ENGINE & SMART ROUTER
# ==========================================
class BayanEngine:
    def __init__(self):
        print("🔄 [1/5] Loading Preprocessor & Tokenizer...")
        self.preprocessor = ArabicTextPreprocessor()
        self.hf_tokenizer = AutoTokenizer.from_pretrained(MBERT_TOKENIZER, use_fast=True)

        print("🔄 [2/5] Loading Classification Model (mDeBERTa-v3)...")
        self.classifier = pipeline("zero-shot-classification", model=CLASSIFICATION_MODEL, device=-1)
        self.labels = CLASSIFICATION_LABELS

        print("🔄 [3/5] Loading NER Model (mBERT)...")
        self.ner = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple", device=-1)

        print("🔄 [4/5] Loading Extractive QA Model (mBERT-SQuAD)...")
        self.qa = pipeline("question-answering", model=QA_MODEL, device=-1)

        print("🔄 [5/5] Loading Semantic Search (MiniLM + FAISS)...")
        self.embedder = SentenceTransformer(SEARCH_MODEL, device="cpu")
        self._init_faiss()
        print("✅ Bayan Engine fully initialized!\n")

    def _init_faiss(self):
        self.corpus = [
            "الخدمة سيئة جدا وتأخر الرد من الموظف", "الموظفون ممتازون وساعدوني كثيرا في إنجاز المعاملة",
            "أريد معرفة طريقة تجديد الهوية الوطنية", "How to reset my password on the portal?",
            "App crashes on login screen every time I open it", "أقترح إضافة ميزة الإشعارات للتطبيق"
        ]
        embeddings = self.embedder.encode(self.corpus, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def classify(self, text: str) -> dict:
        start = time.time()
        clean_text = self.preprocessor.prepare_text(text).model_text
        res = self.classifier(clean_text, self.labels)
        return {
            "task": "classification",
            "label": res["labels"][0],
            "score": round(res["scores"][0], 3),
            "latency_ms": round((time.time() - start) * 1000, 2),
        }

    def extract_entities(self, text: str) -> dict:
        start = time.time()
        clean_text = self.preprocessor.prepare_text(text).model_text
        entities = self.ner(clean_text)
        clean_ents = [{"word": e["word"], "type": e["entity_group"], "score": round(e["score"], 3)} for e in entities]
        return {"task": "ner", "entities": clean_ents, "latency_ms": round((time.time() - start) * 1000, 2)}

    def answer_question(self, question: str, context: str) -> dict:
        start = time.time()
        res = self.qa(question=question, context=context)
        return {
            "task": "qa",
            "answer": res["answer"],
            "score": round(res["score"], 3),
            "latency_ms": round((time.time() - start) * 1000, 2),
        }

    def semantic_search(self, query: str, top_k: int = 2) -> dict:
        start = time.time()
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(q_emb), top_k)
        results = [
            {"document": self.corpus[idx], "score": round(float(scores[0][i]), 3)}
            for i, idx in enumerate(indices[0])
        ]
        return {"task": "semantic_search", "results": results, "latency_ms": round((time.time() - start) * 1000, 2)}

    def answer_from_search(self, question: str, top_k: int = 1) -> dict:
        """QA grounded in the search index instead of a hardcoded fact -
        retrieves the most relevant corpus entry and uses it as context.
        This is the small RAG-style loop: retrieve, then extract an answer
        from what was retrieved, rather than answering from a fixed string.
        """
        search_result = self.semantic_search(question, top_k=top_k)
        if not search_result["results"]:
            return {"task": "qa", "answer": None, "score": 0.0, "note": "no context available"}
        context = " ".join(r["document"] for r in search_result["results"])
        return self.answer_question(question, context)


class SmartRouter:
    """Pure Python agentic router to save compute - avoids calling every
    model on every request by guessing intent from the text first.

    FIX: the previous version matched bare "من" (who/from) and "ما"
    (what/negation) as question signals. Both words are highly ambiguous
    in Arabic - "الرد من الموظف" ("the response FROM the employee") is
    ordinary feedback, not a question, but contains "من". Routing now
    requires an actual question mark before considering it a QA intent,
    with interrogative words only as supporting (not sole) evidence.
    """

    QUESTION_MARK = re.compile(r"[؟?]")
    INTERROGATIVE = re.compile(r"\b(كيف|أين|متى|لماذا|هل|how|where|when|why|which)\b", re.IGNORECASE)
    SEARCH_HINT = re.compile(r"ابحث|وثائق|مستندات|search|find|documents", re.IGNORECASE)
    NER_HINT = re.compile(r"استخرج|كيانات|أسماء|extract|entities", re.IGNORECASE)

    @staticmethod
    def route(text: str) -> str:
        if SmartRouter.QUESTION_MARK.search(text):
            return "qa"
        if SmartRouter.SEARCH_HINT.search(text):
            return "search"
        if SmartRouter.NER_HINT.search(text):
            return "ner"
        return "classification"


# ==========================================
# MODULE 3: FASTAPI APPLICATION
# ==========================================
app = FastAPI(title="Bayan NLP API", description="Bilingual Citizen Feedback System (SDAIA)")
engine = None  # Lazy load to prevent timeout on startup


@app.on_event("startup")
def load_engine():
    global engine
    engine = BayanEngine()


class TextInput(BaseModel):
    text: str


class QAInput(BaseModel):
    question: str
    context: str


@app.post("/analyze")
def analyze(payload: TextInput):
    intent = SmartRouter.route(payload.text)
    if intent == "qa":
        # FIX: previously answered every routed question against a
        # hardcoded Aramco fact regardless of what was actually asked.
        # Now grounds the answer in the semantic search index instead.
        return engine.answer_from_search(payload.text)
    elif intent == "search":
        return engine.semantic_search(payload.text)
    elif intent == "ner":
        return engine.extract_entities(payload.text)
    else:
        return engine.classify(payload.text)


@app.post("/classify")
def classify(payload: TextInput):
    return engine.classify(payload.text)


@app.post("/ner")
def ner(payload: TextInput):
    return engine.extract_entities(payload.text)


@app.post("/search")
def search(payload: TextInput):
    return engine.semantic_search(payload.text)


@app.post("/qa")
def qa(payload: QAInput):
    return engine.answer_question(payload.question, payload.context)


@app.get("/")
def root():
    return {"status": "ok", "message": "Bayan NLP API is running."}


# ==========================================
# MODULE 4: EVALUATION & BENCHMARKING
# ==========================================
def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_benchmark():
    print("🔥 Starting Bayan Inference Benchmark...")
    eng = BayanEngine()

    test_ar = "الخدمة سيئة جدا وتأخر الرد من الموظف في فرع الرياض"
    test_en = "How can I reset my password?"

    results = []

    mem = get_memory_mb()
    res = eng.classify(test_ar)
    results.append({"Task": "Classification (Ar)", "Latency (ms)": res["latency_ms"], "Memory Delta (MB)": round(get_memory_mb() - mem, 2)})

    mem = get_memory_mb()
    res = eng.extract_entities(test_ar)
    results.append({"Task": "NER (Ar)", "Latency (ms)": res["latency_ms"], "Memory Delta (MB)": round(get_memory_mb() - mem, 2)})

    mem = get_memory_mb()
    res = eng.answer_from_search("متى تأسست أرامكو؟")
    results.append({"Task": "Extractive QA (Ar, search-grounded)", "Latency (ms)": res["latency_ms"], "Memory Delta (MB)": round(get_memory_mb() - mem, 2)})

    mem = get_memory_mb()
    res = eng.semantic_search(test_en)
    results.append({"Task": "Semantic Search (En)", "Latency (ms)": res["latency_ms"], "Memory Delta (MB)": round(get_memory_mb() - mem, 2)})

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("📊 BAYAN INFERENCE BENCHMARK (Local CPU)")
    print("=" * 60)
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
    print("\n💡 Optimization Note: For production, models should be exported to ONNX Runtime (INT8).")


# ==========================================
# MODULE 5: DAY 1 TESTS (Preserved + sentence splitting restored)
# ==========================================
def run_day1_tests():
    print("🧪 Running Day 1 Preprocessing & Tokenization Tests...")
    samples = [
        "أهلاً وسهلاً بكم في برنامج بيان! ", "مرحبــاً\u00a0بكم ",
        "Contact us at learner@example.org ", "للتجربة فقط: 0551234567 ",
        "Natural language processing connects text and models. ",
        "Hi i'm Sami, an Ai Engineer ", "Th--is, i,s @#$%^%$@#$%^ & )( &^%$#@!) Sami L A B ",
        "Sure! I can help you Sure! I can help you Sure! I can help you "
    ]

    preprocessor = ArabicTextPreprocessor()
    records = [preprocessor.prepare_text(text) for text in samples]

    assert records[1].raw_text != records[1].model_text
    assert "learner@example.org" not in records[2].model_text
    assert "0551234567" not in records[3].model_text
    assert records[2].raw_text == samples[2]
    print("✅ Two-copy preprocessing contract=PASS")

    # Sentence splitting (previously initialized but never tested/used)
    sentence_examples = {
        "ar": "الخدمة جيدة. لم يصل الرمز!",
        "en": "The portal stopped. Please retry.",
    }
    sentence_splits = {
        lang: preprocessor.split_sentences(preprocessor.prepare_text(text).model_text)
        for lang, text in sentence_examples.items()
    }
    assert sentence_splits["ar"] == ["الخدمة جيدة.", "لم يصل الرمز!"]
    assert sentence_splits["en"] == ["The portal stopped.", "Please retry."]
    print("✅ Sentence splitting=PASS")

    # Router sanity check - the fix from earlier
    ordinary_feedback_with_min = "الخدمة سيئة جدا وتأخر الرد من الموظف"  # contains "من", not a question
    real_question = "كيف يمكنني إعادة تعيين كلمة المرور؟"  # has "؟"
    assert SmartRouter.route(ordinary_feedback_with_min) == "classification"
    assert SmartRouter.route(real_question) == "qa"
    print("✅ Router disambiguation=PASS")

    # Tokenization Metrics
    hf_tokenizer = AutoTokenizer.from_pretrained(MBERT_TOKENIZER)
    model_texts = [r.model_text for r in records]

    def word_count(text):
        return max(1, len(text.split()))

    def token_fertility(text):
        tokens = hf_tokenizer.tokenize(text)
        content = [t for t in tokens if t not in {"[CLS]", "[SEP]", "[PAD]"}]
        return len(content) / word_count(text)

    fertilities = [token_fertility(text) for text in model_texts]
    assert all(x > 0 for x in fertilities)
    print("✅ Tokenisation metrics=PASS")
    print("✅ DAY1_NOTEBOOK1_CORE=PASS\n")


# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [test | benchmark | serve]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "test":
        run_day1_tests()
    elif mode == "benchmark":
        run_benchmark()
    elif mode == "serve":
        print("🚀 Starting Bayan FastAPI Server on http://127.0.0.1:8000")
        print("📄 Swagger UI available at: http://127.0.0.1:8000/docs")
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        print("Invalid mode. Use 'test', 'benchmark', or 'serve'.")
