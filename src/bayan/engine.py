import time
import numpy as np
import faiss
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from .preprocessor import ArabicTextPreprocessor

CLASSIFICATION_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
QA_MODEL = "deepset/bert-base-multilingual-cased-squad2"
SEARCH_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MBERT_TOKENIZER = "bert-base-multilingual-cased"
CLASSIFICATION_LABELS = ["complaint", "inquiry", "suggestion", "praise"]

class BayanEngine:
    def __init__(self):
        print("🔄 Loading Preprocessor & Tokenizer...")
        self.preprocessor = ArabicTextPreprocessor()
        self.hf_tokenizer = AutoTokenizer.from_pretrained(MBERT_TOKENIZER, use_fast=True)

        print("🔄 Loading Classification Model (mDeBERTa-v3)...")
        self.classifier = pipeline("zero-shot-classification", model=CLASSIFICATION_MODEL, device=-1)
        self.labels = CLASSIFICATION_LABELS

        print("🔄 Loading NER Model (mBERT)...")
        self.ner = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple", device=-1)

        print("🔄 Loading Extractive QA Model (mBERT-SQuAD)...")
        self.qa = pipeline("question-answering", model=QA_MODEL, device=-1)

        print("🔄 Loading Semantic Search (MiniLM + FAISS)...")
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
        start = time.perf_counter()
        clean_text = self.preprocessor.prepare_text(text).model_text
        res = self.classifier(clean_text, self.labels)
        return {"task": "classification", "label": res["labels"][0], "score": round(res["scores"][0], 3), "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

    def extract_entities(self, text: str) -> dict:
        start = time.perf_counter()
        clean_text = self.preprocessor.prepare_text(text).model_text
        entities = self.ner(clean_text)
        clean_ents = [{"word": e["word"], "type": e["entity_group"], "score": round(e["score"], 3)} for e in entities]
        return {"task": "ner", "entities": clean_ents, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

    def answer_question(self, question: str, context: str) -> dict:
        start = time.perf_counter()
        res = self.qa(question=question, context=context)
        answer, score = res["answer"], res["score"]
        if score < 0.15: answer = None  # SQuAD2 No-Answer Threshold
        return {"task": "qa", "answer": answer, "score": round(score, 3), "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

    def semantic_search(self, query: str, top_k: int = 2) -> dict:
        start = time.perf_counter()
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(q_emb), top_k)
        results = [{"document": self.corpus[idx], "score": round(float(scores[0][i]), 3)} for i, idx in enumerate(indices[0])]
        return {"task": "semantic_search", "results": results, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

    def answer_from_search(self, question: str, top_k: int = 1) -> dict:
        search_result = self.semantic_search(question, top_k=top_k)
        if not search_result["results"]:
            return {"task": "qa", "answer": None, "score": 0.0, "note": "no context available"}
        context = " ".join(r["document"] for r in search_result["results"])
        return self.answer_question(question, context)