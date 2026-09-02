from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from .engine import BayanEngine
from .router import SmartRouter
from .preprocessor import run_startup_canaries

app = FastAPI(title="Bayan NLP API", description="Bilingual Citizen Feedback System (SDAIA)")
engine = None
CANARIES = {}

@app.on_event("startup")
def load_engine():
    global engine, CANARIES
    CANARIES = run_startup_canaries()
    if not CANARIES["all_pass"]: raise RuntimeError("Startup canaries failed!")
    engine = BayanEngine()

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["الخدمة سيئة جدا", "The app crashes"])

class BatchInput(BaseModel):
    texts: List[str]

class QAInput(BaseModel):
    question: str
    context: str

@app.get("/health")
def health(): return {"status": "ok", "canaries": CANARIES}

@app.post("/analyze")
def analyze(payload: TextInput):
    intent = SmartRouter.route(payload.text)
    if intent == "qa": return engine.answer_from_search(payload.text)
    elif intent == "search": return engine.semantic_search(payload.text)
    elif intent == "ner": return engine.extract_entities(payload.text)
    else: return engine.classify(payload.text)

@app.post("/batch/analyze")
def batch_analyze(payload: BatchInput):
    results = []
    for text in payload.texts:
        intent = SmartRouter.route(text)
        if intent == "qa": result = engine.answer_from_search(text)
        elif intent == "search": result = engine.semantic_search(text)
        elif intent == "ner": result = engine.extract_entities(text)
        else: result = engine.classify(text)
        results.append({"text": text, "result": result})
    return {"batch_size": len(results), "results": results}

@app.post("/classify")
def classify(payload: TextInput): return engine.classify(payload.text)

@app.post("/ner")
def ner(payload: TextInput): return engine.extract_entities(payload.text)

@app.post("/search")
def search(payload: TextInput): return engine.semantic_search(payload.text)

@app.post("/qa")
def qa(payload: QAInput): return engine.answer_question(payload.question, payload.context)