"""
Bayan FastAPI Application.
Serves the NLP pipeline via REST endpoints.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from .engine import BayanEngine
from .router import SmartRouter

app = FastAPI(
    title="Bayan NLP API", 
    description="Bilingual Citizen Feedback System (SDAIA)",
    version="1.0.0"
)
engine = None  # Lazy load to prevent startup timeout

@app.on_event("startup")
def load_engine():
    global engine
    engine = BayanEngine()

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

class QAInput(BaseModel):
    question: str
    context: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Bayan NLP API is running."}

@app.post("/analyze")
def analyze(payload: TextInput):
    """Auto-routes text to the correct NLP task."""
    intent = SmartRouter.route(payload.text)
    if intent == "qa":
        return engine.answer_from_search(payload.text)
    elif intent == "search":
        return engine.semantic_search(payload.text)
    elif intent == "ner":
        return engine.extract_entities(payload.text)
    else:
        return engine.classify(payload.text)

@app.post("/batch/analyze")
def batch_analyze(payload: BatchInput):
    """Extension: Process multiple texts in a single request."""
    results = []
    for text in payload.texts:
        intent = SmartRouter.route(text)
        if intent == "qa":
            result = engine.answer_from_search(text)
        elif intent == "search":
            result = engine.semantic_search(text)
        elif intent == "ner":
            result = engine.extract_entities(text)
        else:
            result = engine.classify(text)
        results.append({"text": text, "result": result})
    return {"batch_size": len(results), "results": results}

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