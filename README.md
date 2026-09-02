# Bayan | بيان
### Bilingual (Arabic/English) Citizen Feedback NLP System

**Course:** SDAIA Academy — Applied NLP (SDA-AIE-211)
**Author:** Sami Al-Rubaiyan — AI Systems Engineer, SBM Labs
**Architecture:** Modular Python package (`src/bayan/`) + 9 Official Lab Notebooks

---

## About the Author

I design and build AI systems and agents, primarily in Python and Go. Outside of coursework, I run a self-hosted AI infrastructure — currently serving multiple LLMs concurrently (Qwen3.6-35B-A3B/27B, among others) via `llama.cpp` on AMD ROCm, on a Proxmox-based homelab with a fully virtualized OPNsense network layer. My main ongoing project is an agentic architecture that develops new agents around a client's use case through a full feedback loop until the client approves the result.

This background shaped a few decisions in Bayan: preferring a rule-based router over calling every model on every request (a habit from optimizing inference on constrained hardware), and treating "grounded in retrieved context" as a real requirement for the QA component rather than a nice-to-have — the same retrieve-then-answer discipline I apply when building agents that need to stay honest about what they actually know versus what they're inferring.

GitHub: [github.com/0c33](https://github.com/0c33)

---

## نظرة عامة (Arabic Overview)

**بيان** نظام لتحليل ملاحظات المستفيدين ثنائي اللغة (عربي/إنجليزي)، يجمع خمس قدرات أساسية في خط أنابيب واحد:

1. **تنظيف النصوص** — إزالة التشكيل، تطبيع الألف، إخفاء البيانات الحساسة (البريد الإلكتروني وأرقام الجوال السعودية)، وتنظيف الرموز العشوائية.
2. **التصنيف** (Zero-Shot) — تصنيف الملاحظة إلى شكوى / استفسار / اقتراح / مدح، دون الحاجة لتدريب نموذج خاص.
3. **التعرف على الكيانات المسماة (NER)** — استخراج الأشخاص والمنظمات والأماكن من النص.
4. **البحث الدلالي** — إيجاد ملاحظات سابقة مشابهة بالمعنى باستخدام FAISS.
5. **الأسئلة والأجوبة المؤسسة على الاسترجاع** — عند اكتشاف سؤال حقيقي (وجود "؟")، يسترجع النظام أقرب سياق من فهرس البحث الدلالي ثم يستخرج الإجابة منه، بدلاً من الإجابة على سياق ثابت.

يتوفر النظام عبر واجهة API قابلة للاختبار (FastAPI + Swagger)، بالإضافة إلى أدوات اختبار ذاتي وقياس أداء (Benchmark) مدمجة.

---

## What This Project Does

Bayan ingests raw, messy, bilingual citizen feedback text and returns structured analysis — not generated prose. Given a sentence like *"الخدمة سيئة جدا وتأخر الرد من الموظف"*, the pipeline can clean it, classify it as a complaint, extract any named entities, and place it in a searchable semantic index.

**This is a retrieval/analysis system, not a generative one.** The only place text is "generated" is in extractive QA, where the answer is a span lifted directly from retrieved context — never freely composed.

## Pipeline Components

| Stage | Model | Purpose |
|---|---|---|
| **Text Cleaning** | Custom regex + Unicode normalization | HTML/diacritic/tatweel/alef normalization, PII redaction, garbage-token removal, repeated-phrase collapse |
| **Sentence Splitting** | spaCy (`xx`, blank + sentencizer) | Splits cleaned text into sentences |
| **Classification** | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (zero-shot) | Tags feedback as `complaint` / `inquiry` / `suggestion` / `praise` — cross-lingual, no fine-tuning required |
| **NER** | `Davlan/bert-base-multilingual-cased-ner-hrl` | Extracts `PER` / `ORG` / `LOC` entities — trained on Arabic (ANERcorp) among 10 languages |
| **Semantic Search** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + FAISS | Finds semantically similar past feedback across languages |
| **Question Answering** | `deepset/bert-base-multilingual-cased-squad2` | Extractive QA, grounded in the top semantic-search result (retrieve → answer) |

## Why These Design Choices

- **Zero-shot over fine-tuning:** given the course timeline, fine-tuning a classifier on labeled Bayan-specific data wasn't feasible. `mDeBERTa-v3-mnli-xnli` is trained cross-lingually on XNLI, so a single English label set (`complaint`, `inquiry`, `suggestion`, `praise`) works correctly on both Arabic and English input without needing per-language label duplication.
- **Retrieval-grounded QA over a fixed context:** rather than answering every routed question against one hardcoded fact, `answer_from_search()` retrieves the most relevant corpus entry via FAISS first, then extracts an answer from that — a small retrieve-then-answer loop rather than a static demo.
- **Question-mark-gated routing:** `SmartRouter` decides intent without calling every model on every request (saves compute on CPU). It requires an actual question mark (`؟` or `?`) before routing to QA — earlier versions matched bare Arabic words like `من` ("who"/"from") or `ما` ("what"/negation), which are too ambiguous and misrouted ordinary feedback like *"الرد من الموظف"* ("the response **from** the employee") into the QA branch.

## Project Structure

```text
.
├── notebooks/                   # 9 Official SDAIA Lab Notebooks (00-08)
├── src/bayan/                   # Versioned, importable Python package
│   ├── preprocessor.py          # Two-copy contract, PII masking, Arabic profiles
│   ├── engine.py                # Multilingual Transformers (mDeBERTa, mBERT, MiniLM)
│   ├── router.py                # Agentic intent routing with Arabic disambiguation
│   └── api.py                   # FastAPI serving, /health canaries, /batch extension
├── tests/                       # Pytest suite for golden tests and API contracts
├── main.py                      # CLI entry point (test / benchmark / serve)
├── DECISIONS.md                 # Architectural decisions & trade-offs
├── BENCHMARKS.md                # Latency, memory, and INT8 optimization reports
├── EVALUATION_REPORT.md         # Error analysis and metric slices
├── PROJECT_SUMMARY.json         # Official SDAIA submission manifest
└── SUBMISSION.yml               # Validator configuration

```

## Summary Benchmark & Assertion Ledger

| Task | Model / Architecture | Metric | Result | Assertion Status |
| :--- | :--- | :---: | :---: | :---: |
| **Tokenisation** | mBERT WordPiece | Truncation Rate @ 64 | `0%` | `PASS` |
| **PII Masking** | Regex + Canaries | Recall on Golden Set | `100%` | `PASS` |
| **Classification** | mDeBERTa-v3 (Zero-Shot) | Macro-F1 (Smoke) | `> Baseline` | `PASS` |
| **NER** | mBERT (Span Alignment) | Entity Span F1 | `Measured` | `PASS` |
| **Extractive QA** | mBERT-SQuAD2 | No-Answer Accuracy | `Thresholded` | `PASS` |
| **Search** | MiniLM-L12 + FAISS | Hits@3 (Cross-Lingual) | `Measured` | `PASS` |
| **Serving** | FastAPI TestClient | Smoke Test | `200 OK` | `PASS` |

*For complete metric breakdowns and epoch-by-epoch history, refer to `BENCHMARKS.md` and `EVALUATION_REPORT.md`.*

## Setup & Usage

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt

# 1. Run Golden Tests & Startup Canaries
python main.py test

# 2. Run Latency/Memory Benchmark (p50/p95/p99)
python main.py benchmark

# 3. Start the FastAPI Server
python main.py serve
```

With `serve` running, open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

## API Reference

| Endpoint | Method | Input | Description |
|---|---|---|---|
| `/health` | GET | - | Returns startup canaries (PII masking, router regression) |
| `/analyze` | POST | `{"text": "..."}` | Auto-routes to classification / NER / search / QA based on content |
| `/batch/analyze` | POST | `{"texts": ["...", "..."]}` | Measured extension: process multiple texts in one HTTP round-trip |
| `/classify` | POST | `{"text": "..."}` | Force classification |
| `/ner` | POST | `{"text": "..."}` | Force entity extraction |
| `/search` | POST | `{"text": "..."}` | Force semantic search |
| `/qa` | POST | `{"question": "...", "context": "..."}` | Direct extractive QA with your own context |

## Known Limitations (documented, not hidden)

- **Sentence splitting** uses a naive rule (`.`/`!`/`?` boundaries) and incorrectly splits on abbreviations — e.g. Arabic "د." (Dr.) is treated as a sentence end. Fixing this needs an explicit abbreviation exception list.
- **NER model** is a general-purpose 3-class model (`PER`/`ORG`/`LOC`) — it won't catch domain-specific entities (e.g. specific government service names) without further fine-tuning.
- **Semantic search corpus** is a small demo set (6 entries) for illustration. A production deployment would index real historical feedback and likely move from an in-memory FAISS `IndexFlatIP` to a persisted, larger-scale index.
- **Router is rule-based, not learned.** It's fast and interpretable, but any future ambiguous phrasing not anticipated by the regex patterns could still misroute — this trade-off was made deliberately to avoid the cost of running every model on every request.

## Future Work

- Fine-tune classification on real labeled Bayan feedback instead of relying on zero-shot.
- Export models to ONNX Runtime (INT8) for production inference speed.
- Replace the in-memory FAISS index with a persisted vector store as the feedback corpus grows.
- Add an abbreviation-aware sentence splitter for Arabic.

---
**Training-program reference:** [SDAIA Academy — Bayan Applied NLP](https://github.com/almiyead-rgb/bayan-applied-nlp-course)
```
