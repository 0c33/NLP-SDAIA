# Architectural Decisions (R6)
Every family/checkpoint choice below is tied to fertility/slice evidence and
attention/architecture limits. Course: SDA-AIE-211 — Natural Language Processing with Transformers.

## D1 — Encoder-only models, not LLMs
- Context: tasks are discriminative (classify/tag/extract/retrieve); hardware is Intel Ultra 7 CPU, 16GB RAM.
- Alternatives: local 7B+ LLMs (slow on CPU, hallucination risk in QA), paid APIs (not reproducible).
- Evidence: encoders give deterministic spans/labels evaluable with EM/F1; notebook 02 shows the attention mechanism that makes cross-lingual zero-shot possible.
- Trade-off: less flexibility than generation; gained speed, evaluability, honesty.

## D2 — Zero-shot mDeBERTa-v3-mnli-xnli for classification
- Context: 4-day timeline, no labeled Bayan corpus at first pass.
- Evidence: XNLI cross-lingual training lets Arabic premises entail English hypotheses; one canonical English label set avoids splitting confidence between synonymous AR/EN labels.
- Architecture limit: DeBERTa-v3 attention window 512; feedback texts are short (fertility table below), so no truncation pressure.

## D3 — Shared mBERT tokenizer (train/eval/serve), decision via fertility + truncation
- Decision: use `bert-base-multilingual-cased` tokenizer everywhere (R1 skew canary).
- Fertility (tokens/word), measured in notebook 01:
  | slice | mean fertility |
  |---|---|
  | Arabic | <!-- paste from nb01 --> |
  | English | <!-- paste from nb01 --> |
- Truncation checkpoints (share of texts exceeding max_length):
  | max_length | 32 | 64 | 128 | 512 |
  |---|---|---|---|---|
  | truncation rate | <!-- paste --> | <!-- paste --> | <!-- paste --> | <!-- paste --> |
- Chosen MAX_LENGTH=64: zero truncation on feedback-length texts at 4x less padding than 512.

## D4 — Arabic profile `feedback_v1` (versioned, PREPROCESSOR_VERSION=1.0.0)
- Diacritics removed + alef normalized: informal feedback rarely carries diacritics; reduces mBERT vocabulary sparsity.
- Stated trade-off: wrong for diacritic-bearing text (classical/poetry) — out of scope.
- CAMeL Tools: evaluated, not integrated at first pass (no morpheme-level task in scope); integration of `camel_tools` normalization comparison added in resubmission notebook 05.
- Same profile object imported in training, evaluation, serving (one implementation).

## D5 — Router: question-mark gate
- Error found: bare `من`/`ما` matched as interrogatives → complaint "الرد من الموظف" misrouted to QA.
- Fix: require `؟`/`?`; interrogatives only secondary. Regression test in notebook 07 + tests/.

## D6 — Retrieval-grounded QA + no-answer threshold
- QA answers only from retrieved context (micro-RAG); score < 0.15 → null (SQuAD2-style no-answer).

## D7 — FAISS IndexFlatIP (exact) for small corpus
- Exact inner-product on L2-normalized embeddings == cosine; no approximation error at demo scale.
- Resubmission adds cross-encoder re-rank (BAAI/bge-reranker-v2-m3) over top-10 with Recall@10/MRR@10 manifest (notebook 06 v2).

## D8 — Extension: batch endpoint `/batch/analyze`
- Benefit/cost measured in notebook 08 (sequential vs one call), results in BENCHMARKS.md.