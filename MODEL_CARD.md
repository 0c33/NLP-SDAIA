# Model Cards
## 1) MoritzLaurer/mDeBERTa-v3-base-mnli-xnli — classification
DeBERTa-v3 encoder, ~86M params; zero-shot NLI over labels [complaint, inquiry, suggestion, praise] + sentiment head; 100+ languages; input ≤512 tokens; CPU inference; limit: confidence sensitive to hypothesis wording.
## 2) Davlan/bert-base-multilingual-cased-ner-hrl — NER
mBERT encoder, ~178M; PER/ORG/LOC/MISC; Arabic via ANERCorp; aggregation=simple; limit: domain-specific service names out of scope.
## 3) deepset/bert-base-multilingual-cased-squad2 — extractive QA
Span extraction with SQuAD2 no-answer behavior; threshold 0.15 → null; limit: answer must be a context span.
## 4) paraphrase-multilingual-MiniLM-L12-v2 — embeddings
384-dim distilled encoder; L2-normalized for IndexFlatIP cosine equivalence; limit: short-text bias.
Pins: transformers==<paste>, torch==<paste>.