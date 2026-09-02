# Benchmarks (MEASURED on this repo's environment)
Mode: MEASURED_SMOKE at first pass; resubmission numbers below are measured by running
notebook 08 / `python main.py benchmark` on the final pipeline (PROJECT_ARTIFACT for this repo's workload).

## Environment (paste `notebooks/00` output)
- Device: Intel Core Ultra 7 155H, CPU, 16GB RAM; OS Windows 11
- Python/PyTorch/transformers/sentence-transformers/faiss versions: <!-- paste -->

## Protocol
Warm-up 5 iterations excluded; 30 measured repetitions per task; p50/p95/p99, throughput, peak RSS.

## FP32 ladder (run `python main.py benchmark`, paste table)
| Task | p50 | p95 | p99 | throughput | peak RSS |
|---|---|---|---|---|---|
| classification_ar | | | | | |
| ner_ar | | | | | |
| qa_search_grounded | | | | | |
| search_en | | | | | |

## INT8 dynamic quantization (notebook 08)
- Size before/after + reduction %: <!-- paste -->
- Honest caveat: CPU dynamic-quant latency gain varies by op mix; size is the reliably measured effect.
- Quality parity: FP32 vs INT8 label agreement on smoke set: <!-- paste --> (rollback = keep FP32 object).

## Extension A/B — batch endpoint
| mode | total ms (N=5) | throughput |
|---|---|---|
| sequential | <!-- paste --> | |
| /batch/analyze | <!-- paste --> | |
Results identical between modes: yes (same engine, asserted in nb08).