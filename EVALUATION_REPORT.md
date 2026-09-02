# Evaluation Report
All first-pass numbers tagged MEASURED_SMOKE (n stated); resubmission adds seeded synthetic
train/val/frozen splits (data/MANIFEST.json hashes) with slices and bootstrap CIs.

## Classification (nb03): smoke accuracy model vs keyword baseline: <!-- paste n and acc -->
Slices ar/en + Macro-F1 + 95% CI (resubmission): <!-- paste -->
## Sentiment head: separate label contract (positive/negative/neutral) added in nb03 v2.
## NER (nb04): entity-level P/R/F1 via span alignment (resubmission): <!-- paste -->
## QA (nb04): EM / token-F1 answerable + no-answer accuracy on frozen no-answer set (target ≥17/20): <!-- paste -->
## Search (nb06): Hits@3 first pass <!-- paste -->; Recall@10 / MRR@10 before vs after cross-encoder re-rank + cross-lingual delta (resubmission): <!-- paste -->
## Invariance / MFT suite: startup canaries (PII email/phone recall 100%, two-copy, router regression) pass rate: <!-- paste -->
## Error analysis
- Named error #1: router `من` misfire → fix D5 (regression-locked).
- Resubmission: errors dumped to reports/errors.csv, ≥100 manual reads, categories (language/length/entity), Top-3 fixes with before/after: <!-- paste -->
## Limitations: demo corpus at first pass; frozen cohort gates (R1–R7 official) pending announced package.