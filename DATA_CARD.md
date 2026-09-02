# Data Card
Sources: course synthetic samples (Day 1) + 6-doc demo corpus; resubmission adds seeded (seed=42)
synthetic bilingual corpus, case-grouped splits 70/15/15, sha256 hashes in data/MANIFEST.json.
No real citizen data. PII: Saudi-mobile + email regex masking to [REDACTED_*]; canary recall 100% at startup.
Splits frozen before tuning; frozen test read once for the report.
Integrity: AI assistants used for scaffolding/review only; all measurements executed and verified by the student on this hardware; references: SDAIA course repo (almiyead-rgb/bayan-applied-nlp-course).