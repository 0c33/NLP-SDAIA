import sys, time, os, psutil, numpy as np
from src.bayan.preprocessor import ArabicTextPreprocessor, run_startup_canaries
from src.bayan.engine import BayanEngine
from transformers import AutoTokenizer

def run_day1_tests():
    print("🧪 Running Day 1 Golden Tests & Canaries...")
    samples = ["أهلاً وسهلاً بكم في برنامج بيان! ", "مرحبــاً\u00a0بكم ", "Contact us at learner@example.org ", "للتجربة فقط: 0551234567 ", "Natural language processing connects text and models. ", "Hi i'm Sami, an Ai Engineer ", "Th--is, i,s @#$%^%$@#$%^ & )( &^%$#@!) Sami L A B ", "Sure! I can help you Sure! I can help you "]
    preprocessor = ArabicTextPreprocessor()
    records = [preprocessor.prepare_text(text) for text in samples]
    assert records[1].raw_text != records[1].model_text
    assert "learner@example.org" not in records[2].model_text
    assert "0551234567" not in records[3].model_text
    assert records[2].raw_text == samples[2]
    print("✅ Two-copy preprocessing contract=PASS")
    canaries = run_startup_canaries()
    assert canaries["all_pass"], "Startup canaries failed!"
    print(f"✅ Startup Canaries (v{canaries['version']})=PASS")
    hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    def token_fertility(text):
        tokens = hf_tokenizer.tokenize(text)
        return len([t for t in tokens if t not in {"[CLS]", "[SEP]", "[PAD]"}]) / max(1, len(text.split()))
    assert all(token_fertility(r.model_text) > 0 for r in records)
    print("✅ Tokenisation metrics=PASS\n✅ DAY1_NOTEBOOK1_CORE=PASS\n")

def run_benchmark(warmup=5, reps=30):
    print("🔥 Starting Bayan Inference Benchmark...")
    eng = BayanEngine()
    ar, en = "الخدمة سيئة جدا وتأخر الرد من الموظف في فرع الرياض", "How can I reset my password?"
    tasks = {"classification_ar": lambda: eng.classify(ar), "ner_ar": lambda: eng.extract_entities(ar), "qa_search": lambda: eng.answer_from_search("كيف أعيد تعيين كلمة المرور؟"), "search_en": lambda: eng.semantic_search(en)}
    print("| Task | p50 | p95 | p99 | RSS_MB |")
    print("|---|---|---|---|---|")
    for name, fn in tasks.items():
        for _ in range(warmup): fn()
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
        p50, p95, p99 = np.percentile(ts, [50, 95, 99])
        mem = round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1)
        print(f"| {name} | {round(float(p50),2)} | {round(float(p95),2)} | {round(float(p99),2)} | {mem} |")

def run_serve():
    import uvicorn
    uvicorn.run("src.bayan.api:app", host="127.0.0.1", port=8000)

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit("Usage: python main.py [test | benchmark | serve]")
    mode = sys.argv[1].lower()
    if mode == "test": run_day1_tests()
    elif mode == "benchmark": run_benchmark()
    elif mode == "serve": run_serve()
