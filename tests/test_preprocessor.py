import sys
sys.path.insert(0, 'src')
from bayan import ArabicTextPreprocessor, run_startup_canaries

def test_pii_masking():
    p = ArabicTextPreprocessor()
    r = p.prepare_text("Contact learner@example.org or 0551234567")
    assert "learner@example.org" not in r.model_text
    assert "0551234567" not in r.model_text

def test_canaries():
    c = run_startup_canaries()
    assert c["all_pass"] is True