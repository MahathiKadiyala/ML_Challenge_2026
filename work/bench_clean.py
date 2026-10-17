import time
import re
import unidecode

def clean_slow(text):
    if not text:
        return ""
    text = unidecode.unidecode(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"([a-z])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_fast(text):
    if not text:
        return ""
    if not text.isascii():
        text = unidecode.unidecode(text).lower()
        text = re.sub(r"([a-z])\1+", r"\1", text)
    else:
        text = text.lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())

s = "boulangerie saint michel 15 rue saint michel 75005 paris france"

t0 = time.time()
for _ in range(30000):
    clean_slow(s)
el_slow = time.time() - t0

t0 = time.time()
for _ in range(30000):
    clean_fast(s)
el_fast = time.time() - t0

print(f"Slow: {el_slow:.4f}s ({30000/el_slow:.0f}/s)")
print(f"Fast: {el_fast:.4f}s ({30000/el_fast:.0f}/s)")
print(f"Speedup: {el_slow/el_fast:.1f}x")
