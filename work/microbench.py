import time
import re
import unidecode
from rapidfuzz import fuzz

def clean_fast(text):
    if not text:
        return ""
    if not text.isascii():
        text = unidecode.unidecode(text).lower()
        text = re.sub(r"([a-z])\1+", r"\1", text)
    else:
        text = text.lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())

s1_names = ["societe generale paris", "boulangerie saint michel", "pharmacie centrale de lyon", "restaurant chez jean"] * 2500
tg_names = ["societe generale", "boulangerie st michel", "pharmacie du centre lyon", "chez jean resto"] * 2500

t0 = time.time()
for s1, tg in zip(s1_names, tg_names):
    c1 = clean_fast(s1)
    c2 = clean_fast(tg)
    f1 = c1.split()[0]
    f2 = c2.split()[0]
    if f1 == f2:
        ok = True
    else:
        ok = fuzz.ratio(c1, c2) >= 45
el = time.time() - t0
print(f"10,000 pairs processed in {el:.4f}s ({10000/el:.0f} pairs/sec)")
