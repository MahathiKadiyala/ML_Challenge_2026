import time
import polars as pl
import unidecode
import re
from rapidfuzz import fuzz
from collections import defaultdict

TEST_DIR = "student_resource/dataset/test"

LEGAL_STOPWORDS = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "pllc", "pc", "co", "company", "gmbh", 
    "services", "enterprises", "consultants", "solutions",
    "technologies", "technology", "management", "trading", "group", "holdings",
    "holding", "associates", "industries", "industry", "lp", "llp",
    "sa", "sarl", "sasu", "sas", "eurl", "sci", "fils", "freres", "cie", "soc",
    "pra", "li", "m/s", "ms", "shri", "smt", "sri", "dr", "mr", "mrs",
    "the", "and", "of", "de", "du", "des", "la", "le", "les", "et"
}

GENERIC_ADDR = {
    "road", "rd", "street", "st", "lane", "ln", "avenue", "ave", "court", "ct",
    "drive", "dr", "boulevard", "blvd", "floor", "fl", "building", "bldg", "near",
    "opp", "opposite", "door", "no", "plot", "shop", "flat", "block", "c/o", "h/no",
    "hno", "house", "room", "sector", "phase", "nagar", "colony", "city", "west",
    "east", "north", "south", "in", "at", "by", "unit", "suite", "ste", "apt", "apartment",
    "rue", "r", "bd", "av", "allee", "allée", "cours", "chemin", "place", "pl"
}

def clean_and_translit_fast(text):
    if not text:
        return ""
    if not text.isascii():
        text = unidecode.unidecode(text).lower()
        text = re.sub(r"([a-z])\1+", r"\1", text)
    else:
        text = text.lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())

def extract_core_tokens(name):
    norm = clean_and_translit_fast(name)
    tokens = norm.split()
    if tokens and tokens[-1] in {"com", "org", "net", "in", "co", "us", "fr"}:
        tokens.pop()
    content_tokens = [t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1]
    return content_tokens

def extract_addr_features(addr):
    norm = clean_and_translit_fast(addr)
    tokens = norm.split()
    raw_nums = [t.lstrip("0") for t in tokens if t.isdigit() and len(t) <= 6]
    nums = [t for t in raw_nums if t]
    addr_words = [t for t in tokens if not t.isdigit() and t not in GENERIC_ADDR and len(t) > 2]
    return nums, addr_words

print("Loading 200k targets...")
s2_df = pl.read_csv(f"{TEST_DIR}/test_source2.tsv", separator="\t").filter(pl.col("country") == "France").head(200000)
s3_df = pl.read_csv(f"{TEST_DIR}/test_source3.tsv", separator="\t").filter(pl.col("country") == "France").head(200000)
target_df = pl.concat([s2_df, s3_df])

tids = target_df["entity_id"].to_list()
names = target_df["business_name"].to_list()
addrs = target_df["business_address"].fill_null("").to_list()

print("Indexing...")
t0 = time.time()
target_dict = {}
idx_sorted_tokens = defaultdict(list)
idx_first2 = defaultdict(list)
idx_num_addr = defaultdict(list)

for tid, name, addr in zip(tids, names, addrs):
    c_tokens = extract_core_tokens(name)
    nums, addr_words = extract_addr_features(addr)
    target_dict[tid] = {
        "tid": tid,
        "clean_name": clean_and_translit_fast(name),
        "clean_addr": clean_and_translit_fast(addr),
        "core_str": " ".join(c_tokens),
        "first1": c_tokens[0] if c_tokens else "",
        "nums_set": set(nums)
    }
    st = " ".join(sorted(c_tokens))
    if st and len(idx_sorted_tokens[st]) < 50:
        idx_sorted_tokens[st].append(tid)
    f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
    if f2 and len(idx_first2[f2]) < 50:
        idx_first2[f2].append(tid)
    if nums and addr_words:
        for num in nums[:2]:
            for word in addr_words[:2]:
                k = (num, word)
                if len(idx_num_addr[k]) < 50:
                    idx_num_addr[k].append(tid)

print(f"Index built in {time.time()-t0:.2f}s.")

s1_df = pl.read_csv(f"{TEST_DIR}/test_source1.tsv", separator="\t").filter(pl.col("country") == "France").head(5000)
s1_tids = s1_df["entity_id"].to_list()
s1_names = s1_df["business_name"].to_list()
s1_addrs = s1_df["business_address"].fill_null("").to_list()

print("Running test on 5,000 S1 records with SHORT-CIRCUITING prefilter...")
t0 = time.time()
passed_pairs = 0
for s1_id, name, addr in zip(s1_tids, s1_names, s1_addrs):
    c_tokens = extract_core_tokens(name)
    nums, addr_words = extract_addr_features(addr)
    s1_clean = clean_and_translit_fast(name)
    s1_nums = set(nums)
    s1_f1 = c_tokens[0] if c_tokens else ""
    
    cands = set()
    st = " ".join(sorted(c_tokens))
    if st in idx_sorted_tokens:
        cands.update(idx_sorted_tokens[st])
    f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else s1_f1
    if f2 in idx_first2:
        cands.update(idx_first2[f2])
    if nums and addr_words:
        for num in nums[:2]:
            for word in addr_words[:2]:
                k = (num, word)
                if k in idx_num_addr:
                    cands.update(idx_num_addr[k])
                    
    for tid in cands:
        tg = target_dict[tid]
        # Short-circuiting pre-filter
        if (s1_nums and tg["nums_set"] and (s1_nums & tg["nums_set"])) or \
           (s1_f1 and tg["first1"] and s1_f1 == tg["first1"]) or \
           (fuzz.ratio(s1_clean, tg["clean_name"]) >= 45):
            passed_pairs += 1

el = time.time() - t0
print(f"5,000 S1 completed in {el:.2f}s ({5000/el:.1f} S1/sec). Passed pairs: {passed_pairs:,}")
