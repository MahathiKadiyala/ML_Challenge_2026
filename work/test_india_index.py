import os
import sys
import gc
import re
import time
import unidecode
import polars as pl
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

def clean_and_translit(text):
    if not text:
        return ""
    if not text.isascii():
        text = unidecode.unidecode(text).lower()
        text = re.sub(r"([a-z])\1+", r"\1", text)
    else:
        text = text.lower()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())

def extract_core_tokens(clean_name):
    tokens = clean_name.split()
    if tokens and tokens[-1] in {"com", "org", "net", "in", "co", "us", "fr"}:
        tokens.pop()
    content_tokens = [t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1]
    return content_tokens

def extract_addr_features(clean_addr):
    tokens = clean_addr.split()
    raw_nums = [t.lstrip("0") for t in tokens if t.isdigit() and len(t) <= 6]
    nums = [t for t in raw_nums if t]
    addr_words = [t for t in tokens if not t.isdigit() and t not in GENERIC_ADDR and len(t) > 2]
    return nums, addr_words

def extract_domain_stem(name):
    m = re.search(r"([a-z0-9]+)\.(com|org|net|in|co|us|fr|info|biz)\b", name.lower())
    if m:
        stem = m.group(1)
        return re.sub(r"([a-z])\1+", r"\1", stem)
    if name.startswith("@"):
        stem = clean_and_translit(name[1:])
        return stem
    return None

class TargetRecord:
    __slots__ = ('tid', 'clean_name', 'clean_addr', 'core_str', 'first1', 'nums_set', 'domain_stem')
    def __init__(self, tid, clean_name, clean_addr, core_str, first1, nums_set, domain_stem):
        self.tid = tid
        self.clean_name = clean_name
        self.clean_addr = clean_addr
        self.core_str = core_str
        self.first1 = first1
        self.nums_set = nums_set
        self.domain_stem = domain_stem

print("Testing full India index memory and build time...")
t0 = time.time()
s2_df = pl.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), separator="\t").filter(pl.col("country") == "India")
s3_df = pl.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), separator="\t").filter(pl.col("country") == "India")
target_df = pl.concat([s2_df, s3_df])
del s2_df, s3_df
total_targets = len(target_df)
print(f"Loaded {total_targets:,} target records in {time.time()-t0:.1f}s.")

tids = target_df["entity_id"].to_list()
names = target_df["business_name"].to_list()
addrs = target_df["business_address"].fill_null("").to_list()
del target_df
gc.collect()

MAX_BLOCK_SIZE = 50
target_dict = {}
idx_sorted_tokens = defaultdict(list)
idx_first2 = defaultdict(list)
idx_num_addr = defaultdict(list)
idx_domain = defaultdict(list)

bl_sorted = set()
bl_first2 = set()
bl_num_addr = set()
bl_domain = set()

def add_to_index(idx, blacklist, key, tid, max_size=50):
    if key in blacklist:
        return
    lst = idx[key]
    if len(lst) >= max_size:
        del idx[key]
        blacklist.add(key)
        return
    lst.append(tid)

t_idx = time.time()
for tid, name, addr in zip(tids, names, addrs):
    c_name = clean_and_translit(name)
    c_addr = clean_and_translit(addr)
    c_tokens = extract_core_tokens(c_name)
    nums, addr_words = extract_addr_features(c_addr)
    d_stem = extract_domain_stem(name)
    
    target_dict[tid] = TargetRecord(
        tid, c_name, c_addr, " ".join(c_tokens),
        c_tokens[0] if c_tokens else "",
        set(nums), d_stem
    )
    
    st = " ".join(sorted(c_tokens))
    if st:
        add_to_index(idx_sorted_tokens, bl_sorted, st, tid, MAX_BLOCK_SIZE)
        
    f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
    if f2:
        add_to_index(idx_first2, bl_first2, f2, tid, MAX_BLOCK_SIZE)
        
    if nums and addr_words:
        for num in nums[:2]:
            for word in addr_words[:2]:
                add_to_index(idx_num_addr, bl_num_addr, (num, word), tid, MAX_BLOCK_SIZE)
                
    if d_stem:
        add_to_index(idx_domain, bl_domain, d_stem, tid, MAX_BLOCK_SIZE)

del tids, names, addrs, bl_sorted, bl_first2, bl_num_addr, bl_domain
gc.collect()

print(f"Full India index built in {time.time()-t_idx:.1f}s.")
print(f"target_dict count: {len(target_dict):,}")
print(f"idx_sorted_tokens: {len(idx_sorted_tokens):,}")
print(f"idx_first2: {len(idx_first2):,}")
print(f"idx_num_addr: {len(idx_num_addr):,}")
