import os
import sys
import gc
import re
import time
import unidecode
import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "student_resource", "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "work", "models", "lightgbm_model.txt")
WORK_DIR = os.path.join(BASE_DIR, "work")

sys.path.append(WORK_DIR)
from step12_train_final_model import (
    LEGAL_STOPWORDS, GENERIC_ADDR,
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem
)

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

def compute_pair_features_fast(s1, tg, nr, ar=None):
    s1_name_clean = s1["clean_name"]
    tg_name_clean = tg.clean_name
    s1_addr_clean = s1["clean_addr"]
    tg_addr_clean = tg.clean_addr
    
    # 1. Name similarities
    name_ratio = nr
    name_token_sort = fuzz.token_sort_ratio(s1_name_clean, tg_name_clean)
    name_token_set = fuzz.token_set_ratio(s1_name_clean, tg_name_clean)
    
    # 2. Core name similarities
    s1_core = s1["core_str"]
    tg_core = tg.core_str
    core_ratio = fuzz.ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    core_token_sort = fuzz.token_sort_ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    core_token_set = fuzz.token_set_ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    
    # First token match
    s1_f1 = s1["first1"]
    tg_f1 = tg.first1
    first_token_match = 1.0 if s1_f1 and tg_f1 and s1_f1 == tg_f1 else 0.0
    
    # 3. Address similarities
    addr_ratio = ar if ar is not None else (fuzz.ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0)
    addr_token_sort = fuzz.token_sort_ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    addr_token_set = fuzz.token_set_ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    
    # 4. Number features
    s1_nums = s1["nums_set"]
    tg_nums = tg.nums_set
    if s1_nums and tg_nums:
        has_nums = 1.0
        num_intersection = len(s1_nums & tg_nums)
        num_jaccard = num_intersection / len(s1_nums | tg_nums)
        num_exact = 1.0 if num_intersection > 0 else 0.0
    else:
        has_nums = 0.0
        num_jaccard = 0.0
        num_exact = 0.0
        
    # 5. Domain / Handle match
    d_stem = tg.domain_stem
    if d_stem:
        has_domain = 1.0
        domain_match = 1.0 if d_stem in s1_name_clean or (s1_core and d_stem in s1_core) else 0.0
    else:
        has_domain = 0.0
        domain_match = 0.0
        
    # 6. Length differences
    len_diff_name = abs(len(s1_name_clean) - len(tg_name_clean))
    len_diff_addr = abs(len(s1_addr_clean) - len(tg_addr_clean))
    
    # 7. Source prefix (S2 vs S3)
    is_s2 = 1.0 if tg.tid.startswith("S2-") else 0.0
    
    return [
        name_ratio, name_token_sort, name_token_set,
        core_ratio, core_token_sort, core_token_set, first_token_match,
        addr_ratio, addr_token_sort, addr_token_set,
        has_nums, num_jaccard, num_exact,
        has_domain, domain_match,
        len_diff_name, len_diff_addr,
        is_s2
    ]

MAX_BLOCK_SIZE = 50

print("Testing TargetRecord India Indexing on 1 million targets...")
t0 = time.time()
s2_df = pl.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), separator="\t").filter(pl.col("country") == "India").head(500000)
s3_df = pl.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), separator="\t").filter(pl.col("country") == "India").head(500000)
target_df = pl.concat([s2_df, s3_df])
del s2_df, s3_df

tids = target_df["entity_id"].to_list()
names = target_df["business_name"].to_list()
addrs = target_df["business_address"].fill_null("").to_list()
del target_df
gc.collect()

target_dict = {}
idx_sorted_tokens = defaultdict(list)
idx_first2 = defaultdict(list)
idx_num_addr = defaultdict(list)
idx_domain = defaultdict(list)

for tid, name, addr in zip(tids, names, addrs):
    c_tokens = extract_core_tokens(name)
    nums, addr_words = extract_addr_features(addr)
    d_stem = extract_domain_stem(name)
    
    target_dict[tid] = TargetRecord(
        tid, clean_and_translit(name), clean_and_translit(addr),
        " ".join(c_tokens), c_tokens[0] if c_tokens else "",
        set(nums), d_stem
    )
    
    st = " ".join(sorted(c_tokens))
    if st and len(idx_sorted_tokens[st]) < MAX_BLOCK_SIZE:
        idx_sorted_tokens[st].append(tid)
        
    f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
    if f2 and len(idx_first2[f2]) < MAX_BLOCK_SIZE:
        idx_first2[f2].append(tid)
        
    if nums and addr_words:
        for num in nums[:2]:
            for word in addr_words[:2]:
                k = (num, word)
                if len(idx_num_addr[k]) < MAX_BLOCK_SIZE:
                    idx_num_addr[k].append(tid)
                    
    if d_stem and len(idx_domain[d_stem]) < MAX_BLOCK_SIZE:
        idx_domain[d_stem].append(tid)

del tids, names, addrs
gc.collect()
print(f"1M targets indexed in {time.time()-t0:.1f}s.")

# Load 10,000 S1
s1_df = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t").filter(pl.col("country") == "India").head(10000)
s1_tids = s1_df["entity_id"].to_list()
s1_names = s1_df["business_name"].to_list()
s1_addrs = s1_df["business_address"].fill_null("").to_list()
del s1_df
gc.collect()

booster = lgb.Booster(model_file=MODEL_PATH)
t_inf = time.time()
batch_pairs = []

for s1_id, name, addr in zip(s1_tids, s1_names, s1_addrs):
    c_tokens = extract_core_tokens(name)
    nums, addr_words = extract_addr_features(addr)
    
    s1_info = {
        "clean_name": clean_and_translit(name),
        "clean_addr": clean_and_translit(addr),
        "core_str": " ".join(c_tokens),
        "first1": c_tokens[0] if c_tokens else "",
        "nums_set": set(nums)
    }
    
    cands = set()
    st = " ".join(sorted(c_tokens))
    if st and st in idx_sorted_tokens:
        cands.update(idx_sorted_tokens[st])
    f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
    if f2 and f2 in idx_first2:
        cands.update(idx_first2[f2])
    if nums and addr_words:
        for num in nums[:2]:
            for word in addr_words[:2]:
                k = (num, word)
                if k in idx_num_addr:
                    cands.update(idx_num_addr[k])
    if c_tokens:
        concat = "".join(c_tokens)
        if concat in idx_domain:
            cands.update(idx_domain[concat])
        for tok in c_tokens[:2]:
            if len(tok) >= 4 and tok in idx_domain:
                cands.update(idx_domain[tok])
                
    s1_clean = s1_info["clean_name"]
    s1_nums = s1_info["nums_set"]
    s1_addr = s1_info["clean_addr"]
    
    for tid in cands:
        tg = target_dict[tid]
        tg_clean = tg.clean_name
        
        has_num = bool(s1_nums and tg.nums_set and (s1_nums & tg.nums_set))
        nr = fuzz.ratio(s1_clean, tg_clean)
        ar = None
        
        if not has_num and nr < 45:
            ar = fuzz.ratio(s1_addr, tg.clean_addr)
            if ar < 40 and not (tg.domain_stem and tg.domain_stem in s1_clean):
                continue
                
        feats = compute_pair_features_fast(s1_info, tg, nr, ar)
        batch_pairs.append((s1_id, tid, feats))

t_feat = time.time() - t_inf
feats_mat = np.array([p[2] for p in batch_pairs], dtype=np.float32)
scores = booster.predict(feats_mat)
total_time = time.time() - t_inf
print(f"10,000 S1 processed in {total_time:.2f}s ({len(s1_tids)/total_time:.0f} S1/sec, {len(batch_pairs):,} pairs).")
