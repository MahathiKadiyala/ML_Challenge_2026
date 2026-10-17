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
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem, compute_pair_features
)

MAX_BLOCK_SIZE = 50

print("Testing India index build and 5000 S1 throughput...")
t0 = time.time()
s2_df = pl.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), separator="\t").filter(pl.col("country") == "India")
s3_df = pl.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), separator="\t").filter(pl.col("country") == "India")
target_df = pl.concat([s2_df, s3_df])
del s2_df, s3_df
print(f"Loaded {len(target_df):,} targets in {time.time()-t0:.1f}s.")

tids = target_df["entity_id"].to_list()
names = target_df["business_name"].to_list()
addrs = target_df["business_address"].fill_null("").to_list()
del target_df
gc.collect()

t_idx = time.time()
target_dict = {}
idx_sorted_tokens = defaultdict(list)
idx_first2 = defaultdict(list)
idx_num_addr = defaultdict(list)
idx_domain = defaultdict(list)

for tid, name, addr in zip(tids, names, addrs):
    c_tokens = extract_core_tokens(name)
    nums, addr_words = extract_addr_features(addr)
    d_stem = extract_domain_stem(name)
    
    target_dict[tid] = {
        "tid": tid,
        "clean_name": clean_and_translit(name),
        "clean_addr": clean_and_translit(addr),
        "core_str": " ".join(c_tokens),
        "first1": c_tokens[0] if c_tokens else "",
        "nums_set": set(nums),
        "domain_stem": d_stem
    }
    
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
print(f"Index built in {time.time()-t_idx:.1f}s.")

s1_df = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t").filter(pl.col("country") == "India").head(5000)
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
        tg_clean = tg["clean_name"]
        
        has_num = bool(s1_nums and tg["nums_set"] and (s1_nums & tg["nums_set"]))
        nr = fuzz.ratio(s1_clean, tg_clean)
        
        if not has_num and nr < 45:
            ar = fuzz.ratio(s1_addr, tg["clean_addr"])
            if ar < 40 and not (tg["domain_stem"] and tg["domain_stem"] in s1_clean):
                continue
                
        feats = compute_pair_features(s1_info, tg)
        batch_pairs.append((s1_id, tid, feats))

t_feat = time.time() - t_inf
print(f"5000 S1 -> {len(batch_pairs):,} pairs in {t_feat:.1f}s.")
feats_mat = np.array([p[2] for p in batch_pairs], dtype=np.float32)
scores = booster.predict(feats_mat)
accepted = sum(1 for sc in scores if sc >= 0.60)
print(f"Inference complete: {accepted:,} accepted. Total time for 5000 S1: {time.time()-t_inf:.1f}s.")
print(f"Speed: {5000 / (time.time()-t_inf):.1f} S1/sec.")
