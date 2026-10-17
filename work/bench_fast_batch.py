"""
Benchmark 2-stage feature evaluation on 30,000 S1 US entities.
"""
import os
import sys
import gc
import time
import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "student_resource", "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "work", "models", "lightgbm_model.txt")

sys.path.append(os.path.join(BASE_DIR, "work"))
from step12_train_final_model import (
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem, compute_pair_features
)

MAX_BLOCK_SIZE = 50

def run():
    print("Loading model...")
    booster = lgb.Booster(model_file=MODEL_PATH)
    
    print("Loading 1M US target records...")
    t0 = time.time()
    s2_df = pl.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), separator="\t").filter(pl.col("country") == "US").head(500000)
    s3_df = pl.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), separator="\t").filter(pl.col("country") == "US").head(500000)
    target_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    
    target_dict = {}
    idx_sorted_tokens = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_num_addr = defaultdict(list)
    idx_domain = defaultdict(list)
    
    tids = target_df["entity_id"].to_list()
    names = target_df["business_name"].to_list()
    addrs = target_df["business_address"].fill_null("").to_list()
    del target_df
    
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
            
    print(f"Loaded and indexed 1M targets in {time.time()-t0:.1f}s.")
    
    # Load 30,000 S1 records
    s1_df = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t").filter(pl.col("country") == "US").head(30000)
    s1_tids = s1_df["entity_id"].to_list()
    s1_names = s1_df["business_name"].to_list()
    s1_addrs = s1_df["business_address"].fill_null("").to_list()
    del s1_df
    
    gc.disable()
    t_batch = time.time()
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
            
            # Fast Stage 1 check
            has_num = bool(s1_nums and tg["nums_set"] and (s1_nums & tg["nums_set"]))
            nr = fuzz.ratio(s1_clean, tg_clean)
            
            if not has_num and nr < 45:
                # Only check address ratio if no number and low name ratio
                ar = fuzz.ratio(s1_addr, tg["clean_addr"])
                if ar < 40:
                    continue
                    
            feats = compute_pair_features(s1_info, tg)
            batch_pairs.append((s1_id, tid, feats))
            
    print(f"Candidate generation and feature extraction for 30,000 S1 took: {time.time()-t_batch:.1f}s. Pairs: {len(batch_pairs):,}")
    
    t_pred = time.time()
    feats_mat = np.array([p[2] for p in batch_pairs], dtype=np.float32)
    scores = booster.predict(feats_mat)
    accepted = sum(1 for sc in scores if sc >= 0.60)
    print(f"LightGBM prediction took: {time.time()-t_pred:.1f}s. Accepted pairs: {accepted:,}")
    print(f"Total time for 30,000 S1 batch: {time.time()-t_batch:.1f}s!")

if __name__ == "__main__":
    run()
