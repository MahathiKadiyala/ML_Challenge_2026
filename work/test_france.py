"""
Test inference on France with block size capping and streaming memory management.
"""
import os
import sys
import gc
import re
import unidecode
import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "student_resource", "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "work", "models", "lightgbm_model.txt")

from step12_train_final_model import (
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem, compute_pair_features
)

def test_france():
    print("Loading model...")
    booster = lgb.Booster(model_file=MODEL_PATH)
    
    print("Loading French target records...")
    s2_df = pl.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), separator="\t").filter(pl.col("country") == "France")
    s3_df = pl.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), separator="\t").filter(pl.col("country") == "France")
    target_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    print(f"Loaded {len(target_df):,} French target records.")
    
    # Process targets
    target_dict = {}
    idx_sorted_tokens = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_num_addr = defaultdict(list)
    idx_domain = defaultdict(list)
    
    MAX_BLOCK_SIZE = 60
    
    for row in target_df.iter_rows():
        tid, name, addr = row[0], row[1], row[2] or ""
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
            
    del target_df
    gc.collect()
    print("French targets indexed.")
    
    # Load S1
    s1_df = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t").filter(pl.col("country") == "France")
    total_s1 = len(s1_df)
    print(f"Loaded {total_s1:,} French S1 records.")
    
    # Test streaming batch of 10,000 S1 records
    batch_size = 10_000
    n_batches = (total_s1 + batch_size - 1) // batch_size
    
    high_score_pairs = []
    total_cands_generated = 0
    
    for b_idx in range(min(3, n_batches)): # Test first 3 batches
        start_idx = b_idx * batch_size
        end_idx = min(start_idx + batch_size, total_s1)
        sub_df = s1_df.slice(start_idx, end_idx - start_idx)
        
        batch_pairs = []
        for row in sub_df.iter_rows():
            s1_id, name, addr = row[0], row[1], row[2]
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
                        
            total_cands_generated += len(cands)
            for tid in cands:
                tg = target_dict[tid]
                feats = compute_pair_features(s1_info, tg)
                batch_pairs.append((s1_id, tid, feats))
                
        if batch_pairs:
            feats_mat = np.array([p[2] for p in batch_pairs], dtype=np.float32)
            scores = booster.predict(feats_mat)
            for i, sc in enumerate(scores):
                if sc >= 0.60:
                    high_score_pairs.append((batch_pairs[i][0], batch_pairs[i][1], float(sc)))
                    
        print(f"Batch {b_idx+1}: {end_idx}/{total_s1} processed. Candidates in batch: {len(batch_pairs):,}. High-score pairs: {len(high_score_pairs):,}")

    print("Success! Test passed cleanly without memory issues.")

if __name__ == "__main__":
    test_france()
