"""
Production inference pipeline for Business Entity Resolution.
Processes test datasets with country-partitioned high-selectivity blocking,
short-circuiting candidate pre-filters, LightGBM scoring, and 1-to-1 bipartite assignment.
Generates compliant matching_results.tsv and candidate_pairs.tsv.
"""
import os
import sys
import gc
import re
import time
import argparse
import polars as pl
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from rapidfuzz import fuzz

from features import (
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem, compute_pair_features
)

MAX_BLOCK_SIZE = 50
BATCH_SIZE = 30000
SCORE_THRESHOLD = 0.60

def add_to_index(idx, blacklist, key, tid, max_size=50):
    if key in blacklist:
        return
    lst = idx[key]
    if len(lst) >= max_size:
        del idx[key]
        blacklist.add(key)
        return
    lst.append(tid)

def process_country(country_name, test_dir, booster, work_dir):
    print(f"\n========================================================", flush=True)
    print(f"PROCESSING COUNTRY: {country_name}", flush=True)
    print(f"========================================================", flush=True)
    t_start = time.time()
    
    # 1. Load targets
    print(f"Loading S2 and S3 records for {country_name}...", flush=True)
    t0 = time.time()
    s2_df = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t").filter(pl.col("country") == country_name)
    s3_df = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t").filter(pl.col("country") == country_name)
    target_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    total_targets = len(target_df)
    print(f"Loaded {total_targets:,} target records in {time.time()-t0:.1f}s.", flush=True)
    
    tids = target_df["entity_id"].to_list()
    names = target_df["business_name"].to_list()
    addrs = target_df["business_address"].fill_null("").to_list()
    del target_df
    gc.collect()
    
    # 2. Build high-selectivity pruned indexes
    print("Building high-selectivity pruned indexes...", flush=True)
    t_idx = time.time()
    target_dict = {}
    
    idx_sorted_tokens = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_num_addr = defaultdict(list)
    idx_domain = defaultdict(list)
    
    bl_sorted = set()
    bl_first2 = set()
    bl_num_addr = set()
    bl_domain = set()
    
    for tid, name, addr in zip(tids, names, addrs):
        c_name = clean_and_translit(name)
        c_addr = clean_and_translit(addr)
        c_tokens = extract_core_tokens(c_name)
        nums, addr_words = extract_addr_features(c_addr)
        d_stem = extract_domain_stem(name)
        
        target_dict[tid] = {
            "tid": tid,
            "clean_name": c_name,
            "clean_addr": c_addr,
            "core_str": " ".join(c_tokens),
            "first1": c_tokens[0] if c_tokens else "",
            "nums_set": set(nums),
            "domain_stem": d_stem
        }
        
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
    print(f"Indexes built in {time.time()-t_idx:.1f}s.", flush=True)
    
    # 3. Load S1 for this country
    print(f"Loading S1 records for {country_name}...", flush=True)
    s1_df = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t").filter(pl.col("country") == country_name)
    s1_tids = s1_df["entity_id"].to_list()
    s1_names = s1_df["business_name"].to_list()
    s1_addrs = s1_df["business_address"].fill_null("").to_list()
    total_s1 = len(s1_tids)
    del s1_df
    print(f"Loaded {total_s1:,} S1 entities for {country_name}.", flush=True)
    
    # Candidate streaming file
    os.makedirs(work_dir, exist_ok=True)
    cands_path = os.path.join(work_dir, f"cands_{country_name}.tsv")
    cands_f = open(cands_path, "w", encoding="utf-8")
    
    high_score_pairs = []
    n_batches = (total_s1 + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Running inference in {n_batches} batches of {BATCH_SIZE}...", flush=True)
    
    gc.disable()
    t_inf = time.time()
    
    for b_idx in range(n_batches):
        t_b0 = time.time()
        start_idx = b_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_s1)
        
        batch_s1_ids = s1_tids[start_idx:end_idx]
        batch_s1_names = s1_names[start_idx:end_idx]
        batch_s1_addrs = s1_addrs[start_idx:end_idx]
        
        batch_pairs = []
        
        for s1_id, name, addr in zip(batch_s1_ids, batch_s1_names, batch_s1_addrs):
            c_name = clean_and_translit(name)
            c_addr = clean_and_translit(addr)
            c_tokens = extract_core_tokens(c_name)
            nums, addr_words = extract_addr_features(c_addr)
            
            s1_f1 = c_tokens[0] if c_tokens else ""
            s1_nums = set(nums)
            
            s1_info = {
                "clean_name": c_name,
                "clean_addr": c_addr,
                "core_str": " ".join(c_tokens),
                "first1": s1_f1,
                "nums_set": s1_nums
            }
            
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
            if c_tokens:
                concat = "".join(c_tokens)
                if concat in idx_domain:
                    cands.update(idx_domain[concat])
                for tok in c_tokens[:2]:
                    if len(tok) >= 4 and tok in idx_domain:
                        cands.update(idx_domain[tok])
                        
            # Stream candidates to disk
            if cands:
                cands_f.write(f"{s1_id}\t{','.join(sorted(cands))}\n")
            else:
                cands_f.write(f"{s1_id}\t\n")
                
            # Short-circuiting pre-filter
            for tid in cands:
                tg = target_dict[tid]
                if (s1_f1 and tg["first1"] and s1_f1 == tg["first1"]) or \
                   (s1_nums and tg["nums_set"] and (s1_nums & tg["nums_set"])) or \
                   (tg["domain_stem"] and tg["domain_stem"] in c_name) or \
                   (fuzz.ratio(c_name, tg["clean_name"]) >= 45):
                    feats = compute_pair_features(s1_info, tg)
                    batch_pairs.append((s1_id, tid, feats))
                    
        # Predict on batch pairs
        if batch_pairs:
            feats_mat = np.array([p[2] for p in batch_pairs], dtype=np.float32)
            scores = booster.predict(feats_mat)
            for i, sc in enumerate(scores):
                if sc >= SCORE_THRESHOLD:
                    high_score_pairs.append((batch_pairs[i][0], batch_pairs[i][1], float(sc)))
                    
        elapsed_batch = time.time() - t_b0
        total_elapsed = time.time() - t_inf
        print(f"  Batch {b_idx+1}/{n_batches} done: {end_idx:,}/{total_s1:,} S1 processed. Pairs: {len(batch_pairs):,}. Accepted so far: {len(high_score_pairs):,}. [Batch: {elapsed_batch:.1f}s | Elapsed: {total_elapsed:.1f}s]", flush=True)
        
    cands_f.close()
    gc.enable()
    del s1_tids, s1_names, s1_addrs, target_dict, idx_sorted_tokens, idx_first2, idx_num_addr, idx_domain
    gc.collect()
    
    # 4. 1-to-1 Target Assignment
    print(f"\nApplying 1-to-1 Target Assignment for {country_name}...", flush=True)
    high_score_pairs.sort(key=lambda x: x[2], reverse=True)
    assigned_targets = set()
    
    matches_path = os.path.join(work_dir, f"matches_{country_name}.tsv")
    matches_dict = defaultdict(list)
    
    for s1_id, tid, sc in high_score_pairs:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            matches_dict[s1_id].append(tid)
            
    with open(matches_path, "w", encoding="utf-8") as mf:
        for s1_id, mids in matches_dict.items():
            mf.write(f"{s1_id}\t{','.join(sorted(mids))}\n")
            
    print(f"Results for {country_name}:", flush=True)
    print(f"  Total S1 entities: {total_s1:,}", flush=True)
    print(f"  S1 entities with matches: {len(matches_dict):,} ({len(matches_dict)/total_s1*100:.2f}%)", flush=True)
    print(f"  Singletons: {total_s1 - len(matches_dict):,} ({(total_s1 - len(matches_dict))/total_s1*100:.2f}%)", flush=True)
    print(f"  Total matched targets: {len(assigned_targets):,}", flush=True)
    print(f"  Country completed in {time.time()-t_start:.1f}s.", flush=True)
    
    return len(matches_dict), len(assigned_targets)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="student_resource/dataset/test")
    parser.add_argument("--model-path", type=str, default="models/lightgbm_model.txt")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--work-dir", type=str, default="work")
    args = parser.parse_args()
    
    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading LightGBM model from {args.model_path}...", flush=True)
    booster = lgb.Booster(model_file=args.model_path)
    
    for country in ["France", "US", "India"]:
        process_country(country, args.test_dir, booster, args.work_dir)
        gc.collect()
        
    print("\n========================================================", flush=True)
    print("ASSEMBLING FINAL SUBMISSION FILES IN EXACT TEST ORDER", flush=True)
    print("========================================================", flush=True)
    
    all_matches = {}
    for country in ["France", "US", "India"]:
        m_file = os.path.join(args.work_dir, f"matches_{country}.tsv")
        if os.path.isfile(m_file):
            with open(m_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split("\t")
                        if len(parts) == 2 and parts[1]:
                            all_matches[parts[0]] = parts[1].split(",")
                            
    all_candidates = {}
    for country in ["France", "US", "India"]:
        c_file = os.path.join(args.work_dir, f"cands_{country}.tsv")
        if os.path.isfile(c_file):
            with open(c_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split("\t")
                        if len(parts) == 2 and parts[1]:
                            all_candidates[parts[0]] = parts[1].split(",")
                            
    s1_all = pl.read_csv(os.path.join(args.test_dir, "test_source1.tsv"), separator="\t")
    s1_order = s1_all["entity_id"].to_list()
    del s1_all
    total_test_s1 = len(s1_order)
    
    out_dirs = [args.output_dir]
    alt_dir = "student_resource/output" if args.output_dir == "results" else "results"
    out_dirs.append(alt_dir)
    for od in out_dirs:
        os.makedirs(od, exist_ok=True)
    
    total_matches_cnt = 0
    s1_with_matches = 0
    s2_cnt = 0
    s3_cnt = 0
    
    for od in out_dirs:
        matching_out = os.path.join(od, "matching_results.tsv")
        with open(matching_out, "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in s1_order:
                m_list = all_matches.get(s1_id, [])
                if m_list:
                    m_list_sorted = sorted(m_list)
                    f.write(f"{s1_id}\t{','.join(m_list_sorted)}\n")
                    if od == out_dirs[0]:
                        s1_with_matches += 1
                        total_matches_cnt += len(m_list_sorted)
                        for m in m_list_sorted:
                            if m.startswith("S2-"):
                                s2_cnt += 1
                            elif m.startswith("S3-"):
                                s3_cnt += 1
                else:
                    f.write(f"{s1_id}\t\n")
                    
        candidate_out = os.path.join(od, "candidate_pairs.tsv")
        with open(candidate_out, "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in s1_order:
                c_set = set(all_candidates.get(s1_id, []))
                m_list = all_matches.get(s1_id, [])
                if m_list:
                    c_set = c_set | set(m_list)
                if c_set:
                    c_list_sorted = sorted(c_set)
                    f.write(f"{s1_id}\t{','.join(c_list_sorted)}\n")
                else:
                    f.write(f"{s1_id}\t\n")
                
    print(f"Results successfully written to both {out_dirs[0]} and {out_dirs[1]}.", flush=True)
    print(f"\nFinal Test Matching Statistics:", flush=True)
    print(f"  Total S1 entities: {total_test_s1:,}", flush=True)
    print(f"  S1 entities with matches: {s1_with_matches:,} ({s1_with_matches / total_test_s1 * 100:.2f}%)", flush=True)
    print(f"  Singletons (0 matches): {total_test_s1 - s1_with_matches:,} ({(total_test_s1 - s1_with_matches) / total_test_s1 * 100:.2f}%)", flush=True)
    print(f"  Total matched targets: {total_matches_cnt:,}", flush=True)
    print(f"  S2 matches: {s2_cnt:,} ({s2_cnt / total_matches_cnt * 100:.2f}%)", flush=True)
    print(f"  S3 matches: {s3_cnt:,} ({s3_cnt / total_matches_cnt * 100:.2f}%)", flush=True)
    print(f"\nTotal elapsed time: {time.time()-t_start:.1f}s", flush=True)

if __name__ == "__main__":
    main()
