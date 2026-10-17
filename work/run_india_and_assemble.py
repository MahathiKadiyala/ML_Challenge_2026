"""
High-Performance Resumable Production Pipeline for Business Entity Resolution.
Features:
1. Resumable execution for India (Source 2 and Source 3) from disk checkpoints.
2. Fast 2-stage candidate pre-filtering (preserving 99.65% true matches while eliminating >80% comparisons).
3. Safe fuzzy wrappers preventing C++ MemoryError/bad allocation.
4. Micro-batches (BATCH_SIZE = 5000) with explicit gc.collect() preventing memory fragmentation.
5. Bipartite 1-to-1 greedy target assignment per country.
6. Global assembly of matching_results.tsv and candidate_pairs.tsv across France, US, and India.
7. Execution of official submission validator.
8. Creation of final submission zip archive.
"""
import os
import sys
import gc
import re
import time
import zipfile
import subprocess
import unidecode
import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "student_resource", "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "student_resource", "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "work", "models", "lightgbm_model.txt")
WORK_DIR = os.path.join(BASE_DIR, "work")

sys.path.append(WORK_DIR)
from step12_train_final_model import (
    LEGAL_STOPWORDS, GENERIC_ADDR,
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem
)

MAX_BLOCK_SIZE = 50
BATCH_SIZE = 5000
SCORE_THRESHOLD = 0.60

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

def safe_ratio(s1, s2):
    try:
        return fuzz.ratio(s1, s2) if s1 and s2 else 0.0
    except Exception:
        return 0.0

def safe_token_sort(s1, s2):
    try:
        return fuzz.token_sort_ratio(s1, s2) if s1 and s2 else 0.0
    except Exception:
        return 0.0

def safe_token_set(s1, s2):
    try:
        return fuzz.token_set_ratio(s1, s2) if s1 and s2 else 0.0
    except Exception:
        return 0.0

def compute_pair_features_fast(s1, tg, nr=None):
    s1_name_clean = s1["clean_name"]
    tg_name_clean = tg.clean_name
    s1_addr_clean = s1["clean_addr"]
    tg_addr_clean = tg.clean_addr
    
    # 1. Name similarities
    name_ratio = nr if nr is not None else safe_ratio(s1_name_clean, tg_name_clean)
    name_token_sort = safe_token_sort(s1_name_clean, tg_name_clean)
    name_token_set = safe_token_set(s1_name_clean, tg_name_clean)
    
    # 2. Core name similarities
    s1_core = s1["core_str"]
    tg_core = tg.core_str
    core_ratio = safe_ratio(s1_core, tg_core)
    core_token_sort = safe_token_sort(s1_core, tg_core)
    core_token_set = safe_token_set(s1_core, tg_core)
    
    # First token match
    s1_f1 = s1["first1"]
    tg_f1 = tg.first1
    first_token_match = 1.0 if s1_f1 and tg_f1 and s1_f1 == tg_f1 else 0.0
    
    # 3. Address similarities
    addr_ratio = safe_ratio(s1_addr_clean, tg_addr_clean)
    addr_token_sort = safe_token_sort(s1_addr_clean, tg_addr_clean)
    addr_token_set = safe_token_set(s1_addr_clean, tg_addr_clean)
    
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

def build_target_index(source_file, source_name):
    print(f"\nLoading {source_name} records for India...", flush=True)
    t0 = time.time()
    df = pl.read_csv(os.path.join(TEST_DIR, source_file), separator="\t").filter(pl.col("country") == "India")
    total = len(df)
    print(f"Loaded {total:,} {source_name} records in {time.time()-t0:.1f}s.", flush=True)
    
    tids = df["entity_id"].to_list()
    names = df["business_name"].to_list()
    addrs = df["business_address"].fill_null("").to_list()
    del df
    gc.collect()
    
    print(f"Building {source_name} index...", flush=True)
    t_idx = time.time()
    target_dict = {}
    idx_sorted_tokens = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_num_addr = defaultdict(list)
    idx_domain = defaultdict(list)
    
    for tid, name, addr in zip(tids, names, addrs):
        c_name = clean_and_translit(name)
        c_addr = clean_and_translit(addr)
        c_tokens = extract_core_tokens(c_name)
        nums, addr_words = extract_addr_features(c_addr)
        d_stem = extract_domain_stem(name)
        
        target_dict[tid] = TargetRecord(
            tid, c_name, c_addr,
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
    print(f"{source_name} index built in {time.time()-t_idx:.1f}s. Target dict: {len(target_dict):,}", flush=True)
    
    return target_dict, idx_sorted_tokens, idx_first2, idx_num_addr, idx_domain

def run_india_source(source_name, source_file, s1_tids, s1_names, s1_addrs, booster):
    cands_path = os.path.join(WORK_DIR, f"cands_India_{source_name}.tsv")
    scores_path = os.path.join(WORK_DIR, f"high_scores_India_{source_name}.tsv")
    total_s1 = len(s1_tids)
    
    already_done = 0
    if os.path.exists(cands_path):
        with open(cands_path, "rb") as fp:
            already_done = sum(1 for _ in fp)
            
    if already_done >= total_s1:
        print(f"--> [{source_name}] Already 100% completed ({already_done:,} S1). Skipping inference.", flush=True)
        return
        
    mode = "a" if already_done > 0 else "w"
    if already_done > 0:
        print(f"--> [{source_name}] RESUMING from entity {already_done:,} / {total_s1:,} ({already_done/total_s1*100:.1f}% previously done).", flush=True)
        
    target_dict, idx_sorted_tokens, idx_first2, idx_num_addr, idx_domain = build_target_index(source_file, source_name)
    
    remaining_s1 = total_s1 - already_done
    n_batches = (remaining_s1 + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nRunning {source_name} candidate matching for remaining {remaining_s1:,} S1 entities in {n_batches} batches of {BATCH_SIZE}...", flush=True)
    
    cands_f = open(cands_path, mode, encoding="utf-8")
    scores_f = open(scores_path, mode, encoding="utf-8")
    
    total_accepted = 0
    t_start = time.time()
    
    for b_idx in range(n_batches):
        t_b0 = time.time()
        start_idx = already_done + b_idx * BATCH_SIZE
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
            if st and st in idx_sorted_tokens:
                cands.update(idx_sorted_tokens[st])
            f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else s1_f1
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
                        
            # Stream candidates
            if cands:
                cands_f.write(f"{s1_id}\t{','.join(sorted(cands))}\n")
            else:
                cands_f.write(f"{s1_id}\t\n")
                
            # Fast 2-Stage Filter
            for tid in cands:
                tg = target_dict[tid]
                
                has_num = bool(s1_nums and tg.nums_set and (s1_nums & tg.nums_set))
                first_word_match = bool(s1_f1 and tg.first1 and s1_f1 == tg.first1)
                domain_match = bool(tg.domain_stem and tg.domain_stem in c_name)
                
                nr = None
                if not (has_num or first_word_match or domain_match):
                    nr = safe_ratio(c_name, tg.clean_name)
                    if nr < 45:
                        continue
                        
                feats = compute_pair_features_fast(s1_info, tg, nr)
                batch_pairs.append((s1_id, tid, feats))
                
        # Predict on batch pairs
        batch_accepted = 0
        if batch_pairs:
            feats_mat = np.array([p[2] for p in batch_pairs], dtype=np.float32)
            scores = booster.predict(feats_mat)
            for i, sc in enumerate(scores):
                if sc >= SCORE_THRESHOLD:
                    scores_f.write(f"{batch_pairs[i][0]}\t{batch_pairs[i][1]}\t{float(sc):.5f}\n")
                    batch_accepted += 1
            scores_f.flush()
            del feats_mat, scores
            
        cands_f.flush()
        total_accepted += batch_accepted
        elapsed_b = time.time() - t_b0
        elapsed_tot = time.time() - t_start
        speed = (end_idx - start_idx) / elapsed_b if elapsed_b > 0 else 0
        rem_sec = (total_s1 - end_idx) / speed if speed > 0 else 0
        print(f"  [{source_name}] Batch {b_idx+1}/{n_batches}: {end_idx:,}/{total_s1:,} ({end_idx/total_s1*100:.1f}%). Pairs: {len(batch_pairs):,}, +{batch_accepted:,} acc. [{elapsed_b:.1f}s | {speed:.0f} S1/s | ETA: {rem_sec/60:.1f}m]", flush=True)
        
        del batch_pairs
        if (b_idx + 1) % 5 == 0:
            gc.collect()
        
    cands_f.close()
    scores_f.close()
    del target_dict, idx_sorted_tokens, idx_first2, idx_num_addr, idx_domain
    gc.collect()
    print(f"[{source_name}] Completed in {time.time()-t_start:.1f}s. Total accepted: {total_accepted:,}.", flush=True)

def process_india(booster):
    matches_india_path = os.path.join(WORK_DIR, "matches_India.tsv")
    cands_india_path = os.path.join(WORK_DIR, "cands_India.tsv")
    
    if os.path.exists(matches_india_path) and os.path.exists(cands_india_path):
        with open(cands_india_path, "rb") as fp:
            c_lines = sum(1 for _ in fp)
        if c_lines == 809986:
            print(f"India already completed ({c_lines:,} S1). Skipping India inference.", flush=True)
            return
            
    print("\n========================================================")
    print("STARTING ROBUST HIGH-SPEED INDIA INFERENCE (S2 & S3)")
    print("========================================================")
    
    print("Loading India S1 entities...", flush=True)
    t0 = time.time()
    s1_df = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t").filter(pl.col("country") == "India")
    s1_tids = s1_df["entity_id"].to_list()
    s1_names = s1_df["business_name"].to_list()
    s1_addrs = s1_df["business_address"].fill_null("").to_list()
    total_s1 = len(s1_tids)
    del s1_df
    gc.collect()
    print(f"Loaded {total_s1:,} India S1 records in {time.time()-t0:.1f}s.", flush=True)
    
    # 1. Process Source 2 (with resumption)
    run_india_source("s2", "test_source2.tsv", s1_tids, s1_names, s1_addrs, booster)
    
    # 2. Process Source 3 (with resumption)
    run_india_source("s3", "test_source3.tsv", s1_tids, s1_names, s1_addrs, booster)
    
    # 3. Merge candidates
    print("\nMerging S2 and S3 candidates into cands_India.tsv...", flush=True)
    t_m = time.time()
    s2_cands_path = os.path.join(WORK_DIR, "cands_India_s2.tsv")
    s3_cands_path = os.path.join(WORK_DIR, "cands_India_s3.tsv")
    
    with open(s2_cands_path, "r", encoding="utf-8") as f2, open(s3_cands_path, "r", encoding="utf-8") as f3, open(cands_india_path, "w", encoding="utf-8") as fout:
        for l2, l3 in zip(f2, f3):
            p2 = l2.strip().split("\t")
            p3 = l3.strip().split("\t")
            s1_id = p2[0]
            cands2 = p2[1].split(",") if len(p2) > 1 and p2[1] else []
            cands3 = p3[1].split(",") if len(p3) > 1 and p3[1] else []
            all_cands = sorted(set(cands2) | set(cands3))
            if all_cands:
                fout.write(f"{s1_id}\t{','.join(all_cands)}\n")
            else:
                fout.write(f"{s1_id}\t\n")
    print(f"Merged candidates written in {time.time()-t_m:.1f}s.", flush=True)
    
    # 4. 1-to-1 Bipartite Target Assignment for India
    print("\nApplying 1-to-1 Target Assignment for India...", flush=True)
    t_assign = time.time()
    high_score_pairs = []
    
    for s_name in ["s2", "s3"]:
        sp = os.path.join(WORK_DIR, f"high_scores_India_{s_name}.tsv")
        with open(sp, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) == 3:
                        high_score_pairs.append((parts[0], parts[1], float(parts[2])))
                        
    print(f"Loaded {len(high_score_pairs):,} total high score pairs. Sorting by score descending...", flush=True)
    high_score_pairs.sort(key=lambda x: x[2], reverse=True)
    
    assigned_targets = set()
    matches_dict = defaultdict(list)
    for s1_id, tid, sc in high_score_pairs:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            matches_dict[s1_id].append(tid)
            
    with open(matches_india_path, "w", encoding="utf-8") as mf:
        for s1_id, mids in matches_dict.items():
            mf.write(f"{s1_id}\t{','.join(sorted(mids))}\n")
            
    print(f"India 1-to-1 Assignment Results in {time.time()-t_assign:.1f}s:")
    print(f"  Total S1 entities: {total_s1:,}")
    print(f"  S1 entities with matches: {len(matches_dict):,} ({len(matches_dict)/total_s1*100:.2f}%)")
    print(f"  Singletons: {total_s1 - len(matches_dict):,} ({(total_s1 - len(matches_dict))/total_s1*100:.2f}%)")
    print(f"  Total matched targets: {len(assigned_targets):,}")

def assemble_submissions():
    print("\n========================================================")
    print("ASSEMBLING FINAL SUBMISSIONS (France, US, India)")
    print("========================================================")
    t0 = time.time()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_matches = {}
    for country in ["France", "US", "India"]:
        m_file = os.path.join(WORK_DIR, f"matches_{country}.tsv")
        print(f"Loading matches for {country} from {m_file}...", flush=True)
        with open(m_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) == 2 and parts[1]:
                        all_matches[parts[0]] = parts[1].split(",")
    print(f"Total S1 entities with matches across all countries: {len(all_matches):,}", flush=True)
    
    all_candidates = {}
    for country in ["France", "US", "India"]:
        c_file = os.path.join(WORK_DIR, f"cands_{country}.tsv")
        print(f"Loading candidates for {country} from {c_file}...", flush=True)
        with open(c_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) == 2 and parts[1]:
                        all_candidates[parts[0]] = parts[1].split(",")
    print(f"Total S1 entities with candidates across all countries: {len(all_candidates):,}", flush=True)
    
    s1_all = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t")
    s1_order = s1_all["entity_id"].to_list()
    total_test_s1 = len(s1_order)
    del s1_all
    print(f"Total test S1 entities in original order: {total_test_s1:,}", flush=True)
    
    # 1. Write matching_results.tsv
    matching_paths = [
        os.path.join(RESULTS_DIR, "matching_results.tsv"),
        os.path.join(OUTPUT_DIR, "matching_results.tsv")
    ]
    
    s1_with_matches = 0
    total_matches_cnt = 0
    s2_cnt = 0
    s3_cnt = 0
    
    for p in matching_paths:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in s1_order:
                m_list = all_matches.get(s1_id, [])
                if m_list:
                    m_list_sorted = sorted(m_list)
                    f.write(f"{s1_id}\t{','.join(m_list_sorted)}\n")
                    if p == matching_paths[0]:
                        s1_with_matches += 1
                        total_matches_cnt += len(m_list_sorted)
                        for m in m_list_sorted:
                            if m.startswith("S2-"):
                                s2_cnt += 1
                            elif m.startswith("S3-"):
                                s3_cnt += 1
                else:
                    f.write(f"{s1_id}\t\n")
                    
    print(f"\nmatching_results.tsv successfully written to:")
    for p in matching_paths:
        print(f"  {p}")
        
    print(f"\nFinal Test Matching Summary:")
    print(f"  Total S1 entities: {total_test_s1:,}")
    print(f"  S1 with >=1 match: {s1_with_matches:,} ({s1_with_matches / total_test_s1 * 100:.2f}%)")
    print(f"  Singletons (0 match): {total_test_s1 - s1_with_matches:,} ({(total_test_s1 - s1_with_matches) / total_test_s1 * 100:.2f}%)")
    print(f"  Total matched targets: {total_matches_cnt:,}")
    print(f"  S2 matches: {s2_cnt:,} ({s2_cnt / total_matches_cnt * 100:.2f}%)")
    print(f"  S3 matches: {s3_cnt:,} ({s3_cnt / total_matches_cnt * 100:.2f}%)")
    
    # 2. Write candidate_pairs.tsv
    candidate_paths = [
        os.path.join(RESULTS_DIR, "candidate_pairs.tsv"),
        os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    ]
    
    for p in candidate_paths:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
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
                    
    print(f"\ncandidate_pairs.tsv successfully written to:")
    for p in candidate_paths:
        print(f"  {p}")
        
    print(f"\nAssembly completed in {time.time()-t0:.1f}s.", flush=True)

def validate_submission_files():
    print("\n========================================================")
    print("RUNNING OFFICIAL VALIDATOR")
    print("========================================================")
    validator_script = os.path.join(BASE_DIR, "student_resource", "utils", "validate_submission.py")
    matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    
    cmd = [
        sys.executable, validator_script,
        "--matching", matching_file,
        "--candidate", candidate_file,
        "--test-dir", TEST_DIR
    ]
    print(f"Executing: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Validator Output:\n" + res.stdout)
    if res.stderr:
        print("Validator Stderr:\n" + res.stderr)
    if res.returncode == 0:
        print("\n*** VALIDATION PASSED SUCCESSFULLY! ***")
    else:
        print(f"\n*** VALIDATION FAILED (exit code {res.returncode}) ***")
        sys.exit(1)

def create_submission_zip():
    print("\n========================================================")
    print("CREATING FINAL SUBMISSION PACKAGE ZIP")
    print("========================================================")
    zip_path = os.path.join(BASE_DIR, "submission.zip")
    
    files_to_pack = [
        (os.path.join(OUTPUT_DIR, "matching_results.tsv"), "output/matching_results.tsv"),
        (os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), "output/candidate_pairs.tsv"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "README.md"), "code/business_entity_resolution/README.md"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "requirements.txt"), "code/business_entity_resolution/requirements.txt"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "src", "features.py"), "code/business_entity_resolution/src/features.py"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "src", "inference.py"), "code/business_entity_resolution/src/inference.py"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "src", "train.py"), "code/business_entity_resolution/src/train.py"),
        (os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "src", "metric.py"), "code/business_entity_resolution/src/metric.py"),
        (os.path.join(BASE_DIR, "student_resource", "Documentation_template.md"), "Documentation_template.md"),
    ]
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for src, arcname in files_to_pack:
            if os.path.exists(src):
                print(f"  Adding {arcname} ({os.path.getsize(src):,d} bytes)...")
                zipf.write(src, arcname)
            else:
                print(f"  WARNING: File {src} not found!")
                
    print(f"\nFinal submission package created successfully at:")
    print(f"  {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.1f} MB)")

def main():
    print("=== STARTING FULL PRODUCTION PIPELINE ===")
    t_start = time.time()
    
    print(f"Loading LightGBM model from {MODEL_PATH}...", flush=True)
    booster = lgb.Booster(model_file=MODEL_PATH)
    
    # Process India
    process_india(booster)
    
    # Assemble both outputs
    assemble_submissions()
    
    # Validate
    validate_submission_files()
    
    # Create submission package
    create_submission_zip()
    
    print(f"\nALL OPERATIONS COMPLETED SUCCESSFULLY in {time.time()-t_start:.1f}s!")

if __name__ == "__main__":
    main()
