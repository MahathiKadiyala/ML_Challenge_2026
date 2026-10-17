"""
Full Test Inference Pipeline.
Runs multi-pass blocking, feature extraction, LightGBM inference, and 1-to-1 target assignment
across the complete test datasets partitioned by country:
1. France (259,452 S1 records)
2. United States (663,106 S1 records)
3. India (809,986 S1 records)

Outputs:
- results/matching_results.tsv
- results/candidate_pairs.tsv
- student_resource/output/matching_results.tsv
- student_resource/output/candidate_pairs.tsv
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
OUTPUT_DIR = os.path.join(BASE_DIR, "student_resource", "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "work", "models", "lightgbm_model.txt")

LEGAL_STOPWORDS = {
    # English & US
    "pvt", "ltd", "private", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "pllc", "pc", "co", "company", "gmbh", 
    "services", "enterprises", "consultants", "solutions",
    "technologies", "technology", "management", "trading", "group", "holdings",
    "holding", "associates", "industries", "industry", "lp", "llp",
    # French
    "sa", "sarl", "sasu", "sas", "eurl", "sci", "fils", "freres", "cie", "soc",
    # Indic & Honorifics
    "pra", "li", "m/s", "ms", "shri", "smt", "sri", "dr", "mr", "mrs",
    "the", "and", "of", "de", "du", "des", "la", "le", "les", "et"
}

GENERIC_ADDR = {
    # English & US
    "road", "rd", "street", "st", "lane", "ln", "avenue", "ave", "court", "ct",
    "drive", "dr", "boulevard", "blvd", "floor", "fl", "building", "bldg", "near",
    "opp", "opposite", "door", "no", "plot", "shop", "flat", "block", "c/o", "h/no",
    "hno", "house", "room", "sector", "phase", "nagar", "colony", "city", "west",
    "east", "north", "south", "in", "at", "by", "unit", "suite", "ste", "apt", "apartment",
    # French
    "rue", "r", "bd", "av", "allee", "allée", "cours", "chemin", "place", "pl"
}

def clean_and_translit(text):
    if not text:
        return ""
    text = unidecode.unidecode(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"([a-z])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_core_tokens(name):
    norm = clean_and_translit(name)
    tokens = norm.split()
    if tokens and tokens[-1] in {"com", "org", "net", "in", "co", "us", "fr"}:
        tokens.pop()
    content_tokens = [t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1]
    return content_tokens

def extract_addr_features(addr):
    norm = clean_and_translit(addr)
    tokens = norm.split()
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

def compute_pair_features(s1, tg):
    s1_name_clean = s1["clean_name"]
    tg_name_clean = tg["clean_name"]
    s1_addr_clean = s1["clean_addr"]
    tg_addr_clean = tg["clean_addr"]
    
    # 1. Name similarities
    name_ratio = fuzz.ratio(s1_name_clean, tg_name_clean)
    name_token_sort = fuzz.token_sort_ratio(s1_name_clean, tg_name_clean)
    name_token_set = fuzz.token_set_ratio(s1_name_clean, tg_name_clean)
    
    # 2. Core name similarities
    s1_core = s1["core_str"]
    tg_core = tg["core_str"]
    core_ratio = fuzz.ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    core_token_sort = fuzz.token_sort_ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    core_token_set = fuzz.token_set_ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    
    # First token match
    s1_f1 = s1["first1"]
    tg_f1 = tg["first1"]
    first_token_match = 1.0 if s1_f1 and tg_f1 and s1_f1 == tg_f1 else 0.0
    
    # 3. Address similarities
    addr_ratio = fuzz.ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    addr_token_set = fuzz.token_set_ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    
    # 4. Number features
    s1_nums = s1["nums_set"]
    tg_nums = tg["nums_set"]
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
    d_stem = tg["domain_stem"]
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
    is_s2 = 1.0 if tg["tid"].startswith("S2-") else 0.0
    
    return [
        name_ratio, name_token_sort, name_token_set,
        core_ratio, core_token_sort, core_token_set, first_token_match,
        addr_ratio, addr_token_sort, addr_token_set,
        has_nums, num_jaccard, num_exact,
        has_domain, domain_match,
        len_diff_name, len_diff_addr,
        is_s2
    ]

def process_country(country_name, booster, threshold=0.60):
    print(f"\n========================================================")
    print(f"PROCESSING COUNTRY: {country_name}")
    print(f"========================================================")
    
    # 1. Load target records for this country
    print(f"Loading S2 records for {country_name}...")
    s2_df = pl.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), separator="\t").filter(pl.col("country") == country_name)
    print(f"Loading S3 records for {country_name}...")
    s3_df = pl.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), separator="\t").filter(pl.col("country") == country_name)
    target_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    print(f"Total target records for {country_name}: {len(target_df):,}")
    
    # 2. Process targets & build blocking indexes
    print("Building blocking indexes...")
    target_dict = {}
    idx_sorted_tokens = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_num_addr = defaultdict(list)
    idx_domain = defaultdict(list)
    
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
        if st:
            idx_sorted_tokens[st].append(tid)
        f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
        if f2:
            idx_first2[f2].append(tid)
            
        if nums and addr_words:
            for num in nums[:2]:
                for word in addr_words[:2]:
                    idx_num_addr[(num, word)].append(tid)
                    
        if d_stem:
            idx_domain[d_stem].append(tid)
            
    del target_df
    gc.collect()
    print("Indexes built.")
    
    # 3. Load S1 for this country
    print(f"Loading S1 records for {country_name}...")
    s1_df = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t").filter(pl.col("country") == country_name)
    total_s1 = len(s1_df)
    print(f"Total S1 entities for {country_name}: {total_s1:,}")
    
    country_candidates = {}
    high_score_pairs = [] # (s1_id, tid, score)
    
    batch_size = 50_000
    n_batches = (total_s1 + batch_size - 1) // batch_size
    
    print(f"Running inference in {n_batches} batches...")
    
    for b_idx in range(n_batches):
        start_idx = b_idx * batch_size
        end_idx = min(start_idx + batch_size, total_s1)
        sub_df = s1_df.slice(start_idx, end_idx - start_idx)
        
        batch_pairs = [] # (s1_id, tid, feats)
        
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
            
            # Blocking
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
                            lst = idx_num_addr[k]
                            if len(lst) <= 50:
                                cands.update(lst)
            if c_tokens:
                concat = "".join(c_tokens)
                if concat in idx_domain:
                    cands.update(idx_domain[concat])
                for tok in c_tokens[:2]:
                    if len(tok) >= 4 and tok in idx_domain:
                        cands.update(idx_domain[tok])
                        
            country_candidates[s1_id] = cands
            
            for tid in cands:
                tg = target_dict[tid]
                feats = compute_pair_features(s1_info, tg)
                batch_pairs.append((s1_id, tid, feats))
                
        if batch_pairs:
            feats_mat = np.array([p[2] for p in batch_pairs], dtype=np.float32)
            scores = booster.predict(feats_mat)
            for i, sc in enumerate(scores):
                if sc >= threshold:
                    high_score_pairs.append((batch_pairs[i][0], batch_pairs[i][1], float(sc)))
                    
        print(f"  Batch {b_idx+1}/{n_batches} done ({end_idx}/{total_s1} S1 entities processed). High-score pairs so far: {len(high_score_pairs):,}")
        
    del s1_df, target_dict, idx_sorted_tokens, idx_first2, idx_num_addr, idx_domain
    gc.collect()
    
    # 4. Apply 1-to-1 Target Assignment on this country
    print(f"\nApplying 1-to-1 Target Assignment for {country_name}...")
    high_score_pairs.sort(key=lambda x: x[2], reverse=True)
    assigned_targets = set()
    
    country_matches = defaultdict(list)
    for s1_id, tid, sc in high_score_pairs:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            country_matches[s1_id].append(tid)
            
    print(f"Results for {country_name}:")
    print(f"  S1 entities: {total_s1:,}")
    print(f"  S1 entities with matches: {len(country_matches):,} ({len(country_matches)/total_s1*100:.2f}%)")
    print(f"  Singletons (0 matches): {total_s1 - len(country_matches):,} ({(total_s1 - len(country_matches))/total_s1*100:.2f}%)")
    print(f"  Total matched target entities: {len(assigned_targets):,}")
    
    return country_candidates, country_matches

def main():
    print("=== STARTING FULL TEST INFERENCE ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading LightGBM model from {MODEL_PATH}...")
    booster = lgb.Booster(model_file=MODEL_PATH)
    
    # Process each country
    all_candidates = {}
    all_matches = {}
    
    for country in ["France", "US", "India"]:
        cands, matches = process_country(country, booster, threshold=0.60)
        all_candidates.update(cands)
        all_matches.update(matches)
        gc.collect()
        
    print("\n========================================================")
    print("WRITING OFFICIAL SUBMISSION FILES")
    print("========================================================")
    
    # Read test_source1 to preserve exact original row order
    s1_all = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t")
    s1_order = s1_all["entity_id"].to_list()
    del s1_all
    
    total_test_s1 = len(s1_order)
    print(f"Total test S1 entities in original order: {total_test_s1:,}")
    
    # 1. Write matching_results.tsv
    matching_paths = [
        os.path.join(RESULTS_DIR, "matching_results.tsv"),
        os.path.join(OUTPUT_DIR, "matching_results.tsv")
    ]
    
    print("Writing matching_results.tsv...")
    total_matches_cnt = 0
    s2_cnt = 0
    s3_cnt = 0
    s1_with_matches = 0
    
    for p in matching_paths:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in s1_order:
                m_list = all_matches.get(s1_id, [])
                if m_list:
                    # Sort for reproducibility
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
                    
    print(f"matching_results.tsv written to:")
    for p in matching_paths:
        print(f"  {p}")
        
    print(f"\nFinal Test Matching Statistics:")
    print(f"  Total S1 entities: {total_test_s1:,}")
    print(f"  S1 entities with >=1 match: {s1_with_matches:,} ({s1_with_matches / total_test_s1 * 100:.2f}%)")
    print(f"  Singletons (0 matches): {total_test_s1 - s1_with_matches:,} ({(total_test_s1 - s1_with_matches) / total_test_s1 * 100:.2f}%)")
    print(f"  Total matched target entities: {total_matches_cnt:,}")
    print(f"  Total S2 matches: {s2_cnt:,} ({s2_cnt / total_matches_cnt * 100:.2f}%)")
    print(f"  Total S3 matches: {s3_cnt:,} ({s3_cnt / total_matches_cnt * 100:.2f}%)")
    
    # 2. Write candidate_pairs.tsv
    candidate_paths = [
        os.path.join(RESULTS_DIR, "candidate_pairs.tsv"),
        os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    ]
    
    print("\nWriting candidate_pairs.tsv...")
    for p in candidate_paths:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in s1_order:
                c_set = all_candidates.get(s1_id, set())
                # Ensure all matched IDs are included in candidates (as required by validator)
                m_list = all_matches.get(s1_id, [])
                if m_list:
                    c_set = c_set | set(m_list)
                if c_set:
                    c_list_sorted = sorted(c_set)
                    f.write(f"{s1_id}\t{','.join(c_list_sorted)}\n")
                else:
                    f.write(f"{s1_id}\t\n")
                    
    print(f"candidate_pairs.tsv written to:")
    for p in candidate_paths:
        print(f"  {p}")

if __name__ == "__main__":
    main()
