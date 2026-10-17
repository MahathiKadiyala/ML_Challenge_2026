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

def compute_pair_features(s1, tg):
    s1_name_clean = s1["clean_name"]
    tg_name_clean = tg["clean_name"]
    s1_addr_clean = s1["clean_addr"]
    tg_addr_clean = tg["clean_addr"]
    
    name_ratio = fuzz.ratio(s1_name_clean, tg_name_clean)
    name_token_sort = fuzz.token_sort_ratio(s1_name_clean, tg_name_clean)
    name_token_set = fuzz.token_set_ratio(s1_name_clean, tg_name_clean)
    
    s1_core = s1["core_str"]
    tg_core = tg["core_str"]
    core_ratio = fuzz.ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    core_token_sort = fuzz.token_sort_ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    core_token_set = fuzz.token_set_ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    
    first_token_match = 1.0 if s1["first1"] and tg["first1"] and s1["first1"] == tg["first1"] else 0.0
    
    addr_ratio = fuzz.ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    addr_token_set = fuzz.token_set_ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    
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
        
    d_stem = tg["domain_stem"]
    if d_stem:
        has_domain = 1.0
        domain_match = 1.0 if d_stem in s1_name_clean or (s1_core and d_stem in s1_core) else 0.0
    else:
        has_domain = 0.0
        domain_match = 0.0
        
    len_diff_name = abs(len(s1_name_clean) - len(tg_name_clean))
    len_diff_addr = abs(len(s1_addr_clean) - len(tg_addr_clean))
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

def add_to_index(idx, blacklist, key, tid, max_size=50):
    if key in blacklist:
        return
    lst = idx[key]
    if len(lst) >= max_size:
        del idx[key]
        blacklist.add(key)
        return
    lst.append(tid)

def main():
    print("Loading LightGBM model...", flush=True)
    booster = lgb.Booster(model_file=MODEL_PATH)
    
    print("Loading France targets...", flush=True)
    t0 = time.time()
    s2_df = pl.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), separator="\t").filter(pl.col("country") == "France")
    s3_df = pl.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), separator="\t").filter(pl.col("country") == "France")
    target_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    print(f"Loaded {len(target_df):,} targets in {time.time()-t0:.1f}s.", flush=True)
    
    tids = target_df["entity_id"].to_list()
    names = target_df["business_name"].to_list()
    addrs = target_df["business_address"].fill_null("").to_list()
    del target_df
    gc.collect()
    
    print("Building high-selectivity pruned indexes...", flush=True)
    t0 = time.time()
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
            add_to_index(idx_sorted_tokens, bl_sorted, st, tid, 50)
            
        f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
        if f2:
            add_to_index(idx_first2, bl_first2, f2, tid, 50)
            
        if nums and addr_words:
            for num in nums[:2]:
                for word in addr_words[:2]:
                    add_to_index(idx_num_addr, bl_num_addr, (num, word), tid, 50)
                    
        if d_stem:
            add_to_index(idx_domain, bl_domain, d_stem, tid, 50)
            
    del tids, names, addrs, bl_sorted, bl_first2, bl_num_addr, bl_domain
    gc.collect()
    print(f"High-selectivity indexes built in {time.time()-t0:.1f}s.", flush=True)
    print(f"  idx_sorted_tokens keys: {len(idx_sorted_tokens):,}")
    print(f"  idx_first2 keys: {len(idx_first2):,}")
    print(f"  idx_num_addr keys: {len(idx_num_addr):,}")
    
    # Load 10,000 S1
    s1_df = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t").filter(pl.col("country") == "France").head(10000)
    s1_tids = s1_df["entity_id"].to_list()
    s1_names = s1_df["business_name"].to_list()
    s1_addrs = s1_df["business_address"].fill_null("").to_list()
    del s1_df
    gc.collect()
    
    print(f"\nBenchmarking inference on {len(s1_tids):,} S1 entities...", flush=True)
    t_inf = time.time()
    batch_pairs = []
    total_cands = 0
    
    for s1_id, name, addr in zip(s1_tids, s1_names, s1_addrs):
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
                    
        total_cands += len(cands)
        
        # Fast short-circuiting pre-filter
        for tid in cands:
            tg = target_dict[tid]
            if (s1_f1 and tg["first1"] and s1_f1 == tg["first1"]) or \
               (s1_nums and tg["nums_set"] and (s1_nums & tg["nums_set"])) or \
               (tg["domain_stem"] and tg["domain_stem"] in c_name) or \
               (fuzz.ratio(c_name, tg["clean_name"]) >= 45):
                feats = compute_pair_features(s1_info, tg)
                batch_pairs.append((s1_id, tid, feats))
                
    t_feat = time.time() - t_inf
    print(f"Generated {total_cands:,} candidates ({total_cands/len(s1_tids):.1f} per S1).", flush=True)
    print(f"Features computed for {len(batch_pairs):,} pairs in {t_feat:.2f}s.", flush=True)
    
    t_pred = time.time()
    feats_mat = np.array([p[2] for p in batch_pairs], dtype=np.float32)
    scores = booster.predict(feats_mat)
    accepted = sum(1 for sc in scores if sc >= 0.60)
    print(f"Model prediction in {time.time()-t_pred:.2f}s. Accepted pairs (score >= 0.60): {accepted:,}", flush=True)
    
    total_time = time.time() - t_inf
    print(f"--> TOTAL 10,000 S1 in {total_time:.2f}s ({len(s1_tids)/total_time:.1f} S1/sec)!", flush=True)
    print(f"--> Projected 259k France: {259452 / (len(s1_tids)/total_time) / 60:.1f} minutes!", flush=True)

if __name__ == "__main__":
    main()
