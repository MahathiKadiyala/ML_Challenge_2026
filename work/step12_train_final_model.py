"""
Train the final production LightGBM model on 80,000 diverse training entities.
Includes expanded multilingual, French, and Indic legal & address normalizations.
Saves model to work/final_lgb_model.txt.
"""
import os
import sys
import json
import re
import unidecode
import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

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

FEATURE_NAMES = [
    "name_ratio", "name_token_sort", "name_token_set",
    "core_ratio", "core_token_sort", "core_token_set", "first_token_match",
    "addr_ratio", "addr_token_sort", "addr_token_set",
    "has_nums", "num_jaccard", "num_exact",
    "has_domain", "domain_match",
    "len_diff_name", "len_diff_addr",
    "is_s2"
]

def main():
    print("=== TRAINING FINAL HIGH-CAPACITY LIGHTGBM MODEL ===")
    
    # Sample 80,000 S1 training entities (48k US, 32k India)
    gt_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t")
    s1_meta = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t").select(["entity_id", "country"])
    gt_with_meta = gt_df.join(s1_meta, left_on="source1_entity_id", right_on="entity_id")
    
    s1_us_sample = gt_with_meta.filter(pl.col("country") == "US").sample(n=48_000, seed=100)
    s1_in_sample = gt_with_meta.filter(pl.col("country") == "India").sample(n=32_000, seed=100)
    train_sample = pl.concat([s1_us_sample, s1_in_sample])
    train_s1_ids = set(train_sample["source1_entity_id"].to_list())
    
    train_gt = {}
    train_target_ids = set()
    for row in train_sample.iter_rows():
        s1_id = row[0]
        mstr = row[1] or ""
        if mstr.strip():
            mids = set(m.strip() for m in mstr.split(",") if m.strip())
            train_gt[s1_id] = mids
            train_target_ids.update(mids)
        else:
            train_gt[s1_id] = set()
    print(f"Sampled {len(train_s1_ids):,} training S1 entities ({len(train_target_ids):,} true target IDs).")
    
    print("Loading S1 records...")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t")
    s1_dict = {}
    for row in s1_df.filter(pl.col("entity_id").is_in(train_s1_ids)).iter_rows():
        eid, name, addr, ctry = row[0], row[1], row[2], row[3]
        c_tokens = extract_core_tokens(name)
        nums, addr_words = extract_addr_features(addr)
        s1_dict[eid] = {
            "eid": eid,
            "name": name,
            "addr": addr,
            "country": ctry,
            "clean_name": clean_and_translit(name),
            "clean_addr": clean_and_translit(addr),
            "tokens": c_tokens,
            "core_str": " ".join(c_tokens),
            "sorted_tokens": " ".join(sorted(c_tokens)),
            "first1": c_tokens[0] if c_tokens else "",
            "first2": " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else ""),
            "nums": nums,
            "nums_set": set(nums),
            "addr_words": addr_words
        }
    del s1_df
    
    print("Loading target records (all true targets + 350,000 distractors)...")
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), separator="\t")
    s2_needed = s2_df.filter(pl.col("entity_id").is_in(train_target_ids))
    s2_dist = s2_df.filter(~pl.col("entity_id").is_in(train_target_ids)).sample(n=175_000, seed=42)
    s2_pool = pl.concat([s2_needed, s2_dist])
    del s2_df, s2_needed, s2_dist
    
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), separator="\t")
    s3_needed = s3_df.filter(pl.col("entity_id").is_in(train_target_ids))
    s3_dist = s3_df.filter(~pl.col("entity_id").is_in(train_target_ids)).sample(n=175_000, seed=42)
    s3_pool = pl.concat([s3_needed, s3_dist])
    del s3_df, s3_needed, s3_dist
    
    target_pool = pl.concat([s2_pool, s3_pool])
    print(f"Total target pool size: {len(target_pool):,}")
    
    target_dict = {}
    idx_sorted_tokens = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_num_addr = defaultdict(list)
    idx_domain = defaultdict(list)
    
    for row in target_pool.iter_rows():
        tid, name, addr, ctry = row[0], row[1], row[2] or "", row[3]
        c_tokens = extract_core_tokens(name)
        nums, addr_words = extract_addr_features(addr)
        d_stem = extract_domain_stem(name)
        
        target_dict[tid] = {
            "tid": tid,
            "name": name,
            "addr": addr,
            "country": ctry,
            "clean_name": clean_and_translit(name),
            "clean_addr": clean_and_translit(addr),
            "tokens": c_tokens,
            "core_str": " ".join(c_tokens),
            "sorted_tokens": " ".join(sorted(c_tokens)),
            "first1": c_tokens[0] if c_tokens else "",
            "nums": nums,
            "nums_set": set(nums),
            "addr_words": addr_words,
            "domain_stem": d_stem
        }
        
        st = " ".join(sorted(c_tokens))
        if st:
            idx_sorted_tokens[(ctry, st)].append(tid)
        f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
        if f2:
            idx_first2[(ctry, f2)].append(tid)
            
        if nums and addr_words:
            for num in nums[:2]:
                for word in addr_words[:2]:
                    idx_num_addr[(ctry, num, word)].append(tid)
                    
        if d_stem:
            idx_domain[(ctry, d_stem)].append(tid)
            
    del target_pool
    print("Blocking indexes built.")
    
    def generate_candidates_for_s1(s1_id):
        s1 = s1_dict[s1_id]
        c = s1["country"]
        s_toks = s1["sorted_tokens"]
        f2 = s1["first2"]
        nums = s1["nums"]
        addr_words = s1["addr_words"]
        tokens = s1["tokens"]
        
        cands = set()
        if s_toks and (c, s_toks) in idx_sorted_tokens:
            cands.update(idx_sorted_tokens[(c, s_toks)])
        if f2 and (c, f2) in idx_first2:
            cands.update(idx_first2[(c, f2)])
        if nums and addr_words:
            for num in nums[:2]:
                for word in addr_words[:2]:
                    k = (c, num, word)
                    if k in idx_num_addr:
                        lst = idx_num_addr[k]
                        if len(lst) <= 50:
                            cands.update(lst)
        if tokens:
            concat = "".join(tokens)
            if (c, concat) in idx_domain:
                cands.update(idx_domain[(c, concat)])
            for tok in tokens[:2]:
                if len(tok) >= 4 and (c, tok) in idx_domain:
                    cands.update(idx_domain[(c, tok)])
        return cands

    print("\nGenerating training candidate pairs...")
    X_train = []
    y_train = []
    
    for s1_id in train_s1_ids:
        s1 = s1_dict[s1_id]
        true_targets = train_gt[s1_id]
        cands = generate_candidates_for_s1(s1_id)
        
        for tid in true_targets:
            if tid in target_dict:
                cands.add(tid)
                
        for tid in cands:
            if tid not in target_dict:
                continue
            tg = target_dict[tid]
            is_true = 1 if tid in true_targets else 0
            feats = compute_pair_features(s1, tg)
            X_train.append(feats)
            y_train.append(is_true)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Generated {len(X_train):,} training pairs: Positives={y_train.sum():,} ({y_train.mean()*100:.2f}%), Negatives={len(y_train) - y_train.sum():,}")
    
    print("\nTraining LightGBM model...")
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 25,
        "feature_fraction": 0.9,
        "n_estimators": 200,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    print("\nFeature Importances:")
    importances = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:25s}: {imp}")
        
    os.makedirs("work/models", exist_ok=True)
    model_path = "work/models/lightgbm_model.txt"
    model.booster_.save_model(model_path)
    print(f"\nFinal model saved to: {model_path}")

if __name__ == "__main__":
    main()
