"""
End-to-End Leakage-Safe Training and Validation Pipeline.
1. Generates realistic training candidate pairs (positives + hard negatives from blocking).
2. Computes informative features (names, core names, transliteration, address tokens, numbers, domain stems).
3. Trains LightGBM with binary objective.
4. Evaluates on 25,000 held-out validation S1 entities.
5. Applies 1-to-1 target assignment and threshold optimization for Macro F0.5.
6. Measures exact official Macro F0.5.
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
from metric import compute_macro_f05

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

LEGAL_STOPWORDS = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "pllc", "pc", "co", "company", "gmbh", 
    "sa", "sarl", "services", "enterprises", "consultants", "solutions",
    "technologies", "technology", "management", "trading", "group", "holdings",
    "holding", "associates", "international", "global", "industries", "industry",
    "pra", "li", "m/s", "ms", "shri", "smt", "sri", "dr", "mr", "mrs", "the", "and", "of"
}

GENERIC_ADDR = {
    "road", "rd", "street", "st", "lane", "ln", "avenue", "ave", "court", "ct",
    "drive", "dr", "boulevard", "blvd", "floor", "fl", "building", "bldg", "near",
    "opp", "opposite", "door", "no", "plot", "shop", "flat", "block", "c/o", "h/no",
    "hno", "house", "room", "sector", "phase", "nagar", "colony", "city", "west",
    "east", "north", "south", "in", "at", "by", "unit", "suite", "ste", "apt", "apartment"
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
    # Normalize numbers (strip leading zeros e.g. 0201 -> 201)
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
    """
    Computes numerical feature vector for (s1, target) pair.
    """
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
    # 1. Load validation metadata
    with open("work/val_benchmark/val_gt.json", "r", encoding="utf-8") as f:
        val_gt_raw = json.load(f)
    val_gt = {k: set(v) for k, v in val_gt_raw.items()}
    val_s1_ids = set(val_gt.keys())
    print(f"Loaded validation GT: {len(val_s1_ids):,} entities.")
    
    # 2. Select separate training S1 entities (30,000 S1 entities, disjoint from validation!)
    print("Selecting separate training S1 entities...")
    gt_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t")
    train_pool = gt_df.filter(~pl.col("source1_entity_id").is_in(val_s1_ids))
    train_s1_sample = train_pool.sample(n=30_000, seed=123)
    train_s1_ids = set(train_s1_sample["source1_entity_id"].to_list())
    
    train_gt = {}
    train_target_ids = set()
    for row in train_s1_sample.iter_rows():
        s1_id = row[0]
        mstr = row[1] or ""
        if mstr.strip():
            mids = set(m.strip() for m in mstr.split(",") if m.strip())
            train_gt[s1_id] = mids
            train_target_ids.update(mids)
        else:
            train_gt[s1_id] = set()
    print(f"Sampled {len(train_s1_ids):,} training S1 entities ({len(train_target_ids):,} true target IDs).")
    
    # Load all required S1 records
    print("Loading S1 records for train + val...")
    needed_s1_ids = train_s1_ids | val_s1_ids
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t")
    s1_dict = {}
    for row in s1_df.filter(pl.col("entity_id").is_in(needed_s1_ids)).iter_rows():
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
    
    # Load targets needed for train + validation
    val_target_ids = set()
    for s in val_gt.values():
        val_target_ids.update(s)
        
    print(f"True targets needed: Train={len(train_target_ids):,}, Val={len(val_target_ids):,}")
    all_needed_targets = train_target_ids | val_target_ids
    
    print("Loading target pool (all true targets + 300,000 distractors from S2 and S3)...")
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), separator="\t")
    s2_needed = s2_df.filter(pl.col("entity_id").is_in(all_needed_targets))
    s2_dist = s2_df.filter(~pl.col("entity_id").is_in(all_needed_targets)).sample(n=150_000, seed=42)
    s2_pool = pl.concat([s2_needed, s2_dist])
    del s2_df, s2_needed, s2_dist
    
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), separator="\t")
    s3_needed = s3_df.filter(pl.col("entity_id").is_in(all_needed_targets))
    s3_dist = s3_df.filter(~pl.col("entity_id").is_in(all_needed_targets)).sample(n=150_000, seed=42)
    s3_pool = pl.concat([s3_needed, s3_dist])
    del s3_df, s3_needed, s3_dist
    
    target_pool = pl.concat([s2_pool, s3_pool])
    print(f"Loaded {len(target_pool):,} target records.")
    
    # Process targets and build blocking indexes
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
        
        # Build indexes
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

    # 3. Generate training data
    print("\nGenerating training candidate pairs...")
    X_train = []
    y_train = []
    
    for s1_id in train_s1_ids:
        s1 = s1_dict[s1_id]
        true_targets = train_gt[s1_id]
        cands = generate_candidates_for_s1(s1_id)
        
        # Include true targets (if in target_dict) to guarantee positive examples
        for tid in true_targets:
            if tid in target_dict:
                cands.add(tid)
                
        # Generate feature vectors
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
    
    # 4. Train LightGBM model
    print("\nTraining LightGBM model...")
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "n_estimators": 150,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    # Feature importances
    print("\nFeature Importances:")
    importances = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:25s}: {imp}")
        
    # 5. Evaluate on Validation Set
    print("\nEvaluating on 25,000 Validation S1 entities...")
    val_scored_pairs = [] # (s1_id, tid, score)
    
    for s1_id in val_s1_ids:
        s1 = s1_dict[s1_id]
        cands = generate_candidates_for_s1(s1_id)
        if not cands:
            continue
        cand_list = [tid for tid in cands if tid in target_dict]
        if not cand_list:
            continue
        feats_list = [compute_pair_features(s1, target_dict[tid]) for tid in cand_list]
        scores = model.predict_proba(np.array(feats_list, dtype=np.float32))[:, 1]
        
        for tid, sc in zip(cand_list, scores):
            val_scored_pairs.append((s1_id, tid, float(sc)))
            
    print(f"Total validation candidate pairs scored: {len(val_scored_pairs):,}")
    
    # 6. Threshold Tuning & Assignment Optimization
    print("\nTuning threshold and evaluating 1-to-1 Target Assignment...")
    
    for assign_1to1 in [False, True]:
        print(f"\n--- EXPERIMENT: 1-to-1 Target Assignment = {assign_1to1} ---")
        best_f05 = 0.0
        best_th = 0.0
        best_metrics = None
        
        for th in [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            # Filter by threshold
            filtered = [(s1, tid, sc) for s1, tid, sc in val_scored_pairs if sc >= th]
            
            if assign_1to1:
                # Group by target tid, sort by score desc, keep only highest scoring S1 for each tid
                filtered.sort(key=lambda x: x[2], reverse=True)
                assigned_targets = set()
                final_pairs = []
                for s1, tid, sc in filtered:
                    if tid not in assigned_targets:
                        assigned_targets.add(tid)
                        final_pairs.append((s1, tid, sc))
            else:
                final_pairs = filtered
                
            # Build predictions dict
            preds = defaultdict(set)
            for s1, tid, _ in final_pairs:
                preds[s1].add(tid)
                
            metrics = compute_macro_f05(val_gt, preds, all_s1_ids=list(val_s1_ids))
            f05 = metrics["macro_f05"]
            p = metrics["micro_precision"]
            r = metrics["micro_recall"]
            
            print(f"Th={th:.2f} | Macro F0.5={f05:.4f} | Prec={p:.4f} | Rec={r:.4f} | Singletons Acc={metrics['singleton_accuracy']:.4f} | Non-Singletons F0.5={metrics['non_singleton_f05']:.4f} | Pred S1s with matches={metrics['predicted_with_matches']:,}")
            
            if f05 > best_f05:
                best_f05 = f05
                best_th = th
                best_metrics = metrics
                
        print(f"\nBEST RESULT (Assign 1-to-1={assign_1to1}): Threshold={best_th:.2f}, Macro F0.5 = {best_f05:.4f}")
        print(f"Comparison with Baseline 0.282: {best_f05:.4f} vs 0.2820 (Improvement: +{best_f05 - 0.282:.4f} / +{(best_f05 - 0.282)/0.282*100:.1f}%)")

if __name__ == "__main__":
    main()
