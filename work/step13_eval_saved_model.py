"""
Evaluate the saved final LightGBM model on the 25,000 validation benchmark.
"""
import os
import json
import polars as pl
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from step12_train_final_model import (
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem, compute_pair_features
)
from metric import compute_macro_f05

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

def run():
    print("Loading validation GT...")
    with open("work/val_benchmark/val_gt.json", "r", encoding="utf-8") as f:
        val_gt_raw = json.load(f)
    val_gt = {k: set(v) for k, v in val_gt_raw.items()}
    val_s1_ids = set(val_gt.keys())
    print(f"Validation entities: {len(val_s1_ids):,}")
    
    val_target_ids = set()
    for s in val_gt.values():
        val_target_ids.update(s)
        
    print("Loading S1 validation records...")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t")
    s1_dict = {}
    for row in s1_df.filter(pl.col("entity_id").is_in(val_s1_ids)).iter_rows():
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
    
    print("Loading target records (all true validation targets + 250,000 distractors)...")
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), separator="\t")
    s2_needed = s2_df.filter(pl.col("entity_id").is_in(val_target_ids))
    s2_dist = s2_df.filter(~pl.col("entity_id").is_in(val_target_ids)).sample(n=125_000, seed=42)
    s2_pool = pl.concat([s2_needed, s2_dist])
    del s2_df, s2_needed, s2_dist
    
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), separator="\t")
    s3_needed = s3_df.filter(pl.col("entity_id").is_in(val_target_ids))
    s3_dist = s3_df.filter(~pl.col("entity_id").is_in(val_target_ids)).sample(n=125_000, seed=42)
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
    print("Loading saved LightGBM model...")
    booster = lgb.Booster(model_file="work/models/lightgbm_model.txt")
    
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

    print("Scoring validation candidates...")
    val_scored_pairs = []
    for s1_id in val_s1_ids:
        s1 = s1_dict[s1_id]
        cands = generate_candidates_for_s1(s1_id)
        if not cands:
            continue
        cand_list = [tid for tid in cands if tid in target_dict]
        if not cand_list:
            continue
        feats_list = [compute_pair_features(s1, target_dict[tid]) for tid in cand_list]
        scores = booster.predict(np.array(feats_list, dtype=np.float32))
        for tid, sc in zip(cand_list, scores):
            val_scored_pairs.append((s1_id, tid, float(sc)))
            
    print(f"Scored {len(val_scored_pairs):,} pairs. Evaluating at thresholds with 1-to-1 assignment...")
    
    for th in [0.55, 0.60, 0.65, 0.70, 0.75]:
        filtered = [(s1, tid, sc) for s1, tid, sc in val_scored_pairs if sc >= th]
        filtered.sort(key=lambda x: x[2], reverse=True)
        assigned_targets = set()
        final_pairs = []
        for s1, tid, sc in filtered:
            if tid not in assigned_targets:
                assigned_targets.add(tid)
                final_pairs.append((s1, tid, sc))
                
        preds = defaultdict(set)
        for s1, tid, _ in final_pairs:
            preds[s1].add(tid)
            
        m = compute_macro_f05(val_gt, preds, all_s1_ids=list(val_s1_ids))
        print(f"Th={th:.2f} | Macro F0.5={m['macro_f05']:.4f} | Prec={m['micro_precision']:.4f} | Rec={m['micro_recall']:.4f} | Singletons Acc={m['singleton_accuracy']:.4f}")

if __name__ == "__main__":
    run()
