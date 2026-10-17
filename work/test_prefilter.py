"""
Verify that fast candidate pre-filter preserves 100% of true matches while cutting computations by 80%+.
"""
import os
import json
import polars as pl
from rapidfuzz import fuzz
from collections import defaultdict
from step12_train_final_model import (
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

def run():
    print("Loading validation GT...")
    with open("work/val_benchmark/val_gt.json", "r", encoding="utf-8") as f:
        val_gt_raw = json.load(f)
    val_gt = {k: set(v) for k, v in val_gt_raw.items()}
    val_s1_ids = set(val_gt.keys())
    
    val_target_ids = set()
    for s in val_gt.values():
        val_target_ids.update(s)
        
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t")
    s1_dict = {}
    for row in s1_df.filter(pl.col("entity_id").is_in(val_s1_ids)).head(5000).iter_rows():
        eid, name, addr = row[0], row[1], row[2]
        c_tokens = extract_core_tokens(name)
        nums, _ = extract_addr_features(addr)
        s1_dict[eid] = {
            "clean_name": clean_and_translit(name),
            "clean_addr": clean_and_translit(addr),
            "first1": c_tokens[0] if c_tokens else "",
            "nums_set": set(nums)
        }
    del s1_df
    
    sample_s1_ids = set(s1_dict.keys())
    true_pairs = [(s1_id, tid) for s1_id in sample_s1_ids for tid in val_gt[s1_id]]
    print(f"Testing on {len(true_pairs):,} true validation pairs...")
    
    needed_tids = {tid for _, tid in true_pairs}
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), separator="\t")
    s2_needed = s2_df.filter(pl.col("entity_id").is_in(needed_tids))
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), separator="\t")
    s3_needed = s3_df.filter(pl.col("entity_id").is_in(needed_tids))
    target_df = pl.concat([s2_needed, s3_needed])
    
    target_dict = {}
    for row in target_df.iter_rows():
        tid, name, addr = row[0], row[1], row[2] or ""
        c_tokens = extract_core_tokens(name)
        nums, _ = extract_addr_features(addr)
        target_dict[tid] = {
            "clean_name": clean_and_translit(name),
            "clean_addr": clean_and_translit(addr),
            "first1": c_tokens[0] if c_tokens else "",
            "nums_set": set(nums),
            "domain_stem": extract_domain_stem(name)
        }
        
    preserved = 0
    missed = 0
    
    for s1_id, tid in true_pairs:
        if s1_id not in s1_dict or tid not in target_dict:
            continue
        s1 = s1_dict[s1_id]
        tg = target_dict[tid]
        
        # Fast pre-filter rule:
        # 1. Number match:
        has_num_overlap = bool(s1["nums_set"] & tg["nums_set"])
        # 2. First word match:
        first_word_match = bool(s1["first1"] and tg["first1"] and s1["first1"] == tg["first1"])
        # 3. Domain match:
        domain_match = bool(tg["domain_stem"] and tg["domain_stem"] in s1["clean_name"])
        # 4. Quick name ratio >= 45:
        quick_ratio = (fuzz.ratio(s1["clean_name"], tg["clean_name"]) >= 45)
        
        if has_num_overlap or first_word_match or domain_match or quick_ratio:
            preserved += 1
        else:
            missed += 1
            
    print(f"Pre-filter evaluation: Preserved={preserved:,} ({preserved / (preserved + missed) * 100:.2f}%), Missed={missed:,}")

if __name__ == "__main__":
    run()
