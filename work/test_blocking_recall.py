import os
import json
import time
import polars as pl
from collections import defaultdict
from step12_train_final_model import (
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem
)

BASE_DIR = "student_resource/dataset"

with open("work/val_benchmark/val_gt.json", "r", encoding="utf-8") as f:
    val_gt = json.load(f)

val_s1_ids = set(val_gt.keys())
print(f"Loaded {len(val_s1_ids):,} validation entities.")

# Load 5000 S1 sample
s1_df = pl.read_csv(f"{BASE_DIR}/train/train_source1.tsv", separator="\t")
sample_s1 = s1_df.filter(pl.col("entity_id").is_in(val_s1_ids)).head(5000)

s1_dict = {}
for row in sample_s1.iter_rows():
    eid, name, addr = row[0], row[1], row[2]
    c_tokens = extract_core_tokens(name)
    nums, addr_words = extract_addr_features(addr)
    s1_dict[eid] = {
        "c_tokens": c_tokens,
        "sorted": " ".join(sorted(c_tokens)),
        "first2": " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else ""),
        "nums": nums,
        "addr_words": addr_words
    }

# Get targets needed
all_true_targets = set()
for eid in s1_dict:
    all_true_targets.update(val_gt[eid])

print(f"Sample has {len(s1_dict):,} S1 and {len(all_true_targets):,} true target IDs.")

# Load target pool
s2_df = pl.read_csv(f"{BASE_DIR}/train/train_source2.tsv", separator="\t").head(300000)
s3_df = pl.read_csv(f"{BASE_DIR}/train/train_source3.tsv", separator="\t").head(300000)
# Ensure all true targets are in pool
s2_extra = pl.read_csv(f"{BASE_DIR}/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(all_true_targets))
s3_extra = pl.read_csv(f"{BASE_DIR}/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(all_true_targets))

target_df = pl.concat([s2_df, s3_df, s2_extra, s3_extra]).unique(subset=["entity_id"])
print(f"Target pool size: {len(target_df):,}")

# Build full indexes without cap during indexing
idx_sorted = defaultdict(list)
idx_first2 = defaultdict(list)
idx_num_addr = defaultdict(list)

for row in target_df.iter_rows():
    tid, name, addr = row[0], row[1], row[2] or ""
    c_tokens = extract_core_tokens(name)
    nums, addr_words = extract_addr_features(addr)
    st = " ".join(sorted(c_tokens))
    if st:
        idx_sorted[st].append(tid)
    f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
    if f2:
        idx_first2[f2].append(tid)
    if nums and addr_words:
        for num in nums[:2]:
            for word in addr_words[:2]:
                idx_num_addr[(num, word)].append(tid)

# Compare strategies:
# Strategy A: Capped indexing (what run_full_inference was doing)
# Strategy B: Threshold filtering (skip block if len > 50)
print("\n--- Evaluating Strategy B: Discarding blocks with >50 items ---")
found = 0
total_pairs = 0
cand_sizes = []

for s1_id, s1 in s1_dict.items():
    cands = set()
    st = s1["sorted"]
    if st in idx_sorted and len(idx_sorted[st]) <= 50:
        cands.update(idx_sorted[st])
    f2 = s1["first2"]
    if f2 in idx_first2 and len(idx_first2[f2]) <= 50:
        cands.update(idx_first2[f2])
    if s1["nums"] and s1["addr_words"]:
        for num in s1["nums"][:2]:
            for word in s1["addr_words"][:2]:
                k = (num, word)
                if k in idx_num_addr and len(idx_num_addr[k]) <= 50:
                    cands.update(idx_num_addr[k])
                    
    cand_sizes.append(len(cands))
    true_targets = set(val_gt[s1_id])
    if true_targets:
        found += len(cands & true_targets)

total_true = sum(len(val_gt[eid]) for eid in s1_dict)
print(f"Recall: {found}/{total_true} = {found/total_true*100:.2f}%")
print(f"Average candidates per S1: {sum(cand_sizes)/len(cand_sizes):.1f}")
print(f"Max candidates: {max(cand_sizes)}")
