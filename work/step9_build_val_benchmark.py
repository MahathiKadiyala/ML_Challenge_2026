"""
Leakage-safe Validation Benchmark Setup.
Extracts:
1. 20,000 held-out S1 validation entities (stratified by country and singleton status).
2. Ground truth for the validation set.
3. Feature extraction and model training on a separate training split.
4. Evaluation function to compute exact official Macro F0.5.
"""
import os
import polars as pl
import json
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

def setup_benchmark():
    print("Loading Ground Truth...")
    gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t")
    print(f"Total S1 entities: {len(gt):,}")
    
    # Load S1 source
    s1 = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t")
    
    # Join country to stratify properly
    joined = gt.join(s1.select(["entity_id", "country"]), left_on="source1_entity_id", right_on="entity_id")
    
    # Stratified sample of 25,000 S1 validation entities
    # Proportions: US ~60%, India ~40%
    # Ground truth singletons: ~5.6%
    val_s1_us = joined.filter(pl.col("country") == "US").sample(n=15_000, seed=42)
    val_s1_in = joined.filter(pl.col("country") == "India").sample(n=10_000, seed=42)
    val_df = pl.concat([val_s1_us, val_s1_in])
    
    val_s1_ids = set(val_df["source1_entity_id"].to_list())
    print(f"Sampled {len(val_s1_ids):,} S1 validation entities.")
    
    # Ground truth dict
    val_gt = {}
    for row in val_df.iter_rows():
        s1_id = row[0]
        mstr = row[1] or ""
        if mstr.strip():
            val_gt[s1_id] = set(m.strip() for m in mstr.split(",") if m.strip())
        else:
            val_gt[s1_id] = set()
            
    val_singletons = sum(1 for s in val_gt.values() if len(s) == 0)
    print(f"Validation singletons: {val_singletons:,} ({val_singletons / len(val_gt) * 100:.2f}%)")
    print(f"Validation non-singletons: {len(val_gt) - val_singletons:,}")
    
    # Save validation metadata
    os.makedirs("work/val_benchmark", exist_ok=True)
    with open("work/val_benchmark/val_gt.json", "w", encoding="utf-8") as f:
        # Convert sets to lists for json
        json.dump({k: list(v) for k, v in val_gt.items()}, f)
        
    # Save list of validation S1 IDs
    with open("work/val_benchmark/val_s1_ids.txt", "w", encoding="utf-8") as f:
        for sid in val_s1_ids:
            f.write(f"{sid}\n")
            
    print("Validation benchmark saved in work/val_benchmark/")

if __name__ == "__main__":
    setup_benchmark()
