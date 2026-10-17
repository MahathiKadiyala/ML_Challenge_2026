"""
Training pipeline for Business Entity Resolution LightGBM model.
Generates balanced pairs (positives + hard negatives from blocking)
and trains a production LightGBM classifier.
"""
import os
import sys
import gc
import json
import time
import argparse
import polars as pl
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from rapidfuzz import fuzz

from features import (
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem, compute_pair_features, FEATURE_NAMES
)
from metric import compute_macro_f05

def load_train_data(train_dir, sample_size=80000):
    print("Loading training data...")
    gt_df = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t")
    gt_map = defaultdict(set)
    for row in gt_df.iter_rows():
        s1_id = row[0]
        if row[1]:
            gt_map[s1_id] = set(row[1].split(","))
        else:
            gt_map[s1_id] = set()
    del gt_df
    
    s1_df = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t")
    if sample_size and sample_size < len(s1_df):
        s1_sample = s1_df.sample(n=sample_size, seed=42)
    else:
        s1_sample = s1_df
    del s1_df
    
    sample_s1_ids = set(s1_sample["entity_id"].to_list())
    needed_target_ids = set()
    for s1_id in sample_s1_ids:
        needed_target_ids.update(gt_map.get(s1_id, set()))
        
    s2_df = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t")
    
    s2_needed = s2_df.filter(pl.col("entity_id").is_in(needed_target_ids))
    s3_needed = s3_df.filter(pl.col("entity_id").is_in(needed_target_ids))
    # Include random distractors for realistic hard-negative mining
    s2_distract = s2_df.sample(n=min(len(s2_df), 300000), seed=42)
    s3_distract = s3_df.sample(n=min(len(s3_df), 300000), seed=42)
    del s2_df, s3_df
    
    target_df = pl.concat([s2_needed, s3_needed, s2_distract, s3_distract]).unique(subset=["entity_id"])
    print(f"Loaded {len(s1_sample):,} S1 entities and {len(target_df):,} target entities.")
    return s1_sample, target_df, gt_map

def train_model(train_dir, output_model_path, sample_size=80000):
    s1_sample, target_df, gt_map = load_train_data(train_dir, sample_size)
    
    print("Extracting features and building blocking index...")
    target_dict = {}
    idx_sorted_tokens = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_num_addr = defaultdict(list)
    
    for row in target_df.iter_rows():
        tid, name, addr = row[0], row[1], row[2] or ""
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
            idx_sorted_tokens[st].append(tid)
        f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
        if f2:
            idx_first2[f2].append(tid)
        if nums and addr_words:
            for num in nums[:2]:
                for word in addr_words[:2]:
                    idx_num_addr[(num, word)].append(tid)
                    
    del target_df
    gc.collect()
    
    # Filter non-discriminative keys (>50 items)
    idx_sorted_tokens = {k: v for k, v in idx_sorted_tokens.items() if len(v) <= 50}
    idx_first2 = {k: v for k, v in idx_first2.items() if len(v) <= 50}
    idx_num_addr = {k: v for k, v in idx_num_addr.items() if len(v) <= 50}
    
    print("Generating training feature pairs (positives + hard negatives)...")
    X = []
    y = []
    
    for row in s1_sample.iter_rows():
        s1_id, name, addr = row[0], row[1], row[2] or ""
        c_name = clean_and_translit(name)
        c_addr = clean_and_translit(addr)
        c_tokens = extract_core_tokens(c_name)
        nums, addr_words = extract_addr_features(c_addr)
        
        s1_info = {
            "clean_name": c_name,
            "clean_addr": c_addr,
            "core_str": " ".join(c_tokens),
            "first1": c_tokens[0] if c_tokens else "",
            "nums_set": set(nums)
        }
        
        true_tids = gt_map.get(s1_id, set())
        # Add positives
        for tid in true_tids:
            if tid in target_dict:
                feats = compute_pair_features(s1_info, target_dict[tid])
                X.append(feats)
                y.append(1)
                
        # Block for negatives
        cands = set()
        st = " ".join(sorted(c_tokens))
        if st in idx_sorted_tokens:
            cands.update(idx_sorted_tokens[st])
        f2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else s1_info["first1"]
        if f2 in idx_first2:
            cands.update(idx_first2[f2])
        if nums and addr_words:
            for num in nums[:2]:
                for word in addr_words[:2]:
                    k = (num, word)
                    if k in idx_num_addr:
                        cands.update(idx_num_addr[k])
                        
        negs = [tid for tid in cands if tid not in true_tids and tid in target_dict]
        # Keep up to 6 hard negatives per S1
        for tid in negs[:6]:
            feats = compute_pair_features(s1_info, target_dict[tid])
            X.append(feats)
            y.append(0)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Training dataset ready: {len(X):,} pairs ({np.sum(y):,} positives, {len(y)-np.sum(y):,} negatives).")
    
    print("Training LightGBM model...")
    dtrain = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "random_state": 42
    }
    
    booster = lgb.train(params, dtrain, num_boost_round=350)
    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    booster.save_model(output_model_path)
    print(f"Model saved to {output_model_path}.")
    
    # Feature importances
    imp = booster.feature_importance(importance_type="gain")
    print("\nFeature Importances (Gain):")
    for name, score in sorted(zip(FEATURE_NAMES, imp), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="student_resource/dataset/train")
    parser.add_argument("--output-model", type=str, default="models/lightgbm_model.txt")
    parser.add_argument("--sample-size", type=int, default=80000)
    args = parser.parse_args()
    train_model(args.train_dir, args.output_model, args.sample_size)
