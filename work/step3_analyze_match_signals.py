"""
Analyze match signals on true matches vs negative pairs:
- Exact name match (normalized)
- Exact address match (normalized)
- Address number overlap
- High name similarity
- High address similarity
- What fraction of true matches have:
    (high name similarity) OR (high address similarity) OR (both)?
"""
import os
import polars as pl
import re
from rapidfuzz import fuzz
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

def clean_str(s):
    if not s:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()

def extract_numbers(s):
    if not s:
        return set()
    return set(re.findall(r"\b\d+\b", s))

def run():
    print("Loading GT...")
    gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t")
    gt_non_empty = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    pairs = gt_non_empty.select([
        pl.col("source1_entity_id").alias("s1_id"),
        pl.col("matched_entity_ids").str.split(",").alias("mid")
    ]).explode("mid")
    
    sample_pairs = pairs.sample(n=100_000, seed=42)
    sample_s1_ids = set(sample_pairs["s1_id"].to_list())
    sample_mids = set(sample_pairs["mid"].to_list())
    
    print("Loading S1...")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t")
    s1_dict = {
        row[0]: (row[1], row[2], row[3]) 
        for row in s1_df.filter(pl.col("entity_id").is_in(sample_s1_ids)).iter_rows()
    }
    del s1_df
    
    print("Loading S2 and S3...")
    s2_mids = {m for m in sample_mids if m.startswith("S2-")}
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), separator="\t")
    target_dict = {
        row[0]: (row[1], row[2] or "", row[3]) 
        for row in s2_df.filter(pl.col("entity_id").is_in(s2_mids)).iter_rows()
    }
    del s2_df
    
    s3_mids = {m for m in sample_mids if m.startswith("S3-")}
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), separator="\t")
    for row in s3_df.filter(pl.col("entity_id").is_in(s3_mids)).iter_rows():
        target_dict[row[0]] = (row[1], row[2] or "", row[3])
    del s3_df
    
    print("Evaluating signals on 100,000 true matches...")
    n_total = len(sample_pairs)
    
    high_name_only = 0
    high_addr_only = 0
    high_both = 0
    neither_high = 0
    
    addr_num_overlap_cnt = 0
    has_addr_nums = 0
    
    name_ratios = []
    addr_ratios = []
    
    for row in sample_pairs.iter_rows():
        s1_id, mid = row[0], row[1]
        if s1_id not in s1_dict or mid not in target_dict:
            continue
        s1_n, s1_a, _ = s1_dict[s1_id]
        tg_n, tg_a, _ = target_dict[mid]
        
        nr = fuzz.token_set_ratio(s1_n, tg_n)
        ar = fuzz.token_set_ratio(s1_a, tg_a)
        name_ratios.append(nr)
        addr_ratios.append(ar)
        
        s1_nums = extract_numbers(s1_a)
        tg_nums = extract_numbers(tg_a)
        if s1_nums and tg_nums:
            has_addr_nums += 1
            if s1_nums & tg_nums:
                addr_num_overlap_cnt += 1
                
        is_name_high = (nr >= 70)
        is_addr_high = (ar >= 70)
        
        if is_name_high and is_addr_high:
            high_both += 1
        elif is_name_high:
            high_name_only += 1
        elif is_addr_high:
            high_addr_only += 1
        else:
            neither_high += 1

    print(f"\n--- TRUE MATCH SIGNAL COVERAGE (threshold = 70 token_set_ratio) ---")
    print(f"High Both (Name >= 70 AND Addr >= 70): {high_both:,} ({high_both / n_total * 100:.2f}%)")
    print(f"High Name Only (Name >= 70): {high_name_only:,} ({high_name_only / n_total * 100:.2f}%)")
    print(f"High Addr Only (Addr >= 70): {high_addr_only:,} ({high_addr_only / n_total * 100:.2f}%)")
    print(f"Union (Name >= 70 OR Addr >= 70): {high_both + high_name_only + high_addr_only:,} ({(high_both + high_name_only + high_addr_only) / n_total * 100:.2f}%)")
    print(f"Neither High (< 70 for both): {neither_high:,} ({neither_high / n_total * 100:.2f}%)")
    
    if has_addr_nums > 0:
        print(f"Address number overlap among pairs with numbers: {addr_num_overlap_cnt:,} / {has_addr_nums:,} ({addr_num_overlap_cnt / has_addr_nums * 100:.2f}%)")

if __name__ == "__main__":
    run()
