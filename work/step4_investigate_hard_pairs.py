"""
Investigate the 19% of true matches that had Neither High under raw token_set_ratio.
"""
import os
import polars as pl
import re
from rapidfuzz import fuzz
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

def run():
    gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t")
    gt_non_empty = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    pairs = gt_non_empty.select([
        pl.col("source1_entity_id").alias("s1_id"),
        pl.col("matched_entity_ids").str.split(",").alias("mid")
    ]).explode("mid")
    
    sample_pairs = pairs.sample(n=50_000, seed=42)
    sample_s1_ids = set(sample_pairs["s1_id"].to_list())
    sample_mids = set(sample_pairs["mid"].to_list())
    
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t")
    s1_dict = {
        row[0]: (row[1], row[2], row[3]) 
        for row in s1_df.filter(pl.col("entity_id").is_in(sample_s1_ids)).iter_rows()
    }
    del s1_df
    
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
    
    indic_regex = re.compile(r"[\u0900-\u0D7F]")
    
    hard_examples = []
    category_counts = Counter()
    
    for row in sample_pairs.iter_rows():
        s1_id, mid = row[0], row[1]
        if s1_id not in s1_dict or mid not in target_dict:
            continue
        s1_n, s1_a, s1_c = s1_dict[s1_id]
        tg_n, tg_a, tg_c = target_dict[mid]
        
        nr = fuzz.token_set_ratio(s1_n, tg_n)
        ar = fuzz.token_set_ratio(s1_a, tg_a)
        
        if nr < 70 and ar < 70:
            if indic_regex.search(tg_n):
                category_counts["Indic Script Name"] += 1
            elif indic_regex.search(tg_a):
                category_counts["Indic Script Addr"] += 1
            elif ".com" in tg_n.lower() or ".org" in tg_n.lower() or ".net" in tg_n.lower():
                category_counts["Domain Name"] += 1
            elif not tg_a.strip():
                category_counts["Target Addr Missing"] += 1
            else:
                category_counts["Other Hard"] += 1
                if len(hard_examples) < 25:
                    hard_examples.append((s1_n, tg_n, s1_a, tg_a, nr, ar, s1_c))
                    
    print("\nBreakdown of pairs with Name < 70 AND Addr < 70:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count:,} ({count / sum(category_counts.values()) * 100:.2f}%)")
        
    print("\nSample of 'Other Hard' pairs:")
    with open("work/hard_samples.txt", "w", encoding="utf-8") as out:
        for s1_n, tg_n, s1_a, tg_a, nr, ar, c in hard_examples:
            out.write(f"Country: {c} | Name Ratio: {nr:.1f} | Addr Ratio: {ar:.1f}\n")
            out.write(f"  S1:  Name='{s1_n}'\n       Addr='{s1_a}'\n")
            out.write(f"  Tgt: Name='{tg_n}'\n       Addr='{tg_a}'\n")
            out.write("-" * 60 + "\n")
    print("Wrote hard samples to work/hard_samples.txt")

if __name__ == "__main__":
    run()
