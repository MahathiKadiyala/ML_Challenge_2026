"""
Inspect match examples with UTF-8 safe printing.
"""
import os
import polars as pl
import re
from rapidfuzz import fuzz

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
    
    print("\n=== CROSS-SCRIPT MATCH EXAMPLES (S1 Latin vs Target Indic) ===")
    cross_shown = 0
    with open("work/cross_script_samples.txt", "w", encoding="utf-8") as out_f:
        for row in sample_pairs.iter_rows():
            s1_id, mid = row[0], row[1]
            if s1_id not in s1_dict or mid not in target_dict:
                continue
            s1_name, s1_addr, s1_country = s1_dict[s1_id]
            tgt_name, tgt_addr, tgt_country = target_dict[mid]
            
            if indic_regex.search(tgt_name):
                out_f.write(f"S1 ({s1_id}): Name='{s1_name}' | Addr='{s1_addr}'\n")
                out_f.write(f"Tgt ({mid}): Name='{tgt_name}' | Addr='{tgt_addr}'\n")
                out_f.write("-" * 60 + "\n")
                cross_shown += 1
                if cross_shown >= 20:
                    break
                    
    print(f"Wrote {cross_shown} cross-script samples to work/cross_script_samples.txt")
    
    print("\n=== LATIN-LATIN LOW RATIO EXAMPLES ===")
    low_shown = 0
    with open("work/latin_low_ratio_samples.txt", "w", encoding="utf-8") as out_f:
        for row in sample_pairs.iter_rows():
            s1_id, mid = row[0], row[1]
            if s1_id not in s1_dict or mid not in target_dict:
                continue
            s1_name, s1_addr, s1_country = s1_dict[s1_id]
            tgt_name, tgt_addr, tgt_country = target_dict[mid]
            
            if not indic_regex.search(tgt_name):
                r = fuzz.ratio(s1_name.lower(), tgt_name.lower())
                ar = fuzz.ratio(s1_addr.lower(), tgt_addr.lower())
                if r < 60:
                    out_f.write(f"Name Ratio: {r:.1f} | Addr Ratio: {ar:.1f} | Country: {s1_country}\n")
                    out_f.write(f"S1 ({s1_id}): Name='{s1_name}' | Addr='{s1_addr}'\n")
                    out_f.write(f"Tgt ({mid}): Name='{tgt_name}' | Addr='{tgt_addr}'\n")
                    out_f.write("-" * 60 + "\n")
                    low_shown += 1
                    if low_shown >= 20:
                        break
    print(f"Wrote {low_shown} low-ratio samples to work/latin_low_ratio_samples.txt")

if __name__ == "__main__":
    run()
