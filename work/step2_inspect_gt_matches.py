"""
Deep inspection of true matching pairs from training ground truth.
Investigates:
1. Country consistency between matched pairs.
2. Cardinality: Does an S2/S3 entity match multiple S1 entities?
3. Cross-script matches (S1 Latin vs S2/S3 Indic scripts).
4. Name and Address similarity distributions among true matches.
5. Exact vs fuzzy matches.
"""
import os
import polars as pl
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

def run():
    print("Loading datasets...")
    # Load S1, S2, S3 sample / chunks or join
    # Since we need to join on ground truth, let's load GT first
    gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t")
    
    # Filter non-empty
    gt_non_empty = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    print(f"Total non-empty S1 in GT: {len(gt_non_empty):,}")
    
    # Explode to pairs (s1_id, match_id)
    pairs = gt_non_empty.select([
        pl.col("source1_entity_id").alias("s1_id"),
        pl.col("matched_entity_ids").str.split(",").alias("mid")
    ]).explode("mid")
    print(f"Total true match pairs: {len(pairs):,}")
    
    # Check cardinality: how many S1s does an S2 or S3 entity match?
    mid_counts = pairs["mid"].value_counts()
    print("Distinct matched target entities (S2/S3):", len(mid_counts))
    max_s1_per_mid = mid_counts["count"].max()
    mean_s1_per_mid = mid_counts["count"].mean()
    multi_s1_mids = mid_counts.filter(pl.col("count") > 1).height
    print(f"Max S1 per S2/S3 entity: {max_s1_per_mid}")
    print(f"Mean S1 per S2/S3 entity: {mean_s1_per_mid:.4f}")
    print(f"Target entities matching >1 S1: {multi_s1_mids:,} ({multi_s1_mids / len(mid_counts) * 100:.3f}%)")
    
    # Now let's inspect country consistency and cross-script matches on a large sample of 200,000 true pairs
    sample_pairs = pairs.sample(n=200_000, seed=42)
    sample_s1_ids = set(sample_pairs["s1_id"].to_list())
    sample_mids = set(sample_pairs["mid"].to_list())
    
    print("Loading S1...")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t")
    s1_dict = {
        row[0]: (row[1], row[2], row[3]) 
        for row in s1_df.filter(pl.col("entity_id").is_in(sample_s1_ids)).iter_rows()
    }
    del s1_df
    
    print("Loading S2...")
    s2_mids = {m for m in sample_mids if m.startswith("S2-")}
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), separator="\t")
    target_dict = {
        row[0]: (row[1], row[2] or "", row[3]) 
        for row in s2_df.filter(pl.col("entity_id").is_in(s2_mids)).iter_rows()
    }
    del s2_df
    
    print("Loading S3...")
    s3_mids = {m for m in sample_mids if m.startswith("S3-")}
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), separator="\t")
    for row in s3_df.filter(pl.col("entity_id").is_in(s3_mids)).iter_rows():
        target_dict[row[0]] = (row[1], row[2] or "", row[3])
    del s3_df
    
    print("Analyzing 200,000 true match pairs...")
    country_mismatches = 0
    exact_name_matches = 0
    exact_name_case_insensitive = 0
    non_latin_targets = 0
    
    import re
    indic_regex = re.compile(r"[\u0900-\u0D7F]")
    
    cross_script_examples = []
    hard_name_examples = []
    
    for row in sample_pairs.iter_rows():
        s1_id, mid = row[0], row[1]
        if s1_id not in s1_dict or mid not in target_dict:
            continue
        s1_name, s1_addr, s1_country = s1_dict[s1_id]
        tgt_name, tgt_addr, tgt_country = target_dict[mid]
        
        if s1_country != tgt_country:
            country_mismatches += 1
            
        if s1_name == tgt_name:
            exact_name_matches += 1
        if s1_name.lower().strip() == tgt_name.lower().strip():
            exact_name_case_insensitive += 1
            
        if indic_regex.search(tgt_name):
            non_latin_targets += 1
            if len(cross_script_examples) < 10:
                cross_script_examples.append((s1_name, tgt_name, s1_addr, tgt_addr, s1_country))
        else:
            # Check if Latin names are very different
            from rapidfuzz import fuzz
            ratio = fuzz.ratio(s1_name.lower(), tgt_name.lower())
            if ratio < 60 and len(hard_name_examples) < 10:
                hard_name_examples.append((s1_name, tgt_name, s1_addr, tgt_addr, ratio))

    print(f"\nCountry mismatch rate in true matches: {country_mismatches} / {len(sample_pairs)} ({country_mismatches / len(sample_pairs) * 100:.4f}%)")
    print(f"Exact name match (case-sensitive): {exact_name_matches:,} ({exact_name_matches / len(sample_pairs) * 100:.2f}%)")
    print(f"Exact name match (case-insensitive): {exact_name_case_insensitive:,} ({exact_name_case_insensitive / len(sample_pairs) * 100:.2f}%)")
    print(f"Cross-script true matches (S1 Latin vs S2/S3 Indic): {non_latin_targets:,} ({non_latin_targets / len(sample_pairs) * 100:.2f}%)")
    
    print("\n--- Cross-script True Match Examples ---")
    for s1_n, tg_n, s1_a, tg_a, ctry in cross_script_examples:
        print(f"S1 Name: '{s1_n}' | Tgt Name: '{tg_n}'")
        print(f"S1 Addr: '{s1_a}' | Tgt Addr: '{tg_a}' | Country: {ctry}\n")
        
    print("\n--- Low String-Similarity True Match Examples (Latin-Latin) ---")
    for s1_n, tg_n, s1_a, tg_a, r in hard_name_examples:
        print(f"Ratio: {r:.1f} | S1: '{s1_n}' <--> Tgt: '{tg_n}'")
        print(f"  S1 Addr: '{s1_a}'\n  Tgt Addr: '{tg_a}'\n")

if __name__ == "__main__":
    run()
