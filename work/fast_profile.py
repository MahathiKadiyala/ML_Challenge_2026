"""
High-performance profiling script using Polars.
Reads all files with multi-threading and extracts complete dataset statistics.
"""
import os
import sys
import polars as pl
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

def profile_ground_truth():
    gt_path = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
    print(f"==================================================")
    print(f"ANALYZING GROUND TRUTH: {gt_path}")
    print(f"==================================================")
    
    # Read with polars
    df = pl.read_csv(gt_path, separator="\t", truncate_ragged_lines=True)
    print("Schema:", df.schema)
    total_s1 = len(df)
    
    # Analyze matched_entity_ids
    # null or empty string
    null_count = df["matched_entity_ids"].null_count()
    empty_str_count = (df["matched_entity_ids"] == "").sum()
    singletons = null_count + empty_str_count
    
    # Non-empty
    non_empty = df.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    
    # Calculate match count per S1
    # Split by comma
    match_lists = non_empty["matched_entity_ids"].str.split(",")
    match_lengths = match_lists.list.len()
    
    all_lengths = [0] * singletons + match_lengths.to_list()
    len_counter = Counter(all_lengths)
    
    total_matches = match_lengths.sum()
    
    # Let's count S2 vs S3
    exploded = non_empty.select(pl.col("matched_entity_ids").str.split(",").alias("mid")).explode("mid")
    s2_count = exploded.filter(pl.col("mid").str.starts_with("S2-")).height
    s3_count = exploded.filter(pl.col("mid").str.starts_with("S3-")).height
    other_count = exploded.height - (s2_count + s3_count)
    
    print(f"Total S1 entities: {total_s1:,}")
    print(f"Singletons (0 matches): {singletons:,} ({singletons / total_s1 * 100:.2f}%)")
    print(f"Entities with >=1 match: {len(non_empty):,} ({len(non_empty) / total_s1 * 100:.2f}%)")
    print(f"Total matched IDs: {total_matches:,} (Average: {total_matches / total_s1:.3f} per S1)")
    print(f"S2 matched IDs: {s2_count:,} ({s2_count / total_matches * 100:.2f}%)")
    print(f"S3 matched IDs: {s3_count:,} ({s3_count / total_matches * 100:.2f}%)")
    print(f"Other prefix matched IDs: {other_count}")
    
    print("\nMatch count per S1 distribution:")
    for c in sorted(len_counter.keys())[:15]:
        print(f"  {c} matches: {len_counter[c]:,} ({len_counter[c] / total_s1 * 100:.2f}%)")
    if max(len_counter.keys()) >= 15:
        print(f"  Max matches for a single S1: {max(len_counter.keys())}")

def profile_source(path, label):
    print(f"\n==================================================")
    print(f"ANALYZING: {label} ({os.path.basename(path)})")
    print(f"==================================================")
    
    df = pl.read_csv(path, separator="\t", truncate_ragged_lines=True, ignore_errors=True)
    n = len(df)
    print(f"Total rows: {n:,}")
    print("Columns:", df.columns)
    
    # Unique IDs
    n_unique_ids = df["entity_id"].n_unique()
    print(f"Unique entity_ids: {n_unique_ids:,} (Duplicates: {n - n_unique_ids:,})")
    
    # Nulls / empty
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            nulls = df[col].null_count()
            empties = (df[col] == "").sum()
            missing = nulls + empties
            print(f"Missing {col}: {missing:,} ({missing / n * 100:.2f}%)")
    
    # Country distribution
    if "country" in df.columns:
        print("Country distribution:")
        cdist = df["country"].value_counts(sort=True)
        for row in cdist.iter_rows():
            cname, cnt = row[0], row[1]
            print(f"  {cname}: {cnt:,} ({cnt / n * 100:.2f}%)")
            
    # Text length stats
    if "business_name" in df.columns:
        name_lens = df["business_name"].fill_null("").str.len_chars()
        print(f"Business Name length: mean={name_lens.mean():.1f}, min={name_lens.min()}, max={name_lens.max()}, median={name_lens.median()}")
    if "business_address" in df.columns:
        addr_lens = df["business_address"].fill_null("").str.len_chars()
        print(f"Business Address length: mean={addr_lens.mean():.1f}, min={addr_lens.min()}, max={addr_lens.max()}, median={addr_lens.median()}")

    # Script analysis via regex
    # Common scripts in South Asia, Latin, etc.
    if "business_name" in df.columns:
        devanagari = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u0900-\u097F]")).height
        bengali = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u0980-\u09FF]")).height
        tamil = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u0B80-\u0BFF]")).height
        telugu = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u0C00-\u0C7F]")).height
        gujarati = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u0A80-\u0AFF]")).height
        kannada = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u0C80-\u0CFF]")).height
        malayalam = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u0D00-\u0D7F]")).height
        gurmukhi = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u0A00-\u0A7F]")).height
        arabic = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u0600-\u06FF]")).height
        cyrillic = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u0400-\u04FF]")).height
        cjk = df.filter(pl.col("business_name").fill_null("").str.contains(r"[\u4E00-\u9FFF]")).height
        latin = df.filter(pl.col("business_name").fill_null("").str.contains(r"[a-zA-Z]")).height
        
        print("Script occurrence in business_name (records containing script):")
        scripts = [
            ("Latin", latin), ("Devanagari", devanagari), ("Bengali", bengali),
            ("Tamil", tamil), ("Telugu", telugu), ("Gujarati", gujarati),
            ("Kannada", kannada), ("Malayalam", malayalam), ("Gurmukhi", gurmukhi),
            ("Arabic", arabic), ("Cyrillic", cyrillic), ("CJK", cjk)
        ]
        for sname, scnt in scripts:
            if scnt > 0:
                print(f"  {sname}: {scnt:,} ({scnt / n * 100:.2f}%)")

if __name__ == "__main__":
    profile_ground_truth()
    
    profile_source(os.path.join(DATA_DIR, "train", "train_source1.tsv"), "TRAIN SOURCE 1")
    profile_source(os.path.join(DATA_DIR, "train", "train_source2.tsv"), "TRAIN SOURCE 2")
    profile_source(os.path.join(DATA_DIR, "train", "train_source3.tsv"), "TRAIN SOURCE 3")
    
    profile_source(os.path.join(DATA_DIR, "test", "test_source1.tsv"), "TEST SOURCE 1")
    profile_source(os.path.join(DATA_DIR, "test", "test_source2.tsv"), "TEST SOURCE 2")
    profile_source(os.path.join(DATA_DIR, "test", "test_source3.tsv"), "TEST SOURCE 3")
