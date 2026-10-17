"""
Profiling script for Amazon ML Challenge 2026 Business Entity Resolution.
Processes complete files without loading everything into memory simultaneously.
"""
import os
import sys
import csv
import unicodedata
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

def analyze_ground_truth():
    gt_path = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
    print(f"=== ANALYZING GROUND TRUTH: {gt_path} ===")
    
    total_s1 = 0
    empty_matches = 0
    non_empty_matches = 0
    total_matched_ids = 0
    s2_matches = 0
    s3_matches = 0
    match_count_dist = Counter()
    s2_count_dist = Counter()
    s3_count_dist = Counter()
    
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        print("Header:", header)
        
        for row in reader:
            if not row:
                continue
            total_s1 += 1
            s1_id = row[0]
            matched_str = row[1] if len(row) > 1 else ""
            if not matched_str.strip():
                empty_matches += 1
                match_count_dist[0] += 1
                s2_count_dist[0] += 1
                s3_count_dist[0] += 1
            else:
                mids = [m.strip() for m in matched_str.split(",") if m.strip()]
                non_empty_matches += 1
                n_m = len(mids)
                total_matched_ids += n_m
                match_count_dist[n_m] += 1
                
                n_s2 = sum(1 for m in mids if m.startswith("S2-"))
                n_s3 = sum(1 for m in mids if m.startswith("S3-"))
                s2_matches += n_s2
                s3_matches += n_s3
                s2_count_dist[n_s2] += 1
                s3_count_dist[n_s3] += 1

    print(f"Total S1 entities in train GT: {total_s1:,}")
    print(f"Singletons (0 matches): {empty_matches:,} ({empty_matches / total_s1 * 100:.2f}%)")
    print(f"Entities with >=1 match: {non_empty_matches:,} ({non_empty_matches / total_s1 * 100:.2f}%)")
    print(f"Total matched IDs: {total_matched_ids:,} (Avg per S1: {total_matched_ids / total_s1:.3f})")
    print(f"Total S2 matched IDs: {s2_matches:,}")
    print(f"Total S3 matched IDs: {s3_matches:,}")
    print("\nMatch count distribution (top 10):")
    for count, freq in match_count_dist.most_common(10):
        print(f"  {count} matches: {freq:,} entities ({freq / total_s1 * 100:.2f}%)")
    print("\nS2 match count distribution (top 5):")
    for count, freq in s2_count_dist.most_common(5):
        print(f"  {count} S2 matches: {freq:,} entities ({freq / total_s1 * 100:.2f}%)")
    print("\nS3 match count distribution (top 5):")
    for count, freq in s3_count_dist.most_common(5):
        print(f"  {count} S3 matches: {freq:,} entities ({freq / total_s1 * 100:.2f}%)")

def detect_script(text):
    if not text:
        return "Empty"
    scripts = Counter()
    for char in text:
        if char.isalpha():
            name = unicodedata.name(char, "")
            if "LATIN" in name:
                scripts["Latin"] += 1
            elif "DEVANAGARI" in name:
                scripts["Devanagari"] += 1
            elif "BENGALI" in name:
                scripts["Bengali"] += 1
            elif "TAMIL" in name:
                scripts["Tamil"] += 1
            elif "TELUGU" in name:
                scripts["Telugu"] += 1
            elif "GUJARATI" in name:
                scripts["Gujarati"] += 1
            elif "KANNADA" in name:
                scripts["Kannada"] += 1
            elif "MALAYALAM" in name:
                scripts["Malayalam"] += 1
            elif "GURMUKHI" in name:
                scripts["Gurmukhi"] += 1
            elif "ARABIC" in name:
                scripts["Arabic"] += 1
            elif "CYRILLIC" in name:
                scripts["Cyrillic"] += 1
            elif "CJK" in name:
                scripts["CJK"] += 1
            else:
                scripts["Other"] += 1
    if not scripts:
        return "Non-Alpha"
    top_script, top_cnt = scripts.most_common(1)[0]
    if len(scripts) > 1 and (top_cnt / sum(scripts.values()) < 0.9):
        return f"Mixed ({top_script}+)"
    return top_script

def analyze_source_file(file_path, name):
    print(f"\n=== ANALYZING: {name} ({file_path}) ===")
    total_rows = 0
    missing_name = 0
    missing_addr = 0
    missing_country = 0
    countries = Counter()
    name_scripts = Counter()
    addr_scripts = Counter()
    id_prefixes = Counter()
    
    total_name_len = 0
    total_addr_len = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        print("Header:", header)
        
        for row in reader:
            if not row:
                continue
            total_rows += 1
            eid = row[0]
            bname = row[1] if len(row) > 1 else ""
            baddr = row[2] if len(row) > 2 else ""
            country = row[3] if len(row) > 3 else ""
            
            id_prefixes[eid[:3]] += 1
            if not bname.strip():
                missing_name += 1
            else:
                total_name_len += len(bname)
                name_scripts[detect_script(bname)] += 1
                
            if not baddr.strip():
                missing_addr += 1
            else:
                total_addr_len += len(baddr)
                addr_scripts[detect_script(baddr)] += 1
                
            if not country.strip():
                missing_country += 1
            else:
                countries[country.strip()] += 1
                
    print(f"Total rows: {total_rows:,}")
    print(f"ID prefixes: {dict(id_prefixes)}")
    print(f"Missing Name: {missing_name:,} ({missing_name / total_rows * 100:.2f}%)")
    print(f"Missing Address: {missing_addr:,} ({missing_addr / total_rows * 100:.2f}%)")
    print(f"Missing Country: {missing_country:,} ({missing_country / total_rows * 100:.2f}%)")
    print(f"Avg Name Length: {total_name_len / max(1, total_rows - missing_name):.1f}")
    print(f"Avg Addr Length: {total_addr_len / max(1, total_rows - missing_addr):.1f}")
    print("Country distribution:")
    for c, cnt in countries.most_common():
        print(f"  {c}: {cnt:,} ({cnt / total_rows * 100:.2f}%)")
    print("Name script distribution (top 5):")
    for s, cnt in name_scripts.most_common(5):
        print(f"  {s}: {cnt:,} ({cnt / total_rows * 100:.2f}%)")
    print("Addr script distribution (top 5):")
    for s, cnt in addr_scripts.most_common(5):
        print(f"  {s}: {cnt:,} ({cnt / total_rows * 100:.2f}%)")

if __name__ == "__main__":
    analyze_ground_truth()
    
    analyze_source_file(os.path.join(DATA_DIR, "train", "train_source1.tsv"), "Train Source 1")
    analyze_source_file(os.path.join(DATA_DIR, "train", "train_source2.tsv"), "Train Source 2")
    analyze_source_file(os.path.join(DATA_DIR, "train", "train_source3.tsv"), "Train Source 3")
    
    analyze_source_file(os.path.join(DATA_DIR, "test", "test_source1.tsv"), "Test Source 1")
    analyze_source_file(os.path.join(DATA_DIR, "test", "test_source2.tsv"), "Test Source 2")
    analyze_source_file(os.path.join(DATA_DIR, "test", "test_source3.tsv"), "Test Source 3")
