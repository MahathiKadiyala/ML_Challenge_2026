"""
Systematic evaluation of Candidate Generation / Blocking Strategies.
Measures:
1. True Match Recall (what % of true matches are captured)
2. Candidates per S1 (efficiency and selectivity)
3. Total candidate volume
"""
import os
import polars as pl
import re
import unidecode
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

LEGAL_SUFFIXES = {
    "pvt ltd", "private limited", "ltd", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "pllc", "pc", "co", "company", "gmbh", 
    "sa", "sarl", "services", "enterprises", "consultants", "solutions",
    "technologies", "technology", "management", "trading", "group", "holdings",
    "holding", "associates", "international", "global", "industries", "industry",
    "pra li", "pvt", "limited private"
}

HONORIFICS = {"m/s", "ms", "shri", "smt", "sri", "dr", "mr", "mrs"}

def normalize_text(text):
    if not text:
        return ""
    # Transliterate to Latin
    text = unidecode.unidecode(text)
    # Lowercase
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_core_name(name):
    norm = normalize_text(name)
    tokens = norm.split()
    # Filter honorifics
    while tokens and tokens[0] in HONORIFICS:
        tokens.pop(0)
    # Filter trailing legal suffixes
    joined = " ".join(tokens)
    for suffix in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if joined.endswith(" " + suffix):
            joined = joined[:-len(suffix)].strip()
            tokens = joined.split()
            break
        elif joined == suffix:
            break
    core = " ".join(tokens)
    return core if core else norm

def extract_numbers(addr):
    if not addr:
        return []
    return re.findall(r"\b\d+\b", addr)

def extract_domain_stem(name):
    # If name looks like a domain name e.g. "holdingmetro.com"
    m = re.search(r"([a-z0-9]+)\.(com|org|net|in|co|us|fr|info|biz)\b", name.lower())
    if m:
        return m.group(1)
    return None

def test_blocking():
    print("Loading GT...")
    gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t")
    gt_non_empty = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    pairs = gt_non_empty.select([
        pl.col("source1_entity_id").alias("s1_id"),
        pl.col("matched_entity_ids").str.split(",").alias("mid")
    ]).explode("mid")
    
    # Take a validation sample of 10,000 S1 entities
    s1_val_ids = set(gt_non_empty.sample(n=10_000, seed=42)["source1_entity_id"].to_list())
    val_pairs = pairs.filter(pl.col("s1_id").is_in(s1_val_ids))
    total_val_pairs = len(val_pairs)
    print(f"Validation sample: {len(s1_val_ids)} S1 entities, {total_val_pairs} true pairs")
    
    val_true_matches = defaultdict(set)
    val_target_ids = set()
    for row in val_pairs.iter_rows():
        val_true_matches[row[0]].add(row[1])
        val_target_ids.add(row[1])
        
    print("Loading S1 validation records...")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t")
    s1_dict = {
        row[0]: {
            "name": row[1],
            "addr": row[2],
            "country": row[3],
            "core_name": extract_core_name(row[1]),
            "norm_addr": normalize_text(row[2]),
            "nums": extract_numbers(row[2])
        }
        for row in s1_df.filter(pl.col("entity_id").is_in(s1_val_ids)).iter_rows()
    }
    del s1_df
    
    print("Loading target records (sample of 500,000 targets including all true validation matches)...")
    # To test blocking realistically, target pool must contain ALL true matches + 500,000 distractors
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), separator="\t")
    s2_needed = s2_df.filter(pl.col("entity_id").is_in(val_target_ids))
    s2_distractors = s2_df.filter(~pl.col("entity_id").is_in(val_target_ids)).sample(n=250_000, seed=42)
    s2_pool = pl.concat([s2_needed, s2_distractors])
    del s2_df, s2_needed, s2_distractors
    
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), separator="\t")
    s3_needed = s3_df.filter(pl.col("entity_id").is_in(val_target_ids))
    s3_distractors = s3_df.filter(~pl.col("entity_id").is_in(val_target_ids)).sample(n=250_000, seed=42)
    s3_pool = pl.concat([s3_needed, s3_distractors])
    del s3_df, s3_needed, s3_distractors
    
    target_pool = pl.concat([s2_pool, s3_pool])
    print(f"Total target pool size: {len(target_pool):,}")
    
    target_records = {}
    for row in target_pool.iter_rows():
        tid = row[0]
        name = row[1]
        addr = row[2] or ""
        ctry = row[3]
        target_records[tid] = {
            "name": name,
            "addr": addr,
            "country": ctry,
            "core_name": extract_core_name(name),
            "norm_addr": normalize_text(addr),
            "nums": extract_numbers(addr),
            "domain_stem": extract_domain_stem(name)
        }
    del target_pool
    
    # Now let's test different blocking keys:
    # 1. Exact core_name + country
    # 2. First 2 tokens of core_name + country
    # 3. Sorted core_name tokens + country
    # 4. Exact normalized address + country
    # 5. Core name prefix (4 chars) + address number + country
    # 6. Domain stem match
    
    print("\nIndexing target pool...")
    index_exact_core = defaultdict(list)
    index_first2_tokens = defaultdict(list)
    index_exact_addr = defaultdict(list)
    index_prefix_num = defaultdict(list)
    index_domain = defaultdict(list)
    
    for tid, rec in target_records.items():
        c = rec["country"]
        cn = rec["core_name"]
        tokens = cn.split()
        
        # Key 1: exact core name
        if cn:
            index_exact_core[(c, cn)].append(tid)
            
        # Key 2: first 2 tokens of core name
        if len(tokens) >= 2:
            index_first2_tokens[(c, " ".join(tokens[:2]))].append(tid)
        elif len(tokens) == 1:
            index_first2_tokens[(c, tokens[0])].append(tid)
            
        # Key 3: normalized address
        na = rec["norm_addr"]
        if na:
            index_exact_addr[(c, na)].append(tid)
            
        # Key 4: prefix 3 chars + address number
        if len(cn) >= 3 and rec["nums"]:
            pref = cn[:3]
            for num in rec["nums"][:2]:
                index_prefix_num[(c, pref, num)].append(tid)
                
        # Key 5: domain stem
        ds = rec["domain_stem"]
        if ds:
            index_domain[(c, ds)].append(tid)

    print("Evaluating candidate generation for 10,000 S1 entities...")
    
    captured_by_key = Counter()
    total_candidates_generated = 0
    all_captured = set() # (s1_id, tid)
    
    candidates_per_s1 = []
    
    for s1_id, s1_rec in s1_dict.items():
        c = s1_rec["country"]
        cn = s1_rec["core_name"]
        tokens = cn.split()
        na = s1_rec["norm_addr"]
        nums = s1_rec["nums"]
        
        s1_candidates = set()
        
        # 1. Exact core name
        if cn and (c, cn) in index_exact_core:
            for tid in index_exact_core[(c, cn)]:
                s1_candidates.add(tid)
                if tid in val_true_matches[s1_id]:
                    captured_by_key["Exact Core Name"] += 1
                    
        # 2. First 2 tokens
        k2 = " ".join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else "")
        if k2 and (c, k2) in index_first2_tokens:
            for tid in index_first2_tokens[(c, k2)]:
                s1_candidates.add(tid)
                if tid in val_true_matches[s1_id]:
                    captured_by_key["First 2 Tokens"] += 1
                    
        # 3. Exact address
        if na and (c, na) in index_exact_addr:
            for tid in index_exact_addr[(c, na)]:
                s1_candidates.add(tid)
                if tid in val_true_matches[s1_id]:
                    captured_by_key["Exact Address"] += 1
                    
        # 4. Prefix 3 + number
        if len(cn) >= 3 and nums:
            pref = cn[:3]
            for num in nums[:2]:
                if (c, pref, num) in index_prefix_num:
                    for tid in index_prefix_num[(c, pref, num)]:
                        s1_candidates.add(tid)
                        if tid in val_true_matches[s1_id]:
                            captured_by_key["Prefix 3 + Num"] += 1
                            
        # 5. Domain match: does domain stem equal concatenated tokens or first token?
        if tokens:
            concat_name = "".join(tokens)
            if (c, concat_name) in index_domain:
                for tid in index_domain[(c, concat_name)]:
                    s1_candidates.add(tid)
                    if tid in val_true_matches[s1_id]:
                        captured_by_key["Domain Stem"] += 1
            if (c, tokens[0]) in index_domain:
                for tid in index_domain[(c, tokens[0])]:
                    s1_candidates.add(tid)
                    if tid in val_true_matches[s1_id]:
                        captured_by_key["Domain First Token"] += 1

        candidates_per_s1.append(len(s1_candidates))
        for tid in s1_candidates:
            if tid in val_true_matches[s1_id]:
                all_captured.add((s1_id, tid))

    total_recall = len(all_captured) / total_val_pairs
    avg_cands = sum(candidates_per_s1) / len(candidates_per_s1)
    max_cands = max(candidates_per_s1)
    
    print("\n=== BLOCKING EVALUATION RESULTS ===")
    print(f"Total True Pairs: {total_val_pairs:,}")
    print(f"Total True Pairs Captured: {len(all_captured):,} ({total_recall * 100:.2f}% RECALL)")
    print(f"Average candidates per S1: {avg_cands:.1f}")
    print(f"Max candidates for any single S1: {max_cands}")
    print("\nTrue match captures by key type:")
    for k, cnt in captured_by_key.most_common():
        print(f"  {k}: {cnt:,}")

if __name__ == "__main__":
    test_blocking()
