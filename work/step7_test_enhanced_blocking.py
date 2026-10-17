"""
Test Enhanced Multi-Pass Blocking with:
1. Legal stopwords removed anywhere in name.
2. Sorted content tokens (handles word order inversion).
3. Address keys: (country, primary_number, address_token).
4. Unidecode consonant deduplication.
5. Domain sub-token matching.
"""
import os
import polars as pl
import re
import unidecode
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")

LEGAL_STOPWORDS = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "pllc", "pc", "co", "company", "gmbh", 
    "sa", "sarl", "services", "enterprises", "consultants", "solutions",
    "technologies", "technology", "management", "trading", "group", "holdings",
    "holding", "associates", "international", "global", "industries", "industry",
    "pra", "li", "m/s", "ms", "shri", "smt", "sri", "dr", "mr", "mrs", "the", "and", "of"
}

GENERIC_ADDR = {
    "road", "rd", "street", "st", "lane", "ln", "avenue", "ave", "court", "ct",
    "drive", "dr", "boulevard", "blvd", "floor", "fl", "building", "bldg", "near",
    "opp", "opposite", "door", "no", "plot", "shop", "flat", "block", "c/o", "h/no",
    "hno", "house", "room", "sector", "phase", "nagar", "colony", "city", "west",
    "east", "north", "south", "in", "at", "by", "unit", "suite", "ste", "apt", "apartment"
}

def clean_and_translit(text):
    if not text:
        return ""
    text = unidecode.unidecode(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    # Deduplicate repeated consonants from unidecode transliteration (e.g. tt -> t, nn -> n)
    text = re.sub(r"([a-z])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_core_tokens(name):
    norm = clean_and_translit(name)
    tokens = norm.split()
    # Remove domain extension if present
    if tokens and tokens[-1] in {"com", "org", "net", "in", "co", "us", "fr"}:
        tokens.pop()
    content_tokens = [t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1]
    return content_tokens

def extract_addr_features(addr):
    norm = clean_and_translit(addr)
    tokens = norm.split()
    nums = [t for t in tokens if t.isdigit() and len(t) <= 6]
    addr_words = [t for t in tokens if not t.isdigit() and t not in GENERIC_ADDR and len(t) > 2]
    return nums, addr_words

def extract_domain_stem(name):
    m = re.search(r"([a-z0-9]+)\.(com|org|net|in|co|us|fr|info|biz)\b", name.lower())
    if m:
        stem = m.group(1)
        # Deduplicate
        return re.sub(r"([a-z])\1+", r"\1", stem)
    return None

def run():
    print("Loading GT...")
    gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t")
    gt_non_empty = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    pairs = gt_non_empty.select([
        pl.col("source1_entity_id").alias("s1_id"),
        pl.col("matched_entity_ids").str.split(",").alias("mid")
    ]).explode("mid")
    
    s1_val_ids = set(gt_non_empty.sample(n=10_000, seed=42)["source1_entity_id"].to_list())
    val_pairs = pairs.filter(pl.col("s1_id").is_in(s1_val_ids))
    total_val_pairs = len(val_pairs)
    print(f"Validation sample: {len(s1_val_ids)} S1 entities, {total_val_pairs} true pairs")
    
    val_true_matches = defaultdict(set)
    val_target_ids = set()
    for row in val_pairs.iter_rows():
        val_true_matches[row[0]].add(row[1])
        val_target_ids.add(row[1])
        
    print("Loading S1...")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), separator="\t")
    s1_dict = {}
    for row in s1_df.filter(pl.col("entity_id").is_in(s1_val_ids)).iter_rows():
        eid, name, addr, ctry = row[0], row[1], row[2], row[3]
        c_tokens = extract_core_tokens(name)
        nums, addr_words = extract_addr_features(addr)
        s1_dict[eid] = {
            "name": name,
            "addr": addr,
            "country": ctry,
            "tokens": c_tokens,
            "sorted_tokens": " ".join(sorted(c_tokens)),
            "first2": " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else ""),
            "nums": nums,
            "addr_words": addr_words
        }
    del s1_df
    
    print("Loading targets...")
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
    
    # Indexes
    idx_sorted_tokens = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_num_addr = defaultdict(list)
    idx_domain = defaultdict(list)
    
    for row in target_pool.iter_rows():
        tid, name, addr, ctry = row[0], row[1], row[2] or "", row[3]
        c_tokens = extract_core_tokens(name)
        nums, addr_words = extract_addr_features(addr)
        sorted_tokens = " ".join(sorted(c_tokens))
        first2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
        d_stem = extract_domain_stem(name)
        
        if sorted_tokens:
            idx_sorted_tokens[(ctry, sorted_tokens)].append(tid)
        if first2:
            idx_first2[(ctry, first2)].append(tid)
            
        # Address key: (country, primary_number, primary_addr_word)
        if nums and addr_words:
            # Index first 2 numbers and first 2 distinctive address words
            for num in nums[:2]:
                for word in addr_words[:2]:
                    idx_num_addr[(ctry, num, word)].append(tid)
                    
        if d_stem:
            idx_domain[(ctry, d_stem)].append(tid)
            
    del target_pool
    print("Evaluating enhanced blocking...")
    all_captured = set()
    captured_by_key = Counter()
    cand_counts = []
    
    for s1_id, s1 in s1_dict.items():
        c = s1["country"]
        s_toks = s1["sorted_tokens"]
        f2 = s1["first2"]
        nums = s1["nums"]
        addr_words = s1["addr_words"]
        tokens = s1["tokens"]
        
        s1_cands = set()
        
        # 1. Sorted core tokens
        if s_toks and (c, s_toks) in idx_sorted_tokens:
            for tid in idx_sorted_tokens[(c, s_toks)]:
                s1_cands.add(tid)
                if tid in val_true_matches[s1_id]:
                    captured_by_key["Sorted Tokens"] += 1
                    
        # 2. First 2 tokens
        if f2 and (c, f2) in idx_first2:
            for tid in idx_first2[(c, f2)]:
                s1_cands.add(tid)
                if tid in val_true_matches[s1_id]:
                    captured_by_key["First 2 Tokens"] += 1
                    
        # 3. Address number + address distinctive word
        if nums and addr_words:
            for num in nums[:2]:
                for word in addr_words[:2]:
                    k = (c, num, word)
                    if k in idx_num_addr:
                        # Safety cap on hyper-frequent address keys
                        cands = idx_num_addr[k]
                        if len(cands) <= 50:
                            for tid in cands:
                                s1_cands.add(tid)
                                if tid in val_true_matches[s1_id]:
                                    captured_by_key["Addr Num + Word"] += 1
                                    
        # 4. Domain stem
        if tokens:
            concat = "".join(tokens)
            if (c, concat) in idx_domain:
                for tid in idx_domain[(c, concat)]:
                    s1_cands.add(tid)
                    if tid in val_true_matches[s1_id]:
                        captured_by_key["Domain Concat"] += 1
            for tok in tokens[:2]:
                if len(tok) >= 4 and (c, tok) in idx_domain:
                    for tid in idx_domain[(c, tok)]:
                        s1_cands.add(tid)
                        if tid in val_true_matches[s1_id]:
                            captured_by_key["Domain Token"] += 1
                            
        cand_counts.append(len(s1_cands))
        for tid in s1_cands:
            if tid in val_true_matches[s1_id]:
                all_captured.add((s1_id, tid))
                
    recall = len(all_captured) / total_val_pairs
    print(f"\n=== ENHANCED BLOCKING RESULTS ===")
    print(f"Total True Pairs: {total_val_pairs:,}")
    print(f"Captured True Pairs: {len(all_captured):,} ({recall * 100:.2f}% RECALL)")
    print(f"Average candidates per S1: {sum(cand_counts) / len(cand_counts):.1f}")
    print(f"Median candidates per S1: {sorted(cand_counts)[len(cand_counts)//2]}")
    print(f"Max candidates: {max(cand_counts)}")
    print("\nCaptures by key type:")
    for k, cnt in captured_by_key.most_common():
        print(f"  {k}: {cnt:,}")

if __name__ == "__main__":
    run()
