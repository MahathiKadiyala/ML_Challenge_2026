"""
Inspect remaining misses from enhanced blocking to reach 92%+ recall.
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
    text = re.sub(r"([a-z])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_core_tokens(name):
    norm = clean_and_translit(name)
    tokens = norm.split()
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

def run():
    gt = pl.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), separator="\t")
    gt_non_empty = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    pairs = gt_non_empty.select([
        pl.col("source1_entity_id").alias("s1_id"),
        pl.col("matched_entity_ids").str.split(",").alias("mid")
    ]).explode("mid")
    
    s1_val_ids = set(gt_non_empty.sample(n=5_000, seed=42)["source1_entity_id"].to_list())
    val_pairs = pairs.filter(pl.col("s1_id").is_in(s1_val_ids))
    
    val_true_matches = defaultdict(set)
    val_target_ids = set()
    for row in val_pairs.iter_rows():
        val_true_matches[row[0]].add(row[1])
        val_target_ids.add(row[1])
        
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
    
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), separator="\t")
    s2_needed = s2_df.filter(pl.col("entity_id").is_in(val_target_ids))
    del s2_df
    
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), separator="\t")
    s3_needed = s3_df.filter(pl.col("entity_id").is_in(val_target_ids))
    del s3_df
    
    target_pool = pl.concat([s2_needed, s3_needed])
    target_records = {}
    idx_sorted_tokens = defaultdict(list)
    idx_first2 = defaultdict(list)
    idx_num_addr = defaultdict(list)
    
    for row in target_pool.iter_rows():
        tid, name, addr, ctry = row[0], row[1], row[2] or "", row[3]
        c_tokens = extract_core_tokens(name)
        nums, addr_words = extract_addr_features(addr)
        sorted_tokens = " ".join(sorted(c_tokens))
        first2 = " ".join(c_tokens[:2]) if len(c_tokens) >= 2 else (c_tokens[0] if c_tokens else "")
        target_records[tid] = {
            "name": name,
            "addr": addr,
            "country": ctry,
            "tokens": c_tokens,
            "nums": nums,
            "addr_words": addr_words
        }
        if sorted_tokens:
            idx_sorted_tokens[(ctry, sorted_tokens)].append(tid)
        if first2:
            idx_first2[(ctry, first2)].append(tid)
        if nums and addr_words:
            for num in nums[:2]:
                for word in addr_words[:2]:
                    idx_num_addr[(ctry, num, word)].append(tid)
    del target_pool
    
    misses = []
    for s1_id, s1 in s1_dict.items():
        c = s1["country"]
        s_toks = s1["sorted_tokens"]
        f2 = s1["first2"]
        nums = s1["nums"]
        addr_words = s1["addr_words"]
        
        s1_cands = set()
        if s_toks and (c, s_toks) in idx_sorted_tokens:
            s1_cands.update(idx_sorted_tokens[(c, s_toks)])
        if f2 and (c, f2) in idx_first2:
            s1_cands.update(idx_first2[(c, f2)])
        if nums and addr_words:
            for num in nums[:2]:
                for word in addr_words[:2]:
                    k = (c, num, word)
                    if k in idx_num_addr:
                        s1_cands.update(idx_num_addr[k])
                        
        for tid in val_true_matches[s1_id]:
            if tid not in s1_cands:
                misses.append((s1_id, tid))
                
    print(f"Total true pairs: {len(val_pairs):,}, Missed: {len(misses):,} ({len(misses) / len(val_pairs) * 100:.2f}%)")
    with open("work/remaining_misses.txt", "w", encoding="utf-8") as out:
        for s1_id, tid in misses[:30]:
            s1 = s1_dict[s1_id]
            tg = target_records[tid]
            out.write(f"S1:  Name='{s1['name']}' (Tokens={s1['tokens']})\n     Addr='{s1['addr']}' (Nums={s1['nums']}, Words={s1['addr_words'][:3]})\n")
            out.write(f"Tgt: Name='{tg['name']}' (Tokens={tg['tokens']})\n     Addr='{tg['addr']}' (Nums={tg['nums']}, Words={tg['addr_words'][:3]})\n")
            out.write("-" * 60 + "\n")
    print("Wrote remaining misses to work/remaining_misses.txt")

if __name__ == "__main__":
    run()
