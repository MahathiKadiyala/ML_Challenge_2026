"""
Analyze true matches missed by the 5 blocking keys.
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
    text = unidecode.unidecode(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_core_name(name):
    norm = normalize_text(name)
    tokens = norm.split()
    while tokens and tokens[0] in HONORIFICS:
        tokens.pop(0)
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
    m = re.search(r"([a-z0-9]+)\.(com|org|net|in|co|us|fr|info|biz)\b", name.lower())
    if m:
        return m.group(1)
    return None

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
    
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), separator="\t")
    s2_needed = s2_df.filter(pl.col("entity_id").is_in(val_target_ids))
    del s2_df
    
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), separator="\t")
    s3_needed = s3_df.filter(pl.col("entity_id").is_in(val_target_ids))
    del s3_df
    
    target_pool = pl.concat([s2_needed, s3_needed])
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
    
    index_exact_core = defaultdict(list)
    index_first2_tokens = defaultdict(list)
    index_exact_addr = defaultdict(list)
    index_prefix_num = defaultdict(list)
    index_domain = defaultdict(list)
    
    for tid, rec in target_records.items():
        c = rec["country"]
        cn = rec["core_name"]
        tokens = cn.split()
        if cn:
            index_exact_core[(c, cn)].append(tid)
        if len(tokens) >= 2:
            index_first2_tokens[(c, " ".join(tokens[:2]))].append(tid)
        elif len(tokens) == 1:
            index_first2_tokens[(c, tokens[0])].append(tid)
        na = rec["norm_addr"]
        if na:
            index_exact_addr[(c, na)].append(tid)
        if len(cn) >= 3 and rec["nums"]:
            pref = cn[:3]
            for num in rec["nums"][:2]:
                index_prefix_num[(c, pref, num)].append(tid)
        ds = rec["domain_stem"]
        if ds:
            index_domain[(c, ds)].append(tid)
            
    misses = []
    
    for s1_id, s1_rec in s1_dict.items():
        c = s1_rec["country"]
        cn = s1_rec["core_name"]
        tokens = cn.split()
        na = s1_rec["norm_addr"]
        nums = s1_rec["nums"]
        
        s1_candidates = set()
        if cn and (c, cn) in index_exact_core:
            s1_candidates.update(index_exact_core[(c, cn)])
        k2 = " ".join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else "")
        if k2 and (c, k2) in index_first2_tokens:
            s1_candidates.update(index_first2_tokens[(c, k2)])
        if na and (c, na) in index_exact_addr:
            s1_candidates.update(index_exact_addr[(c, na)])
        if len(cn) >= 3 and nums:
            pref = cn[:3]
            for num in nums[:2]:
                if (c, pref, num) in index_prefix_num:
                    s1_candidates.update(index_prefix_num[(c, pref, num)])
        if tokens:
            concat_name = "".join(tokens)
            if (c, concat_name) in index_domain:
                s1_candidates.update(index_domain[(c, concat_name)])
            if (c, tokens[0]) in index_domain:
                s1_candidates.update(index_domain[(c, tokens[0])])
                
        for tid in val_true_matches[s1_id]:
            if tid not in s1_candidates:
                misses.append((s1_id, tid))
                
    print(f"Total true pairs: {len(val_pairs):,}, Missed: {len(misses):,} ({len(misses) / len(val_pairs) * 100:.2f}%)")
    
    with open("work/missed_pairs.txt", "w", encoding="utf-8") as out:
        for s1_id, tid in misses[:30]:
            s1 = s1_dict[s1_id]
            tg = target_records[tid]
            out.write(f"S1:  Name='{s1['name']}' (Core='{s1['core_name']}')\n     Addr='{s1['addr']}' (Nums={s1['nums']})\n")
            out.write(f"Tgt: Name='{tg['name']}' (Core='{tg['core_name']}')\n     Addr='{tg['addr']}' (Nums={tg['nums']})\n")
            out.write("-" * 60 + "\n")
    print("Wrote missed pairs to work/missed_pairs.txt")

if __name__ == "__main__":
    run()
