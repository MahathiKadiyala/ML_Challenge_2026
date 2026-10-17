import time
import numpy as np
from rapidfuzz import fuzz

def compute_pair_features(s1, tg):
    s1_name_clean = s1["clean_name"]
    tg_name_clean = tg["clean_name"]
    s1_addr_clean = s1["clean_addr"]
    tg_addr_clean = tg["clean_addr"]
    
    name_ratio = fuzz.ratio(s1_name_clean, tg_name_clean)
    name_token_sort = fuzz.token_sort_ratio(s1_name_clean, tg_name_clean)
    name_token_set = fuzz.token_set_ratio(s1_name_clean, tg_name_clean)
    
    s1_core = s1["core_str"]
    tg_core = tg["core_str"]
    core_ratio = fuzz.ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    core_token_sort = fuzz.token_sort_ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    core_token_set = fuzz.token_set_ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    
    first_token_match = 1.0 if s1["first1"] and tg["first1"] and s1["first1"] == tg["first1"] else 0.0
    
    addr_ratio = fuzz.ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    addr_token_set = fuzz.token_set_ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    
    s1_nums = s1["nums_set"]
    tg_nums = tg["nums_set"]
    if s1_nums and tg_nums:
        has_nums = 1.0
        num_intersection = len(s1_nums & tg_nums)
        num_jaccard = num_intersection / len(s1_nums | tg_nums)
        num_exact = 1.0 if num_intersection > 0 else 0.0
    else:
        has_nums = 0.0
        num_jaccard = 0.0
        num_exact = 0.0
        
    d_stem = tg["domain_stem"]
    if d_stem:
        has_domain = 1.0
        domain_match = 1.0 if d_stem in s1_name_clean or (s1_core and d_stem in s1_core) else 0.0
    else:
        has_domain = 0.0
        domain_match = 0.0
        
    len_diff_name = abs(len(s1_name_clean) - len(tg_name_clean))
    len_diff_addr = abs(len(s1_addr_clean) - len(tg_addr_clean))
    is_s2 = 1.0 if tg["tid"].startswith("S2-") else 0.0
    
    return [
        name_ratio, name_token_sort, name_token_set,
        core_ratio, core_token_sort, core_token_set, first_token_match,
        addr_ratio, addr_token_sort, addr_token_set,
        has_nums, num_jaccard, num_exact,
        has_domain, domain_match,
        len_diff_name, len_diff_addr,
        is_s2
    ]

s1 = {
    "clean_name": "boulangerie saint michel paris",
    "clean_addr": "15 rue saint michel 75005 paris",
    "core_str": "boulangerie saint michel paris",
    "first1": "boulangerie",
    "nums_set": {"15", "75005"}
}

tg = {
    "tid": "S2-12345",
    "clean_name": "boulangerie st michel",
    "clean_addr": "15 r saint michel paris",
    "core_str": "boulangerie st michel",
    "first1": "boulangerie",
    "nums_set": {"15"},
    "domain_stem": None
}

t0 = time.time()
for _ in range(10000):
    f = compute_pair_features(s1, tg)
el = time.time() - t0
print(f"10,000 feature calculations in {el:.4f}s ({10000/el:.0f} pairs/sec)")
