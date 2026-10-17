import os
import sys
import gc
import time
import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "student_resource", "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "work", "models", "lightgbm_model.txt")
WORK_DIR = os.path.join(BASE_DIR, "work")

sys.path.append(WORK_DIR)
from step12_train_final_model import (
    clean_and_translit, extract_core_tokens, extract_addr_features,
    extract_domain_stem
)

class TargetRecord:
    __slots__ = ('tid', 'clean_name', 'clean_addr', 'core_str', 'first1', 'nums_set', 'domain_stem')
    def __init__(self, tid, clean_name, clean_addr, core_str, first1, nums_set, domain_stem):
        self.tid = tid
        self.clean_name = clean_name
        self.clean_addr = clean_addr
        self.core_str = core_str
        self.first1 = first1
        self.nums_set = nums_set
        self.domain_stem = domain_stem

def compute_pair_features_fast(s1, tg, nr=None):
    s1_name_clean = s1["clean_name"]
    tg_name_clean = tg.clean_name
    s1_addr_clean = s1["clean_addr"]
    tg_addr_clean = tg.clean_addr
    
    name_ratio = nr if nr is not None else fuzz.ratio(s1_name_clean, tg_name_clean)
    name_token_sort = fuzz.token_sort_ratio(s1_name_clean, tg_name_clean)
    name_token_set = fuzz.token_set_ratio(s1_name_clean, tg_name_clean)
    
    s1_core = s1["core_str"]
    tg_core = tg.core_str
    core_ratio = fuzz.ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    core_token_sort = fuzz.token_sort_ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    core_token_set = fuzz.token_set_ratio(s1_core, tg_core) if s1_core and tg_core else 0.0
    
    first_token_match = 1.0 if s1["first1"] and tg.first1 and s1["first1"] == tg.first1 else 0.0
    
    addr_ratio = fuzz.ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    addr_token_set = fuzz.token_set_ratio(s1_addr_clean, tg_addr_clean) if s1_addr_clean and tg_addr_clean else 0.0
    
    s1_nums = s1["nums_set"]
    tg_nums = tg.nums_set
    if s1_nums and tg_nums:
        has_nums = 1.0
        num_intersection = len(s1_nums & tg_nums)
        num_jaccard = num_intersection / len(s1_nums | tg_nums)
        num_exact = 1.0 if num_intersection > 0 else 0.0
    else:
        has_nums = 0.0
        num_jaccard = 0.0
        num_exact = 0.0
        
    d_stem = tg.domain_stem
    if d_stem:
        has_domain = 1.0
        domain_match = 1.0 if d_stem in s1_name_clean or (s1_core and d_stem in s1_core) else 0.0
    else:
        has_domain = 0.0
        domain_match = 0.0
        
    len_diff_name = abs(len(s1_name_clean) - len(tg_name_clean))
    len_diff_addr = abs(len(s1_addr_clean) - len(tg_addr_clean))
    is_s2 = 1.0 if tg.tid.startswith("S2-") else 0.0
    
    return [
        name_ratio, name_token_sort, name_token_set,
        core_ratio, core_token_sort, core_token_set, first_token_match,
        addr_ratio, addr_token_sort, addr_token_set,
        has_nums, num_jaccard, num_exact,
        has_domain, domain_match,
        len_diff_name, len_diff_addr,
        is_s2
    ]

print("Verifying fast batch test...")
booster = lgb.Booster(model_file=MODEL_PATH)
print("Model loaded.")
