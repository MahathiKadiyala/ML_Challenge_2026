"""
Test 12-core parallel batch processing speed.
"""
import os
import time
import multiprocessing as mp
from functools import partial
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz

def worker_task(chunk, s1_data, target_data):
    # Process small chunk
    res = 0
    for s1_id in chunk:
        s1 = s1_data[s1_id]
        for tid in target_data:
            tg = target_data[tid]
            r = fuzz.ratio(s1[0], tg[0])
            if r > 50:
                res += 1
    return res

def main():
    print(f"Testing multiprocessing with {mp.cpu_count()} CPU cores...")
    s1_data = {f"S1-{i}": (f"Business Name {i}", f"123 Street Road {i}") for i in range(1000)}
    target_data = {f"S2-{j}": (f"Target Name {j}", f"123 Street Road {j}") for j in range(100)}
    s1_ids = list(s1_data.keys())
    
    t0 = time.time()
    chunk_size = len(s1_ids) // mp.cpu_count()
    chunks = [s1_ids[i:i+chunk_size] for i in range(0, len(s1_ids), chunk_size)]
    
    with mp.Pool(mp.cpu_count()) as pool:
        fn = partial(worker_task, s1_data=s1_data, target_data=target_data)
        results = pool.map(fn, chunks)
        
    print(f"Processed 100k pairs in {time.time()-t0:.2f}s across {mp.cpu_count()} cores. Sum: {sum(results)}")

if __name__ == "__main__":
    main()
