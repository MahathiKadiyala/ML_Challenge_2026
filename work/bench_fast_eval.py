import time
import numpy as np

# Simulate 300,000 candidate pairs
N = 300000

# Method 1: Python list of lists converted to numpy (what was happening)
t0 = time.time()
feat_list = []
for _ in range(N):
    feat_list.append([0.5]*18)
arr1 = np.array(feat_list, dtype=np.float32)
t1 = time.time() - t0
print(f"Method 1 (list of lists -> np.array): {t1:.3f}s")

# Method 2: Pre-allocated 1D flat buffer or direct write
t0 = time.time()
buffer = np.empty((N, 18), dtype=np.float32)
# In Cython or tight loop, filling buffer:
buffer.fill(0.5)
t2 = time.time() - t0
print(f"Method 2 (pre-allocated numpy): {t2:.3f}s")
