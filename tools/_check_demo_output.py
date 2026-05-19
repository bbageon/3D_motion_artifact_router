"""ad-hoc: demo.py output 의 shape + non-zero 확인."""
import sys
import numpy as np

for path in sys.argv[1:]:
    arr = np.load(path)
    print(f"{path}: shape={arr.shape}, ndim={arr.ndim}, total_elem={arr.size}, abs_sum={np.abs(arr).sum():.4f}, nonzero={(np.abs(arr).sum() > 0)}")
