from pathlib import Path
import numpy as np


for p in Path(".").rglob("*npy"):
    print(p)
    arr = np.load(p)
    arr[np.isnan(arr)] = -1
    np.save(p, arr)
