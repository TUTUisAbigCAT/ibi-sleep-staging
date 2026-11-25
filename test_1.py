import numpy as np

ofs = 4
tfs = 2

t_stage = np.array([0,1,2,3,4,5,6,7,8]) / ofs

t_ds = np.array([0,1,2,3]) / tfs

idx = np.searchsorted(t_stage, t_ds, side="right")

print(idx - 1, '##################')
