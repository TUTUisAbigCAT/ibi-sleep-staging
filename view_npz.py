import numpy as np

data = np.load('/home/yxtao/scratch/WatchSleepProcessedDataset/SHHS_MESA_IBI/mesa-6460.npz')

print("文件中的键:", list(data.keys()))

for key in data.keys():
    arr = data[key]
    print(f"{key}: 形状={arr.shape}, 类型={arr.dtype}, 大小={arr.size}")

print("data前10个值:", data['data'][:10])
print("stages前10个值:", data['stages'][:10])
print("fs:", data['fs'])
print("ahi:", data['ahi'])

