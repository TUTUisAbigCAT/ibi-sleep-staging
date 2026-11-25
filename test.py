import torch

# 1. 检查 CUDA 是否可用（即是否有兼容的 NVIDIA GPU 和驱动）
cuda_available = torch.cuda.is_available()
print(f"CUDA 可用性: {cuda_available}")

if cuda_available:
    # 2. 获取当前可用的 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")

    # 3. 打印 PyTorch 编译时是否包含 CUDA 支持 (通常与 is_available() 一致)
    print(f"PyTorch 构建包含 CUDA: {torch.backends.cuda.is_built()}")

    # 4. 打印当前默认的 GPU 设备的名称和版本信息
    current_device_index = torch.cuda.current_device()
    print(f"当前默认设备索引: {current_device_index}")
    print(f"当前默认设备名称: {torch.cuda.get_device_name(current_device_index)}")
    
    # 5. 打印 PyTorch 支持的 CUDA 版本（这是 PyTorch 编译时链接的 CUDA 版本，不一定是你系统安装的驱动版本）
    print(f"PyTorch 编译时链接的 CUDA 版本: {torch.version.cuda}")

    # 6. 检查特定张量是否在 GPU 上
    x = torch.tensor([1.0], device='cuda')
    print(f"示例张量设备: {x.device}")
else:
    print("当前环境中 PyTorch 未检测到可用的 GPU。")

# 打印 PyTorch 的版本
print(f"PyTorch 版本: {torch.__version__}")
