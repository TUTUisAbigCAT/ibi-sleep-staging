import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math

# --- 0. 环境检查与 Mamba 导入 ---
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Error: mamba_ssm not installed (pip install mamba-ssm, needs CUDA)")
    Mamba = None

# --- 1. 基础 CNN 模块 (保留原代码逻辑并优化) ---

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.bn2(self.conv2(x))
        x += residual
        return self.dropout2(self.relu2(x))

class DeepFeatureExtractor(nn.Module):
    def __init__(self, initial_channels, num_layers=4, target_dim=256):
        super(DeepFeatureExtractor, self).__init__()
        layers = []
        out_channels_first = 16
        layers.append(nn.Conv1d(initial_channels, out_channels_first, kernel_size=7, stride=1, padding=3))
        layers.append(nn.ReLU(inplace=True))
        current_channels = out_channels_first

        for i in range(num_layers):
            out_channels = 32 * (2**i)
            layers.append(ResBlock(current_channels, out_channels, stride=4))
            current_channels = out_channels

        self.layers = nn.Sequential(*layers)
        
        # [关键修正]：使用自适应池化将时序长度降为 1，确保输出维度严格对齐 target_dim (256)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.final_conv = nn.Conv1d(current_channels, target_dim, kernel_size=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.final_pool(x)
        x = self.final_conv(x)
        return x.view(x.size(0), -1)

# --- 2. 时序模块: BiMambaBlock (双向扫描) ---

class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super(BiMambaBlock, self).__init__()
        if Mamba is None: raise ImportError("mamba_ssm not found")
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # 投影层：将双向拼接后的维度 (d_model * 2) 降回 d_model
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [Batch, Seq, d_model]
        fwd_out = self.mamba_fwd(x)
        x_rev = torch.flip(x, dims=[1])
        bwd_out = self.mamba_bwd(x_rev)
        bwd_out = torch.flip(bwd_out, dims=[1])
        
        combined = torch.cat([fwd_out, bwd_out], dim=-1)
        # 残差连接并归一化
        return self.norm(self.output_proj(combined) + x)

# --- 3. 主模型: MambaSleepNet（只有分类头，无对比学习） ---

class MambaSleepNet(nn.Module):
    """CNN per 30-s epoch -> bidirectional Mamba over the night -> classifier (trained with cross-entropy)."""

    def __init__(
        self,
        num_features,
        hidden_dim=256,
        num_layers=4,
        num_classes=4,
        d_state=64,
        num_mamba_layers=None,
        **kwargs          # 捕获 engine 传进来的其他冗余参数
    ):
        super(MambaSleepNet, self).__init__()
        if num_mamba_layers is not None:
            num_layers = num_mamba_layers
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        # 1. CNN 特征提取
        self.feature_extractor = DeepFeatureExtractor(
            initial_channels=num_features, target_dim=hidden_dim
        )

        # 2. 时序建模 (Bi-Mamba 序列扫描)
        self.mamba_layers = nn.Sequential(
            *[BiMambaBlock(d_model=hidden_dim, d_state=d_state) for _ in range(num_layers)]
        )

        # 3. 分类头
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward_features(self, x, lengths=None):
        # x: [Batch, Seq, Samples] (如 4, 100, 750)
        batch_size, max_seq, samples = x.shape

        # Step 1: CNN 提取每一个 30s 信号段的特征
        if lengths is None:
            lengths = torch.full((batch_size,), max_seq, device=x.device, dtype=torch.long)
        else:
            lengths = torch.as_tensor(lengths, device=x.device)
        if lengths.shape != (batch_size,) or torch.any(lengths < 1) or torch.any(lengths > max_seq):
            raise ValueError("lengths must contain one valid sequence length per batch item")
        if lengths.is_floating_point() and torch.any(lengths != lengths.floor()):
            raise ValueError("lengths must be integers")
        lengths = lengths.long()
        valid = torch.arange(max_seq, device=x.device)[None, :] < lengths[:, None]
        # Padding must not influence CNN BatchNorm or the backward Mamba scan.
        valid_features = self.feature_extractor(x[valid].reshape(-1, 1, samples))
        x_feat = valid_features.new_zeros(batch_size, max_seq, valid_features.size(-1))
        x_feat[valid] = valid_features

        # Step 2: Mamba 捕捉整晚睡眠的时序逻辑（每晚按真实长度扫描）
        if torch.all(lengths == max_seq):
            return self.mamba_layers(x_feat)
        x_mamba = torch.zeros_like(x_feat)
        for length in lengths.unique().tolist():
            indices = torch.where(lengths == length)[0]
            sequence = x_feat.index_select(0, indices)[:, :length].contiguous()
            encoded = self.mamba_layers(sequence)
            encoded = F.pad(encoded, (0, 0, 0, max_seq - length))
            x_mamba = x_mamba.index_copy(0, indices, encoded)
        return x_mamba

    def forward(self, x, lengths=None):
        # Step 3: 分类输出
        return self.classifier(self.forward_features(x, lengths))


# --- 4. 测试函数 ---

def test_mamba():
    print("--- Starting MambaSleepNet Integration Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 100, 750).to(device)
    model = MambaSleepNet(num_features=1, hidden_dim=256, num_layers=4, num_classes=4).to(device)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Integration Successful.")


if __name__ == "__main__":
    test_mamba()
