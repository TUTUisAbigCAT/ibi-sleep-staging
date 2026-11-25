"""Tests run on CPU: modeling/ is on the path, and a small causal stand-in replaces the CUDA Mamba kernel
when mamba_ssm is not installed (everything else in MambaSleepNet is the real code)."""
from __future__ import annotations

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "modeling", ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

try:
    import mamba_ssm  # noqa: F401
except ImportError:
    import torch
    from torch import nn

    class Mamba(nn.Module):
        """Causal stand-in with Mamba's constructor: depthwise causal conv + gated projection."""

        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.conv = nn.Conv1d(d_model, d_model, d_conv, groups=d_model, padding=d_conv - 1)
            self.gate = nn.Linear(d_model, d_model)

        def forward(self, x):
            y = self.conv(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
            return y * torch.sigmoid(self.gate(x))

    sys.modules["mamba_ssm"] = types.SimpleNamespace(Mamba=Mamba)
