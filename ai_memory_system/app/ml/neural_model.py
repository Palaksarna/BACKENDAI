from pathlib import Path

import torch
import torch.nn as nn


class MemoryMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        return self.net(x)


MODEL_DIM = 384
MODEL_PATH = Path(__file__).resolve().parents[3] / "mlp.pth"

model = MemoryMLP(MODEL_DIM)

try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
except (FileNotFoundError, RuntimeError):
    model.eval()
