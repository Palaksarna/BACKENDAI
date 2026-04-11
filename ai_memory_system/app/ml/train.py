import torch
import torch.nn as nn

from .neural_model import MODEL_PATH, model


def train_mlp(data):
    if not data:
        return

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    inputs = torch.tensor(data, dtype=torch.float32)

    for _ in range(3):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, inputs)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
