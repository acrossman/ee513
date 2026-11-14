import numpy as np
import torch
import torch.nn as nn


class SmallDnCNN(nn.Module):
    def __init__(self, depth=6, channels=32):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(channels, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, y):
        return y - self.body(y)


def n_params(module):
    return sum(t.numel() for t in module.parameters())


def train(inputs, targets, seed, steps=800, patch=64, batch=16, lr=1e-3):
    torch.manual_seed(seed)
    model = SmallDnCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng_patch = np.random.default_rng(seed)
    H, W = inputs.shape
    x_in = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32))
    x_tg = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
    losses = []
    model.train()
    for step in range(steps):
        rows = rng_patch.integers(0, H - patch + 1, batch)
        cols = rng_patch.integers(0, W - patch + 1, batch)
        y = torch.stack([x_in[r:r + patch, c:c + patch] for r, c in zip(rows, cols)])[:, None]
        t = torch.stack([x_tg[r:r + patch, c:c + patch] for r, c in zip(rows, cols)])[:, None]
        loss = torch.mean((model(y) - t) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, np.array(losses)


def denoise(model, img):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None])[0, 0].numpy()
