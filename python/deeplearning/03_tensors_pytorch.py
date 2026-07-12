"""
PyTorch Tensors — the fundamental data structure in PyTorch.

Covers: creation, indexing, math, device transfers (CPU / CUDA),
autograd (automatic differentiation), custom Dataset & DataLoader,
and a minimal learnable model (linear regression via SGD).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math

# ── 1. Tensor Creation ─────────────────────────────────────────────────────

print("── Creation ──")
t1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
t2 = torch.zeros(3, 4)
t3 = torch.ones(2, 3, 2)
t4 = torch.eye(5)
t5 = torch.randn(3, 3)          # standard normal
t6 = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
t7 = torch.linspace(0, 1, 5)    # 5 points from 0 to 1

print(f"tensor from list:\n{t1}")
print(f"zeros(3,4): shape={t2.shape}, dtype={t2.dtype}")
print(f"eye(5):\n{t4}")
print(f"arange(0,10,2): {t6}")
print(f"linspace(0,1,5): {t7}\n")

# ── 2. Indexing & Slicing ──────────────────────────────────────────────────

x = torch.arange(12).reshape(3, 4)
print("── Indexing ──")
print(f"x:\n{x}")
print(f"x[1, 2] = {x[1, 2]}")
print(f"x[:, 1:3] =\n{x[:, 1:3]}")
print(f"x[x > 5] = {x[x > 5]}")           # boolean indexing
print(f"x[[0, 2]] =\n{x[[0, 2]]}\n")      # fancy indexing

# ── 3. Math & Broadcasting ─────────────────────────────────────────────────

a = torch.tensor([[1.0], [2.0], [3.0]])    # (3, 1)
b = torch.tensor([10.0, 20.0, 30.0])       # (3,) → broadcasts to (1, 3)

print("── Broadcasting ──")
print(f"{a.shape} + {b.shape} → {(a + b).shape}")
print(f"result:\n{a + b}\n")

print(f"matmul (3×3) @ (3×2):\n{torch.randn(3,3) @ torch.randn(3,2)}\n")

# ── 4. Device Transfers ────────────────────────────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("── Device: CUDA ──")
else:
    device = torch.device("cpu")
    print("── Device: CPU (no CUDA available) ──")

t_cpu = torch.randn(100, 100)
t_dev = t_cpu.to(device)
print(f"Moved tensor: {t_dev.device}, shape={t_dev.shape}\n")

# ── 5. Autograd — Automatic Differentiation ────────────────────────────────

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x.pow(2).sum()  # y = x0² + x1²
y.backward()        # compute gradients
print("── Autograd ──")
print(f"x = {x.tolist()}")
print(f"y = x0² + x1² = {y.item()}")
print(f"dy/dx = {x.grad.tolist()}")   # expected: [2*x0, 2*x1] = [4, 6]
print()

# ── 6. Mini linear model (gradient descent from scratch) ───────────────────

torch.manual_seed(42)
# True weights
w_true = torch.tensor([[2.0], [-1.5]])
b_true = 0.5

# Generate synthetic data: N=200, 2 features
X = torch.randn(200, 2)
y = X @ w_true + b_true + 0.05 * torch.randn(200, 1)

# Model: y = X @ w + b
w = torch.randn(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([w, b], lr=0.1)

print("── Linear Regression from Scratch ──")
for epoch in range(200):
    y_pred = X @ w + b
    loss = (y_pred - y).pow(2).mean()  # MSE

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0:
        print(f"  Epoch {epoch:3d}  loss = {loss.item():.6f}")

print(f"\nLearned w = {w.detach().flatten().tolist()}")
print(f"True   w = {w_true.flatten().tolist()}")
print(f"Learned b = {b.item():.4f}  (true = {b_true})")
print()

# ── 7. Dataset & DataLoader ────────────────────────────────────────────────

class NoisySinDataset(Dataset):
    """y = sin(x) + small noise."""
    def __init__(self, n=500):
        self.x = torch.linspace(-math.pi, math.pi, n).unsqueeze(1)
        self.y = torch.sin(self.x) + 0.05 * torch.randn(n, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


ds = NoisySinDataset(200)
loader = DataLoader(ds, batch_size=32, shuffle=True)
print("── DataLoader ──")
for bx, by in loader:
    print(f"  batch shape: x={bx.shape}, y={by.shape}")
    break  # one batch is enough to show

# Stack all batches and verify the dataset
xs, ys = next(iter(DataLoader(ds, batch_size=len(ds))))
print(f"Full dataset: x in [{xs.min():.3f}, {xs.max():.3f}], "
      f"y in [{ys.min():.3f}, {ys.max():.3f}]")
print()

# ── 8. Quick nn.Module ────────────────────────────────────────────────────

model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loader = DataLoader(NoisySinDataset(500), batch_size=64, shuffle=True)

print("── nn.Module Training (sin(x) fit) ──")
for epoch in range(50):
    epoch_loss = 0.0
    for bx, by in loader:
        pred = model(bx)
        loss = nn.functional.mse_loss(pred, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * bx.size(0)
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:2d}  loss = {epoch_loss / len(loader.dataset):.6f}")

test_x = torch.tensor([[0.0], [math.pi / 2], [math.pi]])
test_y = model(test_x).detach()
print(f"\n  sin(0) ≈ {test_y[0,0]:.3f}  (true=0.0)")
print(f"  sin(π/2) ≈ {test_y[1,0]:.3f}  (true=1.0)")
print(f"  sin(π) ≈ {test_y[2,0]:.3f}  (true=0.0)")

print("\n✅ PyTorch tensors & training complete.")
