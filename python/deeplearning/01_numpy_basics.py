"""
NumPy — Foundation of the Python data-science stack.

Covers: array creation, reshaping, broadcasting, universal functions,
linear algebra, random sampling, and advanced indexing.
"""

import numpy as np

# ── 1. Array Creation ──────────────────────────────────────────────────────

arr1 = np.array([1, 2, 3, 4, 5])                   # 1-D from list
arr2 = np.zeros((3, 4), dtype=np.float32)          # 3×4 zeros
arr3 = np.ones((2, 3, 2))                          # 2×3×2 ones
arr4 = np.eye(5)                                   # 5×5 identity
arr5 = np.arange(0, 1, 0.2)                        # [0.0, 0.2, 0.4, 0.6, 0.8]
arr6 = np.linspace(0, 1, 5)                        # 5 points evenly spaced [0, 1]
arr7 = np.random.randn(3, 3)                       # 3×3 standard normal

print("── Array Creation ──")
print(f"arange(0,1,0.2) → {arr5}")
print(f"linspace(0,1,5) → {arr6}")
print(f"randn(3,3):\n{arr7}\n")

# ── 2. Shape & Reshaping ───────────────────────────────────────────────────

x = np.arange(12)
print(f"x.shape = {x.shape}  x.ndim = {x.ndim}")
print(f"x={x}")

x_2d = x.reshape(3, 4)               # reshape to 3×4
x_flat = x_2d.ravel()                # flatten (C-contiguous)
x_t = x_2d.T                         # transpose → 4×3
x_col = x[:, np.newaxis]             # add axis → (12, 1)

print(f"reshape(3,4):\n{x_2d}")
print(f"transpose:\n{x_t}")
print(f"newaxis shape: {x_col.shape}\n")

# ── 3. Broadcasting ────────────────────────────────────────────────────────

a = np.array([[1], [2], [3]])        # (3, 1)
b = np.array([10, 20, 30])           # (3,) → broadcasts to (1, 3) → (3, 3)
print("── Broadcasting ──")
print(f"{a} + {b} =\n{a + b}\n")     # result (3, 3)

# ── 4. Universal Functions (ufunc) ─────────────────────────────────────────

angles = np.array([0, np.pi / 2, np.pi])
print("── Ufuncs ──")
print(f"sin(angles) = {np.sin(angles)}")
print(f"exp([-1, 0, 1]) = {np.exp([-1, 0, 1])}")
print(f"relu (manual) = {np.maximum(0, [-2, -1, 0, 1, 2])}\n")

# ── 5. Linear Algebra ──────────────────────────────────────────────────────

A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x_solve = np.linalg.solve(A, b)      # solve Ax = b

print("── Linear Algebra ──")
print(f"A = \n{A}")
print(f"b = {b}")
print(f"solve(A,b) = {x_solve}")
print(f"check A@x = {A @ x_solve}")
print(f"det(A) = {np.linalg.det(A):.3f}")
print(f"eigvals(A) = {np.linalg.eigvals(A)}\n")

# ── 6. Random ──────────────────────────────────────────────────────────────

rng = np.random.default_rng(seed=42)               # modern RNG
print("── Random ──")
print(f"uniform(0,1,3) = {rng.uniform(0, 1, 3)}")
print(f"randint(0,10,5) = {rng.integers(0, 10, 5)}")

# ── 7. Advanced Indexing ───────────────────────────────────────────────────

data = np.arange(25).reshape(5, 5)
print("── Indexing ──")
print(f"data[[0,2,4]] =\n{data[[0, 2, 4]]}")     # fancy indexing
print(f"data[1:4, 1:4] =\n{data[1:4, 1:4]}")     # slicing
print(f"data[data > 15] = {data[data > 15]}")     # boolean mask

# ── 8. Memory & Performance (stride view, no copy) ─────────────────────────

big = np.random.randn(10_000, 10_000).astype(np.float32)
view = big[::2, ::2]                               # strided view, no copy
print(f"\n── Performance ──")
print(f"Full shape {big.shape}, view shape {view.shape}")
print(f"view.base is big → {view.base is big}")   # True → view, not copy
del big, view

print("\n✅ NumPy basics complete.")
