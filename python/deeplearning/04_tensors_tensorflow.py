"""
TensorFlow Tensors — eager execution, shape ops, tf.function, and GradientTape.

Covers: creation, reshaping, broadcasting, tf.function (graph compilation),
automatic differentiation with GradientTape, and a custom training loop.
"""

import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")
print()

# ── 1. Tensor Creation ─────────────────────────────────────────────────────

print("── Creation ──")
t1 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
t2 = tf.zeros([3, 4])
t3 = tf.ones([2, 3, 2])
t4 = tf.eye(5)
t5 = tf.random.normal([3, 3])     # standard normal
t6 = tf.range(0, 10, delta=2)
t7 = tf.linspace(0.0, 1.0, 5)

print(f"constant:\n{t1.numpy()}")
print(f"zeros(3,4): shape={t2.shape}, dtype={t2.dtype}")
print(f"random.normal(3,3):\n{t5.numpy()}")
print(f"range(0,10,2): {t6.numpy()}")
print(f"linspace(0,1,5): {t7.numpy()}\n")

# ── 2. Reshape & Axis Ops ──────────────────────────────────────────────────

x = tf.reshape(tf.range(12), (3, 4))
print("── Reshape ──")
print(f"reshape(12 → 3×4):\n{x.numpy()}")
print(f"transpose → 4×3:\n{tf.transpose(x).numpy()}")
print(f"expand_dims (3,4) → (3,1,4): shape={tf.expand_dims(x, 1).shape}")
print(f"squeeze on (1,3,1,4): shape={tf.squeeze(tf.ones([1, 3, 1, 4])).shape}\n")

# ── 3. Broadcasting ────────────────────────────────────────────────────────

a = tf.constant([[1.0], [2.0], [3.0]])   # (3, 1)
b = tf.constant([10.0, 20.0, 30.0])      # (3,) → (1, 3)
print("── Broadcasting ──")
print(f"{a.shape} + {b.shape} → {(a + b).shape}")
print(f"result:\n{(a + b).numpy()}\n")

# ── 4. tf.function — graph compilation ─────────────────────────────────────

@tf.function
def compute_stats(x):
    """Compute mean and std in graph mode (faster for repeated calls)."""
    return tf.reduce_mean(x), tf.math.reduce_std(x)


data = tf.random.normal([1000, 50])
mean, std = compute_stats(data)
print("── tf.function ──")
print(f"mean={mean.numpy():.4f}, std={std.numpy():.4f}  (expect ≈ 0, 1)")
# Check graph was built
print(f"has graph: {hasattr(compute_stats, 'get_concrete_function')}\n")

# ── 5. GradientTape — Automatic Differentiation ────────────────────────────

x = tf.Variable([2.0, 3.0])
with tf.GradientTape() as tape:
    y = tf.reduce_sum(x ** 2)   # y = x0² + x1²

grad = tape.gradient(y, x)
print("── GradientTape ──")
print(f"x = {x.numpy()}, y = x0² + x1² = {y.numpy()}")
print(f"dy/dx = {grad.numpy()}  (expected [4, 6])")
print()

# ── 6. Custom Training Loop (linear regression) ────────────────────────────

tf.random.set_seed(42)
# True weights
w_true = tf.constant([[2.0], [-1.5]])
b_true = 0.5

# Synthetic data
X = tf.random.normal([200, 2])
y = X @ w_true + b_true + 0.05 * tf.random.normal([200, 1])

# Model variables
w = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(0.0)

optimizer = tf.optimizers.SGD(learning_rate=0.1)

print("── Training Loop ──")
for epoch in range(200):
    with tf.GradientTape() as tape:
        y_pred = X @ w + b
        loss = tf.reduce_mean((y_pred - y) ** 2)

    grads = tape.gradient(loss, [w, b])
    optimizer.apply_gradients(zip(grads, [w, b]))

    if epoch % 40 == 0:
        print(f"  Epoch {epoch:3d}  loss = {loss.numpy():.6f}")

print(f"\nLearned w = {w.numpy().flatten()}")
print(f"True   w = {w_true.numpy().flatten()}")
print(f"Learned b = {b.numpy():.4f}  (true = {b_true})")
print()

# ── 7. Dataset API ─────────────────────────────────────────────────────────

ds = tf.data.Dataset.from_tensor_slices((X, y))
ds = ds.shuffle(200).batch(32)

print("── tf.data ──")
for bx, by in ds.take(1):
    print(f"  batch: x={bx.shape}, y={by.shape}")

# Prefetch for performance
ds_perf = ds.prefetch(tf.data.AUTOTUNE)
print(f"  prefetch buffer: {ds_perf._prefetch_buffer_size if hasattr(ds_perf, '_prefetch_buffer_size') else 'AUTOTUNE'}")

# ── 8. Model subclassing ───────────────────────────────────────────────────

class TwoLayerNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(16, activation="relu")
        self.out = tf.keras.layers.Dense(1)

    def call(self, x):
        return self.out(self.hidden(x))


model = TwoLayerNet()
optimizer = tf.optimizers.Adam(learning_rate=0.01)
ds_train = tf.data.Dataset.from_tensor_slices(
    (tf.linspace(-np.pi, np.pi, 500).reshape(-1, 1),
     tf.sin(tf.linspace(-np.pi, np.pi, 500).reshape(-1, 1)) + 0.05 * tf.random.normal([500, 1]))
).shuffle(500).batch(64)

print("\n── Model Subclassing (sin(x) fit) ──")
for epoch in range(50):
    epoch_loss = 0.0
    num_batches = 0
    for bx, by in ds_train:
        with tf.GradientTape() as tape:
            pred = model(bx)
            loss = tf.reduce_mean((pred - by) ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += loss.numpy()
        num_batches += 1
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:2d}  loss = {epoch_loss / num_batches:.6f}")

test_x = tf.constant([[0.0], [np.pi / 2], [np.pi]], dtype=tf.float32)
test_y = model(test_x)
print(f"\n  sin(0) ≈ {test_y[0, 0].numpy():.3f}  (true=0.0)")
print(f"  sin(π/2) ≈ {test_y[1, 0].numpy():.3f}  (true=1.0)")
print(f"  sin(π) ≈ {test_y[2, 0].numpy():.3f}  (true=0.0)")

print("\n✅ TensorFlow tensors & training complete.")
