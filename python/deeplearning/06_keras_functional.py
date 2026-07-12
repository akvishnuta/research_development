"""
Keras Functional API — non-linear topologies, multi-input / multi-output,
shared layers, and siamese networks.

The Functional API is the go-to when you need:
  - Branching / merging (Inception-style, ResNet skip connections)
  - Multiple inputs (image + metadata)
  - Multiple outputs (classification + bounding box)
  - Shared layers (siamese networks, weight tying)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input

# ── 1. Multi-Input Model ───────────────────────────────────────────────────
# Problem: predict house price from both numeric features AND a text description.

# --- Data simulation ---
NUM_FEATURES = 8
VOCAB_SIZE = 5000
MAX_TEXT_LEN = 50
NUM_SAMPLES = 2000

np.random.seed(42)
# Numeric branch input
numeric_input_data = np.random.randn(NUM_SAMPLES, NUM_FEATURES).astype(np.float32)
# Text branch input (random word indices)
text_input_data = np.random.randint(1, VOCAB_SIZE, (NUM_SAMPLES, MAX_TEXT_LEN))
# Target
y_data = (
    2.0 * numeric_input_data[:, 0]
    - 1.5 * numeric_input_data[:, 1]
    + 0.3 * numeric_input_data[:, 2]
    + np.random.randn(NUM_SAMPLES) * 0.1
).astype(np.float32)

# --- Functional model ---
# Numeric tower
num_in = Input(shape=(NUM_FEATURES,), name="numeric_input")
num_hidden = layers.Dense(32, activation="relu")(num_in)
num_hidden = layers.Dropout(0.3)(num_hidden)

# Text tower
txt_in = Input(shape=(MAX_TEXT_LEN,), name="text_input")
txt_embed = layers.Embedding(VOCAB_SIZE, 16)(txt_in)
txt_pool = layers.GlobalAveragePooling1D()(txt_embed)

# Concatenate
concat = layers.Concatenate()([num_hidden, txt_pool])
output = layers.Dense(1, name="output")(concat)

model_multi = Model(inputs=[num_in, txt_in], outputs=output)
model_multi.compile(optimizer="adam", loss="mse")

print("── Multi-Input Model ──")
model_multi.summary()

history = model_multi.fit(
    {"numeric_input": numeric_input_data, "text_input": text_input_data},
    y_data,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    verbose=0,
)
print(f"Best val loss: {min(history.history['val_loss']):.4f}\n")

# ── 2. Multi-Output Model ──────────────────────────────────────────────────
# Problem: classify a house as "good / bad" AND predict its exact price.

NUM_CLASSES = 2

cls_labels = np.random.randint(0, NUM_CLASSES, (NUM_SAMPLES,)).astype(np.float32)

inp = Input(shape=(NUM_FEATURES,), name="shared_input")
shared = layers.Dense(32, activation="relu")(inp)
shared = layers.Dropout(0.3)(shared)

# Regression head
reg_out = layers.Dense(1, name="regression")(shared)

# Classification head
cls_out = layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(shared)

model_multi_out = Model(inputs=inp, outputs=[reg_out, cls_out])
model_multi_out.compile(
    optimizer="adam",
    loss={"regression": "mse", "classification": "sparse_categorical_crossentropy"},
    loss_weights={"regression": 1.0, "classification": 0.3},
    metrics={"classification": "accuracy"},
)

print("── Multi-Output Model ──")
model_multi_out.summary()

history = model_multi_out.fit(
    numeric_input_data,
    {"regression": y_data, "classification": cls_labels},
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    verbose=0,
)
print("Best val metrics:")
for metric_name, values in history.history.items():
    if metric_name.startswith("val_"):
        print(f"  {metric_name}: {min(values) if 'loss' in metric_name else max(values):.4f}")
print()

# ── 3. Shared Layers (Siamese-style) ───────────────────────────────────────
# Problem: learn a similarity score between two numeric vectors.
# Weights are shared — the same layer processes both inputs.

SHARED_DIM = 16

shared_dense = layers.Dense(SHARED_DIM, activation="relu", name="shared_dense")

input_a = Input(shape=(NUM_FEATURES,), name="input_a")
input_b = Input(shape=(NUM_FEATURES,), name="input_b")

encoded_a = shared_dense(input_a)
encoded_b = shared_dense(input_b)

# L1 distance → similarity score
l1 = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([encoded_a, encoded_b])
sim_out = layers.Dense(1, activation="sigmoid", name="similarity")(l1)

model_siamese = Model(inputs=[input_a, input_b], outputs=sim_out)
model_siamese.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("── Shared-Layer (Siamese) Model ──")
model_siamese.summary()

# Dummy similarity data
np.random.seed(1)
data_a = np.random.randn(NUM_SAMPLES, NUM_FEATURES).astype(np.float32)
data_b = np.random.randn(NUM_SAMPLES, NUM_FEATURES).astype(np.float32)
# Random similarity labels (0 or 1)
sim_labels = np.random.randint(0, 2, (NUM_SAMPLES,)).astype(np.float32)

history = model_siamese.fit(
    [data_a, data_b],
    sim_labels,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    verbose=0,
)
best_val_acc = max(history.history["val_accuracy"])
print(f"Best val accuracy (similarity): {best_val_acc:.4f}\n")

# ── 4. Subclassing (custom forward pass) ───────────────────────────────────

class WideDeepModel(keras.Model):
    """Model with a wide (linear) path + deep path, merged."""
    def __init__(self, deep_units=16, **kwargs):
        super().__init__(**kwargs)
        self.deep = keras.Sequential([
            layers.Dense(deep_units, activation="relu"),
            layers.Dense(deep_units, activation="relu"),
        ])
        self.wide = layers.Dense(1, use_bias=False)   # linear bypass
        self.merge = layers.Dense(1)

    def call(self, inputs):
        return self.merge(self.deep(inputs) + self.wide(inputs))


wd_model = WideDeepModel()
wd_model.compile(optimizer="adam", loss="mse")
wd_model.fit(numeric_input_data, y_data, validation_split=0.2, epochs=10,
             batch_size=64, verbose=0)
print("── Subclassed Model ──")
print(f"Best val loss (Wide & Deep): {min(wd_model.history.history['val_loss']):.4f}")

print("\n✅ Keras Functional API complete.")
