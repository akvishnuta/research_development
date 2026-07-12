"""
CNN — Convolutional Neural Network for image classification on CIFAR-10.

Covers: Conv2D, BatchNormalization, MaxPooling2D, Dropout,
data augmentation, learning-rate scheduling, and evaluation
(confusion-matrix-like class accuracy breakdown).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

print(f"TensorFlow version: {tf.__version__}\n")

# ── 1. Load CIFAR-10 ──────────────────────────────────────────────────────

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

print("── CIFAR-10 ──")
print(f"Train: {x_train.shape},  Test: {x_test.shape}")
print(f"Labels: {y_train.min()}..{y_train.max()}")

# Normalize to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# ── 2. Data Augmentation ──────────────────────────────────────────────────

data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# ── 3. Model Definition ───────────────────────────────────────────────────

def build_cnn(dropout_rate=0.3, l2_reg=1e-4):
    """Build a CNN with BatchNorm, Dropout, and L2 regularization."""
    regularizer = keras.regularizers.l2(l2_reg)

    inputs = keras.Input(shape=(32, 32, 3))
    x = data_aug(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n── CNN Architecture ──")
model.summary()

# ── 4. Training ───────────────────────────────────────────────────────────

lr_schedule = callbacks.ReduceLROnPlateau(
    factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1)

print("\n── Training ──")
history = model.fit(
    x_train, y_train,
    validation_split=0.15,
    epochs=50,          # EarlyStopping will likely cut this short
    batch_size=128,
    verbose=1,          # show per-epoch progress
    callbacks=[lr_schedule, early_stop],
)

# ── 5. Evaluation ─────────────────────────────────────────────────────────

print("\n── Evaluation ──")
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}  (loss: {test_loss:.4f})")

# Per-class accuracy
preds = model.predict(x_test, verbose=0)
pred_classes = np.argmax(preds, axis=1)
true_classes = y_test.flatten()

print("\nPer-class accuracy:")
for i, name in enumerate(CLASS_NAMES):
    mask = true_classes == i
    acc = (pred_classes[mask] == i).mean()
    bar = "█" * int(acc * 30)
    print(f"  {name:>10}: {acc:.3f}  {bar}")

# ── 6. Prediction Demo ────────────────────────────────────────────────────

indices = np.random.choice(len(x_test), 5, replace=False)
samples = x_test[indices]
true_labels = y_test[indices].flatten()
pred_probs = model.predict(samples, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)

print("\n── Sample Predictions ──")
for i, idx in enumerate(indices):
    correct = "✓" if pred_labels[i] == true_labels[i] else "✗"
    conf = pred_probs[i, pred_labels[i]]
    print(f"  [{correct}] True={CLASS_NAMES[true_labels[i]]:>10}  "
          f"Pred={CLASS_NAMES[pred_labels[i]]:>10}  "
          f"conf={conf:.3f}")

print("\n✅ CNN complete.")
