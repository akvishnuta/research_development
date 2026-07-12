"""
Keras Sequential API — build models layer-by-layer.

Covers: regression (Boston Housing), binary classification (IMDB),
multi-class classification (Reuters), and common callbacks
(EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

print(f"Keras version: {keras.__version__}\n")

# ── 1. Regression — predict median house value ────────────────────────────

# Use the California housing dataset from sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model_reg = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1),  # Linear output for regression
])
model_reg.compile(optimizer="adam", loss="mse", metrics=["mae"])

print("── Regression (California Housing) ──")
history = model_reg.fit(
    X_train_s, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0,
    callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ],
)

# Evaluate
val_mae = min(history.history["val_mae"])
print(f"Best val MAE = ${val_mae * 100_000:.0f} (scaled to house price)")
print(f"Test MAE = ${model_reg.evaluate(X_test_s, y_test, verbose=0)[1] * 100_000:.0f}\n")

# ── 2. Binary Classification — IMDB sentiment ─────────────────────────────

max_features = 10_000
max_len = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=max_features
)

# Pad sequences to uniform length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

model_clf = keras.Sequential([
    layers.Embedding(max_features, 64, input_length=max_len),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])
model_clf.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("── Binary Classification (IMDB) ──")
history = model_clf.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    verbose=0,
)
val_acc = max(history.history["val_accuracy"])
print(f"Best val accuracy = {val_acc:.4f}")
print(f"Test accuracy = {model_clf.evaluate(x_test, y_test, verbose=0)[1]:.4f}\n")

# ── 3. Multi-class Classification — Reuters ────────────────────────────────

num_classes = 46
(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(
    num_words=max_features
)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model_mc = keras.Sequential([
    layers.Embedding(max_features, 64, input_length=max_len),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.5),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])
model_mc.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("── Multi-class Classification (Reuters) ──")
history = model_mc.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=128,
    verbose=0,
)
val_acc = max(history.history["val_accuracy"])
print(f"Best val accuracy = {val_acc:.4f}")
print(f"Test accuracy = {model_mc.evaluate(x_test, y_test, verbose=0)[1]:.4f}\n")

# ── 4. Callbacks demo ─────────────────────────────────────────────────────

model_demo = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(X.shape[1],)),
    layers.Dense(1),
])
model_demo.compile(optimizer="adam", loss="mse")

print("── Callback Demo ──")
cb_list = [
    callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=0),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6, verbose=0),
]
history = model_demo.fit(
    X_train_s, y_train,
    validation_split=0.2,
    epochs=100,       # would run 100 — EarlyStopping will cut it short
    batch_size=32,
    verbose=0,
    callbacks=cb_list,
)
print(f"Stopped at epoch {len(history.history['loss'])} / 100 "
      f"(early stopping triggered)")
print(f"Final val_loss = {history.history['val_loss'][-1]:.4f}")

print("\n✅ Keras Sequential API complete.")
