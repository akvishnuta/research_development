"""
Transfer Learning — leverage a pretrained backbone for a new task.

Strategy:
  1. Load MobileNetV2 (pretrained on ImageNet), excluding the top.
  2. Freeze the backbone weights.
  3. Add a new classification head for CIFAR-10.
  4. Train only the head first.
  5. Optionally fine-tune the top layers of the backbone.

This yields good accuracy with a fraction of the training time
and data required to train from scratch.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

print(f"TensorFlow version: {tf.__version__}\n")

# ── 1. Load & Prepare CIFAR-10 ────────────────────────────────────────────

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

IMG_SIZE = 96  # MobileNetV2 expects at least 32×32, but was designed for 224×224.
               # 96×96 is a reasonable middle ground.

def preprocess(images, labels):
    """Resize small CIFAR images to IMG_SIZE and normalize."""
    images = tf.image.resize(images, (IMG_SIZE, IMG_SIZE))
    images = tf.cast(images, tf.float32) / 255.0
    labels = tf.squeeze(labels)
    return images, labels


BATCH_SIZE = 64

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("── Data prepared ──")
for img, lbl in train_ds.take(1):
    print(f"  Batch shape: {img.shape}  Range: [{img.numpy().min():.2f}, {img.numpy().max():.2f}]")

# ── 2. Load Pretrained Backbone ───────────────────────────────────────────

backbone = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,       # drop the ImageNet classifier head
    weights="imagenet",      # pretrained on ImageNet
    pooling="avg",           # global average pooling on the feature maps
)

# Freeze backbone initially
backbone.trainable = False

print(f"\n── Backbone: MobileNetV2 ──")
print(f"  Top excluded, pooling='avg'")
print(f"  Trainable: {backbone.trainable}  "
      f"(params: {backbone.count_params():,})")

# ── 3. Build Model ────────────────────────────────────────────────────────

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    backbone,
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n── Full Model (backbone frozen) ──")
model.summary()

# ── 4. Phase 1 — Train the head only ──────────────────────────────────────

print("\n── Phase 1: Training Head ──")
history_phase1 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    verbose=1,
    callbacks=[
        callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=0),
    ],
)

acc_phase1 = max(history_phase1.history["val_accuracy"])
print(f"\nPhase 1 best val accuracy: {acc_phase1:.4f}")

# ── 5. Phase 2 — Fine-tune the backbone ──────────────────────────────────

# Unfreeze the top layers of the backbone while keeping early layers frozen.
backbone.trainable = True

# Freeze early layers (typically the first ~100 layers are generic features)
FINE_TUNE_AT = 100
for layer in backbone.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# Recompile with a lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # 100× lower than phase 1
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"\n── Phase 2: Fine-tuning (unfrozen layers {FINE_TUNE_AT}+) ──")
print(f"  Trainable backbone params: "
      f"{sum(l.count_params() for l in backbone.layers if l.trainable):,}")
model.summary()

history_phase2 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,
    verbose=1,
    callbacks=[
        callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7, verbose=0),
    ],
)

acc_phase2 = max(history_phase2.history["val_accuracy"])
print(f"\nPhase 2 best val accuracy: {acc_phase2:.4f}")

# ── 6. Final Evaluation ───────────────────────────────────────────────────

print("\n── Final Evaluation ──")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {test_acc:.4f}  (loss: {test_loss:.4f})")

# Per-class
all_preds = []
all_true = []
for img, lbl in test_ds:
    logits = model.predict(img, verbose=0)
    all_preds.extend(np.argmax(logits, axis=1))
    all_true.extend(lbl.numpy())

all_preds = np.array(all_preds)
all_true = np.array(all_true)

print("\nPer-class accuracy:")
for i, name in enumerate(CLASS_NAMES):
    mask = all_true == i
    acc = (all_preds[mask] == i).mean()
    bar = "█" * int(acc * 30)
    print(f"  {name:>10}: {acc:.3f}  {bar}")

# ── 7. Prediction Demo ───────────────────────────────────────────────────

indices = np.random.choice(len(x_test), 5, replace=False)
samples = x_test[indices]
true_labels = y_test[indices].flatten()

# Preprocess for prediction
samples_resized = tf.image.resize(samples, (IMG_SIZE, IMG_SIZE)) / 255.0
pred_probs = model.predict(samples_resized, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)

print("\n── Sample Predictions ──")
for i, idx in enumerate(indices):
    correct = "✓" if pred_labels[i] == true_labels[i] else "✗"
    conf = pred_probs[i, pred_labels[i]]
    print(f"  [{correct}] True={CLASS_NAMES[true_labels[i]]:>10}  "
          f"Pred={CLASS_NAMES[pred_labels[i]]:>10}  "
          f"conf={conf:.3f}")

print("\n✅ Transfer learning complete.")
