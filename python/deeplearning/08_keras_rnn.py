"""
RNN — Recurrent Neural Networks for sequence modeling.

Covers: Simple RNN, LSTM, bidirectional LSTM, stacked LSTMs,
and text sentiment classification with an embedding layer.

Demonstrates the different inductive biases:
  - SimpleRNN: baseline, suffers from vanishing gradients
  - LSTM: gated memory — handles long sequences well
  - Bidirectional LSTM: reads left-to-right AND right-to-left
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow version: {tf.__version__}\n")

# ── 1. Load & Prepare IMDBN
# Using a smaller subset for fast iteration.

VOCAB_SIZE = 10_000
MAX_LEN = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE
)

# Pad sequences to uniform length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

print("── IMDB Dataset ──")
print(f"Train: {x_train.shape},  Test: {x_test.shape}")
print(f"Positive ratio: {y_train.mean():.3f}\n")

# ── 2. Simple RNN (baseline) ──────────────────────────────────────────────

def build_simple_rnn():
    return keras.Sequential([
        layers.Embedding(VOCAB_SIZE, 32, input_length=MAX_LEN),
        layers.SimpleRNN(32, dropout=0.3),
        layers.Dense(1, activation="sigmoid"),
    ])


model_rnn = build_simple_rnn()
model_rnn.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

print("── Simple RNN ──")
model_rnn.summary()

history_rnn = model_rnn.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    verbose=0,
)
val_acc_rnn = max(history_rnn.history["val_accuracy"])
print(f"Val accuracy: {val_acc_rnn:.4f}")
test_acc_rnn = model_rnn.evaluate(x_test, y_test, verbose=0)[1]
print(f"Test accuracy: {test_acc_rnn:.4f}\n")

# ── 3. LSTM ───────────────────────────────────────────────────────────────

def build_lstm(units=32):
    return keras.Sequential([
        layers.Embedding(VOCAB_SIZE, 32, input_length=MAX_LEN),
        layers.LSTM(units, dropout=0.3, return_sequences=False),
        layers.Dense(1, activation="sigmoid"),
    ])


model_lstm = build_lstm(64)
model_lstm.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

print("── LSTM ──")
model_lstm.summary()

history_lstm = model_lstm.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    verbose=0,
)
val_acc_lstm = max(history_lstm.history["val_accuracy"])
print(f"Val accuracy: {val_acc_lstm:.4f}")
test_acc_lstm = model_lstm.evaluate(x_test, y_test, verbose=0)[1]
print(f"Test accuracy: {test_acc_lstm:.4f}\n")

# ── 4. Bidirectional LSTM ─────────────────────────────────────────────────

def build_bilstm(units=32):
    return keras.Sequential([
        layers.Embedding(VOCAB_SIZE, 32, input_length=MAX_LEN),
        layers.Bidirectional(layers.LSTM(units, dropout=0.3)),
        layers.Dense(1, activation="sigmoid"),
    ])


model_bilstm = build_bilstm(32)
model_bilstm.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

print("── Bidirectional LSTM ──")
model_bilstm.summary()

history_bilstm = model_bilstm.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    verbose=0,
)
val_acc_bi = max(history_bilstm.history["val_accuracy"])
print(f"Val accuracy: {val_acc_bi:.4f}")
test_acc_bi = model_bilstm.evaluate(x_test, y_test, verbose=0)[1]
print(f"Test accuracy: {test_acc_bi:.4f}\n")

# ── 5. Stacked LSTM ───────────────────────────────────────────────────────

def build_stacked_lstm():
    return keras.Sequential([
        layers.Embedding(VOCAB_SIZE, 32, input_length=MAX_LEN),
        layers.LSTM(32, dropout=0.2, return_sequences=True),
        layers.LSTM(32, dropout=0.2, return_sequences=False),
        layers.Dense(1, activation="sigmoid"),
    ])


model_stacked = build_stacked_lstm()
model_stacked.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

print("── Stacked LSTM ──")
model_stacked.summary()

history_stacked = model_stacked.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    verbose=0,
)
val_acc_stack = max(history_stacked.history["val_accuracy"])
print(f"Val accuracy: {val_acc_stack:.4f}")
test_acc_stack = model_stacked.evaluate(x_test, y_test, verbose=0)[1]
print(f"Test accuracy: {test_acc_stack:.4f}\n")

# ── 6. Comparison Table ───────────────────────────────────────────────────

print("── Architecture Comparison ──")
print(f"  {'Model':<22} {'Val Acc':>8} {'Test Acc':>8}  {'Params':>8}")
print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'Simple RNN (32)':<22} {val_acc_rnn:>8.4f} {test_acc_rnn:>8.4f} "
      f"{model_rnn.count_params():>8,}")
print(f"  {'LSTM (64)':<22} {val_acc_lstm:>8.4f} {test_acc_lstm:>8.4f} "
      f"{model_lstm.count_params():>8,}")
print(f"  {'BiLSTM (32)':<22} {val_acc_bi:>8.4f} {test_acc_bi:>8.4f} "
      f"{model_bilstm.count_params():>8,}")
print(f"  {'Stacked LSTM (32)':<22} {val_acc_stack:>8.4f} {test_acc_stack:>8.4f} "
      f"{model_stacked.count_params():>8,}")

# ── 7. Inference Demo ─────────────────────────────────────────────────────

# Decode IMDB word index
word_index = keras.datasets.imdb.get_word_index()
reverse_word_index = {v + 3: k for k, v in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"

def decode_review(seq):
    return " ".join(reverse_word_index.get(i, "?") for i in seq if i > 0)

sample_indices = np.random.choice(len(x_test), 3, replace=False)
sample_reviews = x_test[sample_indices]
sample_labels = y_test[sample_indices]

preds = model_bilstm.predict(sample_reviews, verbose=0).flatten()

print("\n── Inference Demo (BiLSTM) ──")
for i, idx in enumerate(sample_indices):
    sentiment = "positive" if preds[i] > 0.5 else "negative"
    correct = "✓" if (preds[i] > 0.5) == bool(sample_labels[i]) else "✗"
    # Show just the first 200 chars of the decoded review
    text = decode_review(sample_reviews[i])[:200]
    print(f"\n  [{correct}] Pred={preds[i]:.3f} ({sentiment}), "
          f"True={'positive' if sample_labels[i] else 'negative'}")
    print(f"  Review preview: {text}...")

print("\n✅ RNN / LSTM complete.")
