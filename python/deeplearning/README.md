# Deep Learning — Python Examples

A hands-on collection of worked examples covering the core Python data science and deep learning stack.

## Contents

| # | File | Topics |
|---|------|--------|
| 1 | `01_numpy_basics.py` | Arrays, reshaping, broadcasting, linear algebra, random sampling |
| 2 | `02_pandas_basics.py` | Series, DataFrames, I/O, groupby, merge, pipelines |
| 3 | `03_tensors_pytorch.py` | PyTorch tensors, indexing, device transfers, autograd, simple nn |
| 4 | `04_tensors_tensorflow.py` | TF tensors, eager ops, shape manipulation, `tf.function` |
| 5 | `05_keras_sequential.py` | Keras Sequential API — regression, classification, callbacks |
| 6 | `06_keras_functional.py` | Keras Functional API — multi-input, multi-output, shared layers |
| 7 | `07_keras_cnn.py` | CNN on CIFAR-10 with data augmentation, BatchNorm, Dropout |
| 8 | `08_keras_rnn.py` | RNN / LSTM for text sentiment classification |
| 9 | `09_transfer_learning.py` | Transfer learning with a pretrained backbone (MobileNetV2) |

## Quick Start

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run any example
python 01_numpy_basics.py
python 07_keras_cnn.py        # downloads CIFAR-10 on first run
```

## Requirements

- Python 3.10+
- TensorFlow 2.x
- PyTorch (CPU or CUDA)
- pandas, numpy, matplotlib, scikit-learn

See `requirements.txt` for pinned versions.
