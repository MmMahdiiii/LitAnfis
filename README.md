# LitANFIS – A PyTorch implementation of literal-aware (positive/negative) neuro-fuzzy systems

This repository contains the reference implementation for the **LitANFIS** family of models together with a few baselines (plain ANFIS and UNFIS).  
The code is 100 % PyTorch and ships with minimal utilities for training, evaluation, hyper-parameter tuning and interactive exploration in notebooks.

## Repository layout

````

LitAnfis-main/
├── model/                  # Core model definitions
│   ├── anfis.py            # Classic ANFIS baseline
│   ├── litanfis.py         # LitANFIS (positive + negative literals, reconstruction head, dropout)
│   └── unfis.py            # UNFIS baseline (feature–relaxed rules)
├── train/                  # Simple training / evaluation scripts
│   ├── classification_test.py
│   └── early_stop.py
├── tests/                  # Dataset wrappers + quick tests
│   ├── uci_test.py         # UCI data-set helpers
│   └── class_test.py       # Digits / Segmentation helpers
├── notebooks/              # Jupyter demo (no heavy dependencies beyond the requirements file)
├── requirements.txt
└── .gitignore

````

### Key files in a bit more detail

| Path | What it does |
|------|--------------|
| `model/litanfis.py` | Implements **LitANFIS** (`nn.Module`) + an `SklearnLitAnfisWrapper` for seamless integration with Scikit-Learn/PyCaret. |
| `train/classification_test.py` | End-to-end example: loads the UCI Wine data-set, trains LitANFIS with early stopping and reports accuracy. |
| `tests/*` | Lightweight wrappers around popular UCI/open data-sets.  They normalise + split data and offer ready-to-use `TensorDataset`s. |
| `notebooks/test_class.ipynb` | Point-and-click walkthrough – great for first contact or teaching. |

---

## Installation

> **Prerequisites** – Python ≥3.10, pip, and a working C/C++ tool-chain (required by some PyCaret extras).

1. **Clone**
   ```bash
   git clone https://github.com/MmMahdiiii/LitAnfis.git
   cd LitAnfis
    ```

2. **Create (optional) virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt        # Installs torch, numpy, scikit-learn, pycaret[full], ucimlrepo, …
   ```

---

## Quick start

### Using the high-level wrapper (scikit-learn style)

```python
from model.litanfis import LitAnfis, SklearnLitAnfisWrapper
from tests.uci_test import Wine                            # tiny helper that downloads & pre-processes data
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wine = Wine()

model_core = LitAnfis(
    in_features = wine.train_numpy()[0].shape[1],
    out_features = len(set(wine.train_numpy()[1])),
    rules = 3,
    drop_out_p = 0.3,
    binary = False,
    device = device
)
```
---

## Model highlights (implementation-centric)

* Generic **`nn.Module`** – drop straight into any PyTorch pipeline.
* Handles **positive *and negative* fuzzy literals** via a learnable “literal gate”.
* Optional **feature relaxation** (constructs incomplete rules).
* **Semi-supervised loss** (cross-entropy + reconstruction) implemented in fewer than 200 LoC.
* Built-in **dropout over rules** for regularisation.
* **`SklearnLitAnfisWrapper`** gives `.predict()` and `.score()` out of the box.
* Minimal dependencies, no obscure CUDA kernels – runs on CPU or GPU.

---
