# Architecture-Aware Explainability for Aortic Stenosis Detection

## Paper link
https://proceedings.mlr.press/v317/buol26a.html

This repository contains experiments for binary aortic stenosis (AS) detection from ECG signals stored as `.npy` arrays. The code compares three model families:

- CNN (custom ResNet-18 variant)
- LSTM (sequence model for 1D ECG)
- ViT-MAE (Vision Transformer MAE fine-tuning on ECG-shaped inputs)

The workflow also includes explainability pipelines (Integrated Gradients and LIME) to inspect model behavior.

## Repository Layout

- `CNN-LSTM models.ipynb`  
  End-to-end notebook with dataset preparation, CNN training, LSTM training, and explanation routines.

- `vitmae-model.py`  
  Standalone PyTorch training script for ViT-MAE-based AS classification.

## Data Format and Directory Structure

The code expects ECG samples in NumPy format (`.npy`) grouped by class folder.

Each file should contain a flattened ECG vector with 1280 values.

Example directory layout:

```text
<DATA_ROOT>/
  train/
    AS/
      sample_001.npy
      ...
    No_AS/
      sample_002.npy
      ...
  val/
    AS/
    No_AS/
  test/
    AS/
    No_AS/
```

## Training Strategy Implemented in Code

- Class imbalance handling via oversampling multipliers:
  - `AS`: x21
  - `No_AS`: x5
- Early stopping (validation-guided)
- Cosine learning-rate scheduling
- Standard classification metrics (accuracy, precision, recall, F1)
- Confusion matrix reporting

## Environment Setup

Use Python 3.10+ with a virtual environment.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy pandas matplotlib seaborn plotly tqdm scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow opencv-python scikit-image lime shap wandb captum
```

### ViT-MAE Dependency Note

`vitmae-model.py` loads a local ECG-pretrained checkpoint via:

- model directory: `.ecgmr/vitmae-ecg`

The script comments also reference a forked `transformers` workflow from:

- `Alsalivan/ecgcmr` (external transformers package)

If you do not already have this local checkpoint/fork integration, update the model-loading section accordingly before training.

## How to Run

## 1) Notebook Workflow (CNN + LSTM + Explainability)

Open `CNN-LSTM models.ipynb` and run sections in order:

1. Library imports and environment checks
2. Dataset construction (`NpyDataset`)
3. Data loaders (`train/val/test`)
4. Model section:
   - ResNet-18 training block, or
   - LSTM training block
5. Explainability section (IG/LIME)

Important: update dataset root paths inside the notebook (`/train`, `/val`, `/test`) to your local paths.

## 2) ViT-MAE Script

From the project root:

```bash
python vitmae-model.py
```

Before running, verify:

- Dataset paths in script are valid (`/70_20_10/train`, `/70_20_10/val`, `/70_20_10/test`)
- Local pretrained ViT-MAE checkpoint exists at `.ecgmr/vitmae-ecg`

## Outputs

Depending on the section/script you run, expected artifacts include:

- `best_ecg_vitmae_model.pth`
- `training_curves.png`
- `resnet18_trained_best.pth`
- `resnet18_trained_final.pth`
- `best_lstm_light.pth`

## Reproducibility and Practical Notes

- Keep train/val/test split fixed across experiments.
- Document any path edits and hyperparameter changes before reporting metrics.
- If comparing model families, keep preprocessing and class-balancing policy consistent.
- For explainability, treat IG/LIME as diagnostic tools; validate findings against domain knowledge.

## Current Scope

This repository is focused on model development and explainability experiments. It is not packaged yet as a production training framework.

For cleaner reproducibility, a recommended next step is to factor reusable components from the notebook into Python modules (`datasets.py`, `models.py`, `train.py`, `explain.py`) and add a frozen `requirements.txt`.
