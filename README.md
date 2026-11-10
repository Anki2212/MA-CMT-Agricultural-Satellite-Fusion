# MA-CMT: Missing-Aware Cross-Modal Temporal Transformer

**Deep Learning-Based Missing-Aware Multi-Sensor Fusion for Agricultural Satellite Time-Series**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.x-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178c6.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-4.x-38b2ac.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/Tests-40%2F40%20Passing-brightgreen.svg)]()

---

## Abstract & Research Motivation

Accurate agricultural crop classification using Earth Observation time-series data requires fusing complementary modalities: **Synthetic Aperture Radar (SAR from Sentinel-1)** and **Multispectral Optical imagery (from Sentinel-2)**. In real-world agricultural monitoring, satellite observations are frequently missing due to cloud obstruction, cloud shadows, variable revisit intervals, and sensor failures. 

Traditional fusion methods typically:
1. Discard time steps with incomplete sensor acquisitions, resulting in severe temporal information loss.
2. Rely on synthetic imputation or interpolation, introducing artifacts and uncertainty into temporal signals.
3. Apply naive feature concatenation without awareness of which modality was present or missing at each time step.

This project designs, implements, and benchmarks the **MA-CMT (Missing-Aware Cross-Modal Temporal Transformer)**, an end-to-end deep learning architecture that explicitly integrates binary sensor availability masks into a gated cross-modal fusion mechanism and temporal attention transformer.

---

## Architecture Overview

```
                                      ┌──────────────────────┐
                                      │ Sentinel-1 (VV, VH)  │
                                      │    (B, 12, 2)        │
                                      └──────────┬───────────┘
                                                 │
                                         Linear + LayerNorm
                                                 │
                                                 ▼
┌──────────────────────┐              ┌──────────────────────┐
│ Sentinel-2 (B2..B12) │              │    S1 Encoding       │
│    (B, 12, 11)       │              │    (B, 12, 128)      │
└──────────┬───────────┘              └──────────┬───────────┘
           │                                     │
   Linear + LayerNorm                            │
           │                                     │
           ▼                                     │
┌──────────────────────┐                         │
│    S2 Encoding       │                         │
│    (B, 12, 128)      │                         │
└──────────┬───────────┘                         │
           │                                     │
           └───────────────────┬─────────────────┘
                               │
                Availability-Guided Gating
       g = Sigmoid(MLP([h_S1, h_S2, mask_S1, mask_S2]))
                               │
                               ▼
                   Fused Representation
                 h_fused = g1*h_S1 + g2*h_S2
                               │
                               ▼
               Temporal Transformer Encoder
                   (3 Layers, 4 Heads)
                               │
                               ▼
                     Temporal Mean Pooling
                 (across non-masked timesteps)
                               │
                               ▼
                   FC Classification Head
                     (10 Crop Classes)
```

### Key Architectural Components

1. **Per-Modality Linear Encoders**: Separate linear projections with LayerNorm, GELU, and Dropout projecting Sentinel-1 ($2 \to 128$) and Sentinel-2 ($11 \to 128$) into a shared representation space.
2. **Missing Token Zeroing**: Representations for unavailable sensor acquisitions are zeroed out via element-wise multiplication with binary availability masks:
   $$\tilde{h}_{t}^{\text{S1}} = h_{t}^{\text{S1}} \odot m_{t}^{\text{S1}}, \quad \tilde{h}_{t}^{\text{S2}} = h_{t}^{\text{S2}} \odot m_{t}^{\text{S2}}$$
3. **Availability-Guided Gated Cross-Modal Fusion**: A gating network conditions the modality fusion weights on the concatenated encodings and availability masks:
   $$[g_{t}^{\text{S1}}, g_{t}^{\text{S2}}] = \sigma\left(W_g [\tilde{h}_{t}^{\text{S1}} \,\|\, \tilde{h}_{t}^{\text{S2}} \,\|\, m_{t}^{\text{S1}} \,\|\, m_{t}^{\text{S2}}] + b_g\right)$$
   $$h_{t}^{\text{fused}} = W_f \left(g_{t}^{\text{S1}} \odot \tilde{h}_{t}^{\text{S1}} + g_{t}^{\text{S2}} \odot \tilde{h}_{t}^{\text{S2}}\right)$$
   If a modality is missing, its contribution is strictly zero, and the network dynamically reweights the available sensor.
4. **Temporal Transformer Encoder**: 3-layer Transformer encoder with sinusoidal positional encoding operating across the 12 monthly tokens. Completely missing months are masked via attention key padding.
5. **Mask-Aware Pooling & Classification Head**: Temporal mean pooling over valid tokens followed by a multi-layer classification head outputting logits over 10 crop classes.

### Evaluated Architectures

- **MA-CMT (Proposed)**: Availability-mask-guided gated cross-modal fusion + Temporal Transformer.
- **S1-Only Baseline**: Sentinel-1 SAR features alone with Temporal Transformer.
- **S2-Only Baseline**: Sentinel-2 optical features alone with Temporal Transformer.
- **Concat Fusion Baseline**: Naive feature concatenation ($2 + 11 = 13$) with Temporal Transformer (no missing awareness).

---

## Dataset & Preprocessing

The model is trained and benchmarked on the globally distributed **CropHarvest** agricultural remote sensing dataset.

- **Raw Labels**: `data/raw/labels.geojson` (113,893 labeled polygons)
- **Raw Features**: `data/raw/features.tar.gz` (87,464 HDF5 monthly time-series files)
- **Matched Samples**: **42,603** verified samples with ground-truth FAO crop classification labels.
- **Selected Features (13 bands)**:
  - **Sentinel-1 (SAR)**: `VV`, `VH` (Indices 0–1)
  - **Sentinel-2 (Optical)**: `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B9`, `B11`, `B12` (Indices 2–12)
  - *Excluded*: Weather, elevation, slope, and NDVI.
- **Crop Classes (10 Top Classes)**:
  1. `cereals` (13,259 samples)
  2. `other` (8,406 samples)
  3. `non_crop` (5,533 samples)
  4. `fruits_nuts` (4,110 samples)
  5. `vegetables_melons` (3,755 samples)
  6. `oilseeds` (2,868 samples)
  7. `leguminous` (1,837 samples)
  8. `root_tuber` (1,179 samples)
  9. `beverage_spice` (908 samples)
  10. `sugar` (748 samples)
- **Split Strategy**: Stratified 70% Train (29,818) / 15% Validation (6,385) / 15% Test (6,400).
- **Normalization**: Z-score standardization computed **strictly on the training split**, applied identically during training and real-time inference.

---

## Experimental Benchmark Results

Evaluated across **5 missing-data scenarios** and **5 missing rates** ($0\%, 10\%, 20\%, 30\%, 50\%$) on 6,400 test samples.

### 1. Complete Observations Benchmark ($0\%$ Missing)

| Model | Test Accuracy | Macro-F1 | Macro-Precision | Macro-Recall | Best Val Acc |
|---|:---:|:---:|:---:|:---:|:---:|
| **MA-CMT (Proposed)** | **63.61%** | **59.06%** | **57.30%** | **67.60%** | **64.71%** |
| **S2-Only Baseline** | 60.48% | 55.39% | 54.31% | 64.17% | 60.86% |
| **Concat Fusion Baseline** | 60.03% | 55.27% | 53.58% | 64.87% | 60.64% |
| **S1-Only Baseline** | 36.47% | 32.21% | 34.24% | 41.27% | 36.21% |

### 2. Missing-Data Robustness Across Scenarios

- **Random Missing**: MA-CMT maintains superior accuracy over baselines as random observation dropouts increase from 10% to 50%.
- **S1 Sensor Failure**: When radar observations are degraded or completely unavailable, MA-CMT dynamically gates to Sentinel-2 representations, avoiding the severe performance collapse observed in naive concatenation.
- **S2 Sensor Failure**: Optical observations suffering heavy cloud cover are handled by reweighting to radar backscatter signals.
- **Consecutive Month Dropouts**: Simulating prolonged seasonal cloud cover (monsoon seasons), MA-CMT preserves temporal context through attention over surviving intervals.

Full quantitative logs are stored in:
- [`results/experiment_results.csv`](results/experiment_results.csv) (84 evaluation conditions)
- [`results/experiment_results.json`](results/experiment_results.json)
- [`results/accuracy_vs_missing.png`](results/accuracy_vs_missing.png) (Multi-scenario accuracy curves)

---

## Web Application

The interactive research dashboard provides real-time model inference and experimental analytics:

1. **Dashboard (`/`)**: High-level dataset summary metrics, 10-class sample distributions, feature tags, and live model checkpoint statuses (`macmt.pt`, `s1only.pt`, `s2only.pt`, `concat.pt`).
2. **Crop Prediction (`/predict`)**:
   - Filterable sample browser loading test samples from the dataset.
   - Interactive 12-month temporal observation table displaying SAR backscatter (`VV`, `VH`) and optical reflectance values (`B2`–`B12`).
   - Month-level and sensor-level dropout toggles to simulate missing data interactively.
   - Real-time MA-CMT inference displaying predicted crop class, confidence score, and horizontal probability distributions.
3. **Experiments (`/experiments`)**:
   - Comparative benchmark tables across models.
   - Dynamic Recharts line plots for Accuracy and Macro-F1 vs. Missing Rates.
   - Interactive $10 \times 10$ confusion matrix with dynamic heatmap shading.
   - Overall 4-scenario benchmark figure.
4. **About (`/about`)**: Full architectural breakdown, methodology, dataset characteristics, and technical documentation.

---

## Project Structure

```
RioProject/
├── data/
│   ├── raw/                           # Raw CropHarvest files (excluded from Git)
│   │   ├── labels.geojson
│   │   └── features.tar.gz
│   └── processed/                     # Caches (generated on initial run)
│       ├── dataset_cache.pkl
│       └── normalization.npz
├── ml/
│   ├── __init__.py
│   ├── dataset.py                     # Dataset loading, missing simulation, transforms
│   ├── train.py                       # Training pipeline (AdamW, Cosine LR, early stopping)
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py                   # MA-CMT and baseline PyTorch architectures
│   └── experiments/
│       ├── __init__.py
│       └── run_experiments.py         # Multi-scenario evaluation benchmark suite
├── backend/
│   ├── __init__.py
│   └── main.py                        # FastAPI REST API
├── frontend/                          # React + TypeScript + Vite + Tailwind CSS
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── api.ts
│       ├── App.tsx
│       └── pages/
│           ├── Dashboard.tsx
│           ├── CropPrediction.tsx
│           ├── Experiments.tsx
│           └── About.tsx
├── tests/
│   ├── test_all.py                    # 26 unit tests (models, masks, preprocessing)
│   └── test_api.py                    # 14 integration tests (endpoints, real inference)
├── notebooks/
│   └── training_colab.ipynb           # End-to-end Google Colab GPU training notebook
├── checkpoints/                       # Trained PyTorch model checkpoints
│   ├── macmt.pt                       # Proposed MA-CMT checkpoint (~2 MB)
│   ├── s1only.pt                      # S1-Only baseline (~1.65 MB)
│   ├── s2only.pt                      # S2-Only baseline (~1.65 MB)
│   └── concat.pt                      # Concat Fusion baseline (~1.65 MB)
├── results/                           # Benchmark artifacts
│   ├── experiment_results.csv
│   ├── experiment_results.json
│   └── accuracy_vs_missing.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Node.js 18+ and npm

### 1. Clone & Set Up Backend

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Set Up Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

---

## Running the Application

### Start the FastAPI Backend

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
*API interactive documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).*

### Start the React Frontend

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
*Access the web dashboard at [http://127.0.0.1:5173](http://127.0.0.1:5173).*

---

## Running Tests

Execute the comprehensive 40-test validation suite:

```bash
python -m pytest tests/ -v
```

Output:
```
======================= 40 passed in 8.88s =======================
```

To run frontend linting and type checks:
```bash
cd frontend
npm run lint
npm run build
```

---

## Model Training & Reproduction

### Local Training
```bash
# Train proposed MA-CMT model
python ml/train.py --model macmt --epochs 60 --batch 256 --patience 10

# Train baselines
python ml/train.py --model s1only --epochs 60 --batch 256
python ml/train.py --model s2only --epochs 60 --batch 256
python ml/train.py --model concat --epochs 60 --batch 256

# Execute full evaluation benchmarks
python ml/experiments/run_experiments.py
```

### Google Colab GPU Training
Open [`notebooks/training_colab.ipynb`](notebooks/training_colab.ipynb) in Google Colab, select a T4 GPU runtime, and run cells 1–10 to train all models and reproduce the experiment graphs.

---

## License

This project is developed for final-year academic research in agricultural remote sensing and deep learning.
