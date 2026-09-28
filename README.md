# Quantum Machine Learning for Physiological Stress Classification

**A Comparative Study of QSVM and VQC on WESAD & DREAMER**

> 🔬 **Research in progress:** a manuscript based on this work is in preparation. Results and code may change before publication.

> © 2026 Kenza Qribis and Lina Harcharras. All rights reserved. This repository contains unpublished research. Please do not reuse, reproduce, or cite the code or results without the authors' permission until publication.

A comparative study of quantum, hybrid, and classical machine learning models for detecting stress and emotional arousal from physiological signals (ECG, EDA, respiration, EEG). The central question is whether quantum models offer any advantage over classical baselines under realistic, **subject-independent** evaluation.

## Research questions

- How do **Quantum Support Vector Machines (QSVM)** and **Variational Quantum Classifiers (VQC)** compare with classical models for physiological stress classification?
- Can a **hybrid classical–quantum neural network** close the gap with deep learning?
- Do models **generalize across datasets** (trained on one, tested on the other)?

**Tasks:**
- WESAD binary — baseline vs. stress
- WESAD 3-class — baseline vs. stress vs. amusement
- DREAMER binary — arousal, EEG features only

## Datasets

| Dataset | Signals | Participants | Task |
|---|---|---|---|
| [WESAD](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) | Chest ECG, EDA, respiration, temperature, accelerometer | 15 | Baseline vs stress (binary), plus amusement (3-class) |
| [DREAMER](https://zenodo.org/records/546113) | 14-channel EEG, ECG | 23 | Low vs high arousal (binary) |

The datasets are not included in this repository. Download them from their official sources (see [Dataset setup](#dataset-setup) below for placement instructions).

## Methodology

**Evaluation:** Leave-One-Subject-Out (LOSO) cross-validation. Every model is tested on a person it has never seen, which reflects real-world use far better than random splits.

**No data leakage:** Scaling, imputation, and PCA are fitted inside each training fold only.

**Fair comparison:** Quantum kernel estimation is expensive to simulate, so classical and quantum models are trained on identical stratified subsamples per fold.

**Models compared:**

| Family | Models |
|---|---|
| Classical | Logistic Regression, SVM (RBF), Random Forest |
| Deep learning | 1D-CNN on raw signals |
| Quantum kernel | QSVM with ZZ and Pauli feature maps (grid over entanglement, repetitions, PCA size) |
| Variational quantum | VQC with RealAmplitudes and EfficientSU2 ansätze, COBYLA and SPSA optimizers |
| Hybrid | Classical neural front-end + parameterized quantum circuit, trained end-to-end |

**Features:** Hand-crafted physiological features (HRV metrics such as RMSSD and SDNN, EDA, respiration), EEG band power, and frontal asymmetry (DASM/RASM), reduced with PCA before quantum encoding.

## Repository structure

```
Applied-Research/
├── data/                      # not included — see "Dataset setup" below
│   ├── WESAD/
│   └── DREAMER/DREAMER.mat
├── notebooks/
│   ├── 01_wesad_exploration.ipynb
│   ├── 02_wesad_preprocessing.ipynb
│   ├── 03_dreamer_exploration.ipynb
│   ├── 04_dreamer_preprocessing.ipynb
│   ├── 05_classical_baselines.ipynb
│   ├── 06_cnn_baseline.ipynb
│   ├── 07_qsvm_experiments.ipynb
│   ├── 08_vqc_experiments-SPSA.ipynb
│   ├── 08_vqc_experiments-cobyla.ipynb
│   ├── 09_hybrid_cnn_qnn.ipynb
│   ├── 10_cross_dataset.ipynb
│   └── 11_results_analysis.ipynb
├── results/
│   ├── output_data/            # per-fold and summary CSVs, JSON metadata
│   └── plots/
└── requirements.txt
```

## Dataset setup

Neither dataset is redistributed in this repo (license terms require downloading
from the original source). Download and place them as follows before running the
notebooks:

- **WESAD** — Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable
  Stress and Affect Detection", ICMI 2018. Available from the University of Siegen:
  https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html
  Unzip so that each subject folder (`S2/`, `S3/`, ...) sits under `data/WESAD/`.

- **DREAMER** — Katsigiannis & Ramzan, "DREAMER: A Database for Emotion Recognition
  Through EEG and ECG Signals from Wireless Low-cost Off-the-Shelf Devices", JBHI 2018.
  Access requires registering with the dataset's distributor (IEEE DataPort) per the
  authors' terms. Place the resulting file at `data/DREAMER/DREAMER.mat`.

Each exploration/preprocessing notebook (`01`-`04`) has a `WESAD_PATH` /
`DREAMER_PATH` variable near the top of its first code cell — update these to your
local paths before running.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10-3.11. `numpy` is pinned below 2.0 for compatibility with
`qiskit-machine-learning` 0.7.x.

## Tech stack

Python, Qiskit, Qiskit Machine Learning, PennyLane, PyTorch, scikit-learn, NeuroKit2, pandas, NumPy, Matplotlib, seaborn.

## Running the pipeline

The notebooks are numbered and meant to be run in order — later notebooks load
artifacts (features, pickled windows, JSON metadata) written by earlier ones to
`results/output_data/`:

1. **01-02** — WESAD exploration and preprocessing (windowing, feature extraction)
2. **03-04** — DREAMER exploration and preprocessing
3. **05** — Classical baselines (Logistic Regression, SVM, Random Forest) under LOSO.
   Currently run on a stratified subsample (150 windows/class/fold) to keep the
   comparison fair against the quantum models in `07`-`08`, which are constrained to
   small inputs by simulator cost.
4. **06** — 1D-CNN baseline, trained on the full (unsampled) dataset.
5. **07** — QSVM experiments (ZZFeatureMap / PauliFeatureMap kernels).
6. **08** — VQC experiments (RealAmplitudes / EfficientSU2 circuits), with separate
   notebooks for the SPSA and COBYLA optimizers.
7. **09** — Hybrid CNN-QNN (classical front-end + PennyLane parameterized quantum
   circuit back-end), trained on the full dataset.
8. **10** — Cross-dataset validation (train on one dataset, test on the other).
9. **11** — Aggregates all per-model results into summary tables and comparison plots.

Quantum experiments run on a classical simulator, and some take several hours.

## Results

Per-fold and summary metrics for every model/task are written to
`results/output_data/` (e.g. `classical_baselines_per_fold.csv`,
`cnn_baseline_per_fold.csv`, `master_results_table.csv`). Plots are written to
`results/plots/`.

## Known limitations / reproducibility notes

- Classical baselines in `05` are subsampled for comparability with the quantum
  models; this makes them *not* directly comparable to the full-data CNN (`06`) and
  hybrid (`09`) results without a separate full-data classical run.
- Quantum models are run on classical simulators (Qiskit Statevector / PennyLane
  default), not real quantum hardware — no hardware noise is modeled.
- Qubit counts are small (3-6), constraining the quantum models to PCA-reduced
  inputs rather than the full feature set used by the classical/CNN models.

## Authors

Kenza Qribis and Lina Harcharras, Al Akhawayn University.
