# Applied-Research

**Quantum Machine Learning for Physiological Stress Classification: A Comparative Study of QSVM and VQC on WESAD**

Authors: Kenza Qribis, Lina Harcharras

## Overview

This project benchmarks classical machine learning, a 1D-CNN, and quantum/hybrid-quantum
models (QSVM, VQC, and a hybrid CNN-QNN) on physiological-signal stress and affect
classification, using two public datasets: **WESAD** (chest/wrist wearable signals, 15
subjects) and **DREAMER** (EEG/ECG, self-reported valence/arousal).

**Research question:** How do QSVM and VQC compare against classical models (Logistic
Regression, SVM, Random Forest) and a CNN for physiological stress classification, under
subject-independent (Leave-One-Subject-Out, LOSO) evaluation?

**Tasks:**
- WESAD binary — baseline vs. stress
- WESAD 3-class — baseline vs. stress vs. amusement
- DREAMER binary — arousal, EEG features only

**Evaluation protocol:** LOSO cross-validation (one fold per subject).

## Repository structure

```
Applied-Research/
├── data/                      # not included — see "Datasets" below
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

## Datasets

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