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

**Fair comparison:** Quantum kernel estimation is expensive to simulate, so the classical baselines in `05` are trained on identical stratified subsamples per fold as the quantum models. A separate full-data rerun (`05b`) checks how much this subsampling actually costs the classical models (see [Results](#results)).

**Models compared:**

| Family | Models |
|---|---|
| Classical | Logistic Regression, SVM (RBF), Random Forest |
| Deep learning | 1D-CNN on raw signals |
| Quantum kernel | QSVM with ZZ and Pauli feature maps (grid over entanglement, repetitions, PCA size) |
| Variational quantum | VQC with RealAmplitudes and EfficientSU2 ansätze, COBYLA and SPSA optimizers |
| Hybrid | Classical neural front-end + parameterized quantum circuit, trained end-to-end |

**Features:** Hand-crafted physiological features (HRV metrics such as RMSSD and SDNN, EDA, respiration), EEG band power, and frontal asymmetry (DASM/RASM), reduced with PCA before quantum encoding. The CNN and hybrid models instead learn directly from normalized raw signal windows.

## Repository structure

```
quantum-ml-affective-computing/
├── data/                      # not included — see "Dataset setup" below
│   ├── WESAD/
│   └── DREAMER/DREAMER.mat
├── notebooks/
│   ├── 01_wesad_exploration.ipynb
│   ├── 02_wesad_preprocessing.ipynb
│   ├── 03_dreamer_exploration.ipynb
│   ├── 04_dreamer_preprocessing.ipynb
│   ├── 05_classical_baselines.ipynb
│   ├── 05b_classical_baselines_full.ipynb
│   ├── 06_cnn_baseline_PATCHED.ipynb
│   ├── 07_qsvm_experiments.ipynb
│   ├── 08_vqc_experiments-SPSA.ipynb
│   ├── 08_vqc_experiments-cobyla.ipynb
│   ├── 09_hybrid_cnn_qnn.ipynb
│   ├── 10_cross_dataset.ipynb
│   ├── 11_results_analysis.ipynb
│   └── 12_significance_testing.ipynb
├── results/
│   ├── output_data/            # per-fold and summary CSVs, JSON metadata
│   └── plots/
├── related_work_and_limitations.md
└── requirements.txt
```

> Note: `06_cnn_baseline_PATCHED.ipynb` supersedes an earlier, buggy version of the CNN notebook (removed from this repo — see [Notes on the CNN baseline](#notes-on-the-cnn-baseline)).

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
3. **05** — Classical baselines (Logistic Regression, SVM, Random Forest) under LOSO,
   trained on a stratified subsample (150 windows/class/fold) to keep the comparison
   fair against the quantum models in `07`-`08`, which are constrained to small
   inputs by simulator cost.
4. **05b** — Reruns the same classical baselines on the *full* training fold (no
   subsampling), as a sensitivity check and a fair comparison point against the
   full-data CNN (`06`) and hybrid (`09`) models. Also writes a subsample-vs-full
   comparison table.
5. **06** — 1D-CNN baseline, trained on the full (unsampled) dataset. See
   [Notes on the CNN baseline](#notes-on-the-cnn-baseline) below.
6. **07** — QSVM experiments (ZZFeatureMap / PauliFeatureMap kernels).
7. **08** — VQC experiments (RealAmplitudes / EfficientSU2 circuits), with separate
   notebooks for the SPSA and COBYLA optimizers.
8. **09** — Hybrid CNN-QNN (classical front-end + PennyLane parameterized quantum
   circuit back-end), trained on the full dataset.
9. **10** — Cross-dataset validation (train on one dataset, test on the other).
10. **11** — Aggregates all per-model results into a master summary table and
    comparison plots.
11. **12** — Paired statistical significance testing (Wilcoxon signed-rank +
    paired t-test, Holm-Bonferroni corrected) across LOSO folds, for every pair
    of models within each task.

Quantum experiments run on a classical simulator, and some take several hours. The CNN and hybrid notebooks are also slow on CPU (roughly 8-10 minutes/fold at 50 epochs) — expect a full WESAD run of either to take a few hours.

## Notes on the CNN baseline

The 1D-CNN initially performed far worse than every other model (F1≈0.39 binary,
0.23 3-class) due to two bugs, both fixed in `06_cnn_baseline_PATCHED.ipynb`:

- **Per-window normalization was destroying the discriminative signal.**
  Z-scoring each 30s window independently erased absolute-level information
  (mean heart rate, mean skin conductance level, mean temperature) that the
  classical models' engineered features rely on. Fixed by normalizing once per
  subject instead.
- **The respiration bandpass filter (0.1–0.5 Hz at 700 Hz) was numerically
  unstable.** `scipy.signal.butter` + `filtfilt` in transfer-function ("ba") form
  produces `NaN` on a signal this long at this cutoff/sample-rate ratio, which
  silently poisoned training from epoch 1. Fixed by switching to second-order-
  sections form (`output='sos'` + `sosfiltfilt`).

A learning-rate reduction (3e-3 → 3e-4) and gradient clipping were also added as
general training-stability insurance.

After the fix, the CNN became the **best-performing model on both WESAD tasks**
(F1=0.93 binary, F1=0.75 3-class) — see `related_work_and_limitations.md` for how
that compares to the literature, and `results/output_data/significance_tests.csv`
for which differences are statistically significant.

## Results

Per-fold and summary metrics for every model/task are written to
`results/output_data/` (e.g. `classical_baselines_per_fold.csv`,
`classical_baselines_full_per_fold.csv`, `cnn_baseline_per_fold.csv`,
`hybrid_cnn_qnn_per_fold.csv`, `master_results_table.csv`). Paired significance
test results (Wilcoxon + t-test, Holm-Bonferroni corrected) are in
`significance_tests.csv`. Plots are written to `results/plots/`.

## Related work & limitations

See [`related_work_and_limitations.md`](related_work_and_limitations.md) for a
comparison against published WESAD/DREAMER and quantum-ML-for-physiological-signals
results, and a full discussion of this project's limitations (simulator-only
quantum execution, qubit-count-constrained inputs, statistical power, DREAMER
underperformance across all models, and more).

## Authors

Kenza Qribis and Lina Harcharras, Al Akhawayn University.