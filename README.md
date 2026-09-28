# Quantum Machine Learning for Physiological Stress Classification

> 🔬 **Research in progress:** a manuscript based on this work is in preparation. Results and code may change before publication.

A comparative study of quantum, hybrid, and classical machine learning models for detecting stress and emotional arousal from physiological signals (ECG, EDA, respiration, EEG). The central question is whether quantum models offer any advantage over classical baselines under realistic, **subject-independent** evaluation.

## Research questions

- How do **Quantum Support Vector Machines (QSVM)** and **Variational Quantum Classifiers (VQC)** compare with classical models for physiological stress classification?
- Can a **hybrid classical–quantum neural network** close the gap with deep learning?
- Do models **generalize across datasets** (trained on one, tested on the other)?

## Datasets

| Dataset | Signals | Participants | Task |
|---|---|---|---|
| [WESAD](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) | Chest ECG, EDA, respiration, temperature, accelerometer | 15 | Baseline vs stress (binary), plus amusement (3-class) |
| [DREAMER](https://zenodo.org/records/546113) | 14-channel EEG, ECG | 23 | Low vs high arousal (binary) |

The datasets are not included in this repository. Download them from their official sources.

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
notebooks/
  01_wesad_exploration.ipynb       Data exploration
  02_wesad_preprocessing.ipynb     Windowing and feature extraction
  03_dreamer_exploration.ipynb
  04_dreamer_preprocessing.ipynb
  05_classical_baselines.ipynb     LR, SVM, Random Forest
  06_cnn_baseline.ipynb            1D-CNN
  07_qsvm_experiments.ipynb        Quantum kernel SVM
  08_vqc_experiments-*.ipynb       Variational classifiers (COBYLA, SPSA)
  09_hybrid_cnn_qnn.ipynb          Hybrid classical-quantum network
  10_cross_dataset.ipynb           WESAD <-> DREAMER generalization
  11_results_analysis.ipynb        Comparison across all models
results/
  plots/                           Figures for every notebook
  output_data/                     Processed features and result tables
```

## Tech stack

Python, Qiskit, Qiskit Machine Learning, PennyLane, PyTorch, scikit-learn, NeuroKit2, pandas, NumPy, Matplotlib, seaborn.

## Running the notebooks

```bash
git clone https://github.com/qskenza/quantum-ml-affective-computing.git
cd quantum-ml-affective-computing
pip install -r requirements.txt
jupyter notebook
```

Set `WESAD_PATH` and `DREAMER_PATH` in each notebook's configuration cell, then run the notebooks in order. Quantum experiments run on a classical simulator, and some take several hours.

## Authors

Kenza Qribis and Lina Harcharras, Al Akhawayn University.
