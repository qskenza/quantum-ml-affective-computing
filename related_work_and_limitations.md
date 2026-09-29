# Related Work

| Study | Dataset | Protocol | Models | Best Result |
|---|---|---|---|---|
| Schmidt et al. (2018) — WESAD's original paper | WESAD | LOSO | RF, AdaBoost, LDA, kNN | 93% accuracy (binary), 80% accuracy (3-class) |
| Siirtola (2019) | WESAD (wrist, low-res signals) | Subject-independent | Random Forest, LDA | 88.33% average accuracy (RF, binary); 87.4% balanced accuracy (LDA on skin temp+BVP+HR) |
| Padha & Sahoo (2022) | SWELL-KW (not WESAD) | — | QSVM, VQC, QKNN | Review + comparative implementation on multimodal knowledge-worker stress data |
| Wearable-anxiety QML study (2026, *Applied Sciences*) | Wearable HR + respiration anxiety data | — | VQC, QSVC, PegasosQSVC vs. 7 classical baselines | QSVC 99.90%/99.37% accuracy (noise-free/noisy); VQC 99.47%/97.30% |
| Katsigiannis & Ramzan (2017) — DREAMER's original paper | DREAMER | Subject-independent | Fusion EEG+ECG baseline | 62.32% accuracy (arousal) |
| TDMNN (2026) | DREAMER | 5-fold, subject-dependent | Temporal-difference deep network | 99.51% accuracy (arousal) |
| **This work** | WESAD + DREAMER | LOSO (subject-independent) | LogReg/SVM/RF, 1D-CNN, QSVM, VQC, Hybrid CNN-QNN | F1=0.93 (CNN, WESAD binary); F1=0.75 (CNN, WESAD 3-class); F1≈0.46–0.50 (all models, DREAMER) |

**Reading this fairly:** the WESAD binary CNN result (F1=0.93, ~93% accuracy) lands right in the range of published subject-independent WESAD numbers — a real, literature-consistent result, not an outlier. DREAMER is a different story: most of the very high DREAMER numbers (85–99%) come from subject-*dependent* evaluation and/or fuse EEG+ECG, which isn't comparable to this work's fully subject-independent, EEG-only protocol. The fairer comparison point is DREAMER's own original subject-independent baseline (62.3%), and this work's numbers (~50%) sit below even that — worth stating plainly as a real limitation rather than glossing over it. The two quantum papers don't touch WESAD or DREAMER at all, so their strong numbers support "quantum methods show promise on *some* physiological stress data" as a motivating citation, not a benchmark this work should have matched.

---

# Limitations

1. **Simulator-only quantum execution.** QSVM and VQC results come from Qiskit statevector simulation, not real quantum hardware — no hardware noise, decoherence, or gate error is modeled. (Contrast point: the wearable-anxiety QML study above explicitly benchmarks under noisy conditions; this project doesn't.)

2. **Qubit-count-constrained input dimensionality.** With 3–6 qubits, QSVM/VQC operate on PCA-reduced inputs, while the classical models and CNN see the full feature set / raw signal. Any quantum-vs-classical gap may reflect this dimensionality mismatch as much as the learning algorithm itself.

3. **Classical subsampling, sensitivity-checked.** The classical baselines in `05` are trained on a 150-windows/class/fold subsample for comparability with the quantum models. The full-data rerun (`05b`) showed this cost ≤0.05 F1 — small relative to the 0.10–0.26 fold-to-fold std — so the subsampling itself isn't a major confound, but it's worth stating explicitly rather than leaving implicit.

4. **Small sample size limits statistical power.** LOSO gives only 15 paired folds for WESAD; after Holm-Bonferroni correction across all pairwise model comparisons, some real differences may fail to reach significance simply from limited n, not because no true difference exists.

5. **Hyperparameter search asymmetry.** Classical models get an exhaustive inner-CV grid search; QSVM/VQC's feature-map/entanglement/optimizer space is far larger and was explored less exhaustively due to simulation cost. This makes classical-vs-quantum comparisons somewhat favorable to the classical side.

6. **DREAMER underperformance across every model tested.** F1 stays near 0.46–0.50 for classical, CNN, QSVM, VQC, *and* hybrid — this is a task/feature-set limitation (EEG-only by design, ECG excluded), not specific to any one model family, and sits below even the original DREAMER paper's subject-independent baseline.

7. **Within-subject window overlap.** 50% overlapping windows within a subject/condition aren't fully independent samples — standard practice in this literature, but worth naming rather than leaving implicit.

8. **Subject-to-subject variability.** Some WESAD LOSO folds reach F1=1.0, others (e.g. S8, S9) drop to 0.5–0.6 — consistent with known individual differences in physiological stress response, but it means the mean F1 alone understates how much performance varies by subject.
