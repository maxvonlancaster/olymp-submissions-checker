# Main Classification Experiment Results

Executed once by `run_main_classification_experiment.py` with seed 42.

## Protocol

- Main condition retains only records for which `ast.parse` succeeds on the original Parquet code.
- Sensitivity condition uses all manifest records; its results may be confounded by syntax validity and zero-valued fallback features.
- The fixed manifest train/test assignment is preserved and all preprocessing is fit on training rows only.
- Bootstrap intervals resample complete test task groups with replacement, retaining every row in each sampled task; 500 valid replicates are used.
- This experiment evaluates this dataset and its provenance; it does not establish generalization beyond them.

## Integrity

- manifest and feature rows joined one-to-one: **True**
- main valid-Python rows: **9635**
- all manifest rows: **11729**
- task overlap: **0**
- normalized-code overlap: **0**
- non-finite feature values: **0**
- preprocessing: **training data only**
- test-set resplitting: **False**

## valid_python

Train rows: **7,709**; test rows: **1,926**.

| classifier | accuracy | macro-F1 | ROC-AUC | macro-F1 95% CI | ROC-AUC 95% CI |
|---|---:|---:|---:|---|---|
| Random Forest | 0.7741 | 0.7740 | 0.8661 | [0.7529, 0.7957] | [0.8463, 0.8844] |
| XGBoost | 0.7767 | 0.7766 | 0.8723 | [0.7540, 0.7981] | [0.8517, 0.8916] |
| SVM RBF | 0.7773 | 0.7768 | 0.8587 | [0.7543, 0.7998] | [0.8374, 0.8789] |
| Compact 1D CNN | 0.7544 | 0.7542 | 0.8455 | [0.7317, 0.7738] | [0.8241, 0.8665] |

Class-level precision, recall, F1, and confusion matrices are in `results.json`.
Train distribution: `{'Human_written': 3453, 'Ai_generated': 4256}`; test distribution: `{'Human_written': 867, 'Ai_generated': 1059}`.

## all

Train rows: **9,381**; test rows: **2,348**.

| classifier | accuracy | macro-F1 | ROC-AUC | macro-F1 95% CI | ROC-AUC 95% CI |
|---|---:|---:|---:|---|---|
| Random Forest | 0.7666 | 0.7659 | 0.8554 | [0.7436, 0.7847] | [0.8359, 0.8728] |
| XGBoost | 0.7662 | 0.7656 | 0.8593 | [0.7454, 0.7861] | [0.8405, 0.8765] |
| SVM RBF | 0.7624 | 0.7620 | 0.8459 | [0.7391, 0.7820] | [0.8252, 0.8681] |
| Compact 1D CNN | 0.7530 | 0.7518 | 0.8291 | [0.7310, 0.7724] | [0.8080, 0.8515] |

Class-level precision, recall, F1, and confusion matrices are in `results.json`.
Train distribution: `{'Human_written': 4535, 'Ai_generated': 4846}`; test distribution: `{'Human_written': 1136, 'Ai_generated': 1212}`.

## Main Six-Feature Ablation

Each entry removes one feature and reports accuracy / macro-F1 / ROC-AUC on the unchanged valid-Python test set.

| classifier | removed feature | accuracy | macro-F1 | ROC-AUC |
|---|---|---:|---:|---:|
| Random Forest | avg_var_length | 0.6661 | 0.6660 | 0.7346 |
| Random Forest | comment_density | 0.7736 | 0.7734 | 0.8655 |
| Random Forest | num_variables | 0.7767 | 0.7767 | 0.8568 |
| Random Forest | indentation_consistency | 0.7664 | 0.7663 | 0.8551 |
| Random Forest | repetition_score | 0.7705 | 0.7704 | 0.8625 |
| Random Forest | overexplanation_score | 0.7705 | 0.7704 | 0.8600 |
| XGBoost | avg_var_length | 0.6599 | 0.6597 | 0.7324 |
| XGBoost | comment_density | 0.7747 | 0.7745 | 0.8698 |
| XGBoost | num_variables | 0.7752 | 0.7752 | 0.8646 |
| XGBoost | indentation_consistency | 0.7674 | 0.7674 | 0.8626 |
| XGBoost | repetition_score | 0.7710 | 0.7710 | 0.8700 |
| XGBoost | overexplanation_score | 0.7747 | 0.7746 | 0.8666 |

## Reproduction

- Command: `C:\Users\hmeln\sources\repos\olymp-submissions-checker\.venv\Scripts\python.exe C:\Users\hmeln\sources\repos\olymp-submissions-checker\run_main_classification_experiment.py`
- Dependency versions: see `dependency_versions.json`.
- Installation command used after missing dependencies were detected: `.venv\Scripts\python.exe -m pip install numpy scikit-learn xgboost torch`.
- Model binaries are intentionally not saved; exact fitted-model parameters are in `model_parameters.json`.
