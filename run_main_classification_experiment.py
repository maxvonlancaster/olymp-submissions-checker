"""Run the reproducible six-feature classification experiment."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "ai_code_detection.parquet"
MANIFEST = ROOT / "datasets" / "ai_code_detection_manifest.csv"
FEATURES_FILE = ROOT / "datasets" / "ai_code_detection_features.csv"
OUTPUT_DIR = ROOT / "classification_experiment_results"
FEATURE_NAMES = (
    "avg_var_length",
    "comment_density",
    "num_variables",
    "indentation_consistency",
    "repetition_score",
    "overexplanation_score",
)
LABELS = ("Human_written", "Ai_generated")
SEED = 42
BOOTSTRAPS = 500


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def load_records() -> tuple[list[dict], dict[int, dict], dict[int, dict], dict[int, bool]]:
    manifest_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    feature_rows = list(csv.DictReader(FEATURES_FILE.open(encoding="utf-8", newline="")))
    source_rows = parquet.read_table(DATASET).to_pylist()
    manifest_by_index = {int(row["original_row_index"]): row for row in manifest_rows}
    features_by_index = {int(row["original_row_index"]): row for row in feature_rows}
    assert len(manifest_by_index) == len(manifest_rows) == len(feature_rows)
    assert set(manifest_by_index) == set(features_by_index)
    validity: dict[int, bool] = {}
    for index in manifest_by_index:
        code = source_rows[index]["code"]
        try:
            ast.parse(code)
            validity[index] = True
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            validity[index] = False
    records = []
    for index, manifest_row in manifest_by_index.items():
        feature_row = features_by_index[index]
        assert manifest_row["label"] == feature_row["label"]
        assert manifest_row["split"] == feature_row["split"]
        records.append({**manifest_row, **{name: float(feature_row[name]) for name in FEATURE_NAMES}, "syntax_valid": validity[index]})
    return records, manifest_by_index, features_by_index, validity


def assert_integrity(records: list[dict], condition: str) -> None:
    selected = [row for row in records if condition == "all" or row["syntax_valid"]]
    train = [row for row in selected if row["split"] == "train"]
    test = [row for row in selected if row["split"] == "test"]
    assert {row["task_name"] for row in train}.isdisjoint({row["task_name"] for row in test})
    assert {row["code_sha256"] for row in train}.isdisjoint({row["code_sha256"] for row in test})
    assert all(np.isfinite(row[name]) for row in selected for name in FEATURE_NAMES)
    assert len({row["original_row_index"] for row in selected}) == len(selected)
    assert Counter(row["label"] for row in selected) == (Counter({"Human_written": 4320, "Ai_generated": 5315}) if condition == "valid_python" else Counter({"Human_written": 5671, "Ai_generated": 6058}))


def make_split(records: list[dict], condition: str):
    selected = [row for row in records if condition == "all" or row["syntax_valid"]]
    train = [row for row in selected if row["split"] == "train"]
    test = [row for row in selected if row["split"] == "test"]
    x_train = np.asarray([[row[name] for name in FEATURE_NAMES] for row in train], dtype=np.float64)
    x_test = np.asarray([[row[name] for name in FEATURE_NAMES] for row in test], dtype=np.float64)
    y_train = np.asarray([1 if row["label"] == "Ai_generated" else 0 for row in train])
    y_test = np.asarray([1 if row["label"] == "Ai_generated" else 0 for row in test])
    return train, test, x_train, x_test, y_train, y_test


class CompactCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 6, 1),
        )

    def forward(self, values):
        return self.network(values).squeeze(1)


def fit_cnn(x_train, y_train, x_test):
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    train_values = preprocessor.fit_transform(x_train)
    test_values = preprocessor.transform(x_test)
    model = CompactCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32))
    features = torch.tensor(train_values, dtype=torch.float32).reshape(-1, 1, 6)
    targets = torch.tensor(y_train, dtype=torch.float32)
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss_function(model(features), targets).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.tensor(test_values, dtype=torch.float32).reshape(-1, 1, 6))).numpy()
    return model, probabilities, {"epochs": 40, "learning_rate": 0.01, "weight_decay": 1e-4}


def metric_values(y_true, probabilities, task_indices=None):
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, predictions, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": {LABELS[i]: float(precision[i]) for i in range(2)},
        "recall": {LABELS[i]: float(recall[i]) for i in range(2)},
        "f1": {LABELS[i]: float(f1[i]) for i in range(2)},
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "support": {LABELS[i]: int(support[i]) for i in range(2)},
    }


def bootstrap_intervals(y_true, probabilities, task_names, replicates=BOOTSTRAPS):
    grouped = defaultdict(list)
    for index, task in enumerate(task_names):
        grouped[task].append(index)
    groups = list(grouped.values())
    rng = np.random.default_rng(SEED)
    values = []
    while len(values) < replicates:
        sampled = rng.choice(len(groups), size=len(groups), replace=True)
        indices = np.concatenate([groups[index] for index in sampled])
        sampled_y = y_true[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        sampled_probabilities = probabilities[indices]
        sampled_predictions = (sampled_probabilities >= 0.5).astype(int)
        values.append((f1_score(sampled_y, sampled_predictions, average="macro", zero_division=0), roc_auc_score(sampled_y, sampled_probabilities)))
    samples = np.asarray(values)
    return {"replicates": replicates, "unit": "task_name groups sampled with replacement", "seed": SEED, "macro_f1": [float(value) for value in np.quantile(samples[:, 0], [0.025, 0.975])], "roc_auc": [float(value) for value in np.quantile(samples[:, 1], [0.025, 0.975])]}


def model_specs(y_train):
    counts = Counter(y_train)
    scale = counts[0] / counts[1]
    return {
        "Random Forest": (Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=1))]), {}),
        "XGBoost": (Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0, scale_pos_weight=scale, eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=SEED))]), {}),
        "SVM RBF": (Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=SEED))]), {}),
    }


def run_condition(records, condition):
    train, test, x_train, x_test, y_train, y_test = make_split(records, condition)
    task_names = [row["task_name"] for row in test]
    result = {"condition": condition, "train_count": len(train), "test_count": len(test), "train_class_distribution": dict(Counter(row["label"] for row in train)), "test_class_distribution": dict(Counter(row["label"] for row in test)), "models": {}}
    params = {}
    for name, (model, model_params) in model_specs(y_train).items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        result["models"][name] = {"metrics": metric_values(y_test, probabilities), "bootstrap_95_ci": bootstrap_intervals(y_test, probabilities, task_names)}
        params[f"{condition}/{name}"] = {"parameters": model.get_params(deep=True), "preprocessing_fit": "train only"}
    model, probabilities, cnn_params = fit_cnn(x_train, y_train, x_test)
    result["models"]["Compact 1D CNN"] = {"metrics": metric_values(y_test, probabilities), "bootstrap_95_ci": bootstrap_intervals(y_test, probabilities, task_names)}
    params[f"{condition}/Compact 1D CNN"] = {"parameters": cnn_params, "preprocessing_fit": "train only"}
    return result, params


def run_ablation(records):
    train, test, x_train, x_test, y_train, y_test = make_split(records, "valid_python")
    output = {}
    for model_name in ("Random Forest", "XGBoost"):
        output[model_name] = {}
        for removed in FEATURE_NAMES:
            keep = [index for index, name in enumerate(FEATURE_NAMES) if name != removed]
            model = model_specs(y_train)[model_name][0]
            model.fit(x_train[:, keep], y_train)
            probabilities = model.predict_proba(x_test[:, keep])[:, 1]
            metrics = metric_values(y_test, probabilities)
            output[model_name][removed] = {key: metrics[key] for key in ("accuracy", "macro_f1", "roc_auc")}
    return output


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def dependency_versions():
    import pyarrow
    import sklearn
    import xgboost

    return {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pyarrow.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "torch": torch.__version__}


def write_report(results, ablation, integrity, command):
    lines = ["# Main Classification Experiment Results", "", "Executed once by `run_main_classification_experiment.py` with seed 42.", "", "## Protocol", "", "- Main condition retains only records for which `ast.parse` succeeds on the original Parquet code.", "- Sensitivity condition uses all manifest records; its results may be confounded by syntax validity and zero-valued fallback features.", "- The fixed manifest train/test assignment is preserved and all preprocessing is fit on training rows only.", "- Bootstrap intervals resample complete test task groups with replacement, retaining every row in each sampled task; 500 valid replicates are used.", "- This experiment evaluates this dataset and its provenance; it does not establish generalization beyond them.", "", "## Integrity", ""]
    for key, value in integrity.items():
        lines.append(f"- {key}: **{value}**")
    for condition, condition_result in results.items():
        lines.extend(["", f"## {condition}", "", f"Train rows: **{condition_result['train_count']:,}**; test rows: **{condition_result['test_count']:,}**.", "", "| classifier | accuracy | macro-F1 | ROC-AUC | macro-F1 95% CI | ROC-AUC 95% CI |", "|---|---:|---:|---:|---|---|"])
        for name, model_result in condition_result["models"].items():
            metrics = model_result["metrics"]
            ci = model_result["bootstrap_95_ci"]
            lines.append(f"| {name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['roc_auc']:.4f} | [{ci['macro_f1'][0]:.4f}, {ci['macro_f1'][1]:.4f}] | [{ci['roc_auc'][0]:.4f}, {ci['roc_auc'][1]:.4f}] |")
        lines.extend(["", "Class-level precision, recall, F1, and confusion matrices are in `results.json`.", f"Train distribution: `{condition_result['train_class_distribution']}`; test distribution: `{condition_result['test_class_distribution']}`."])
    lines.extend(["", "## Main Six-Feature Ablation", "", "Each entry removes one feature and reports accuracy / macro-F1 / ROC-AUC on the unchanged valid-Python test set.", "", "| classifier | removed feature | accuracy | macro-F1 | ROC-AUC |", "|---|---|---:|---:|---:|"])
    for model, removals in ablation.items():
        for removed, metrics in removals.items():
            lines.append(f"| {model} | {removed} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['roc_auc']:.4f} |")
    lines.extend(["", "## Reproduction", "", f"- Command: `{command}`", "- Dependency versions: see `dependency_versions.json`.", "- Installation command used after missing dependencies were detected: `.venv\\Scripts\\python.exe -m pip install numpy scikit-learn xgboost torch`.", "- Model binaries are intentionally not saved; exact fitted-model parameters are in `model_parameters.json`."])
    (OUTPUT_DIR / "results_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything()
    records, _, _, validity = load_records()
    assert_integrity(records, "valid_python")
    assert_integrity(records, "all")
    results = {}
    parameters = {}
    for condition in ("valid_python", "all"):
        results[condition], condition_parameters = run_condition(records, condition)
        parameters.update(condition_parameters)
    ablation = run_ablation(records)
    integrity = {"manifest and feature rows joined one-to-one": True, "main valid-Python rows": sum(validity.values()), "all manifest rows": len(records), "task overlap": 0, "normalized-code overlap": 0, "non-finite feature values": 0, "preprocessing": "training data only", "test-set resplitting": False}
    command = " ".join([str(Path(sys.executable).resolve()), str(Path(__file__).resolve())])
    (OUTPUT_DIR / "results.json").write_text(json.dumps(json_safe({"seed": SEED, "features": FEATURE_NAMES, "results": results, "ablation": ablation}), indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "model_parameters.json").write_text(json.dumps(json_safe(parameters), indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "dependency_versions.json").write_text(json.dumps(dependency_versions(), indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "executed_command.txt").write_text(command + "\n", encoding="utf-8")
    write_report(results, ablation, integrity, command)
    print(f"Wrote {OUTPUT_DIR / 'results_report.md'}")


if __name__ == "__main__":
    main()