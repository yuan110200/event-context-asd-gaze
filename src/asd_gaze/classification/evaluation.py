"""Evaluation, OOF prediction, and model persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

CI_SEED = 20260915
CI_REPLICATES = 5000

from .feature_selection import SelectionConfig, select_oof_fold_features
from .rf_model import (
    CLASSIFICATION_THRESHOLD,
    RandomForestConfig,
    fit_final_train300_calibrated_rf,
    fit_oof_calibrated_rf,
    predict_labels_from_probability,
    predict_positive_probability,
)


def compute_binary_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = CLASSIFICATION_THRESHOLD,
    label: str = "model",
) -> dict[str, float | str]:
    """Compute AUC, ACC, SEN, SPE, and F1 at the supplied threshold."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = predict_labels_from_probability(p, threshold=threshold)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": label,
        "threshold": float(threshold),
        "AUC": float(roc_auc_score(y, p)),
        "ACC": float(accuracy_score(y, pred)),
        "SEN": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "SPE": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


def compute_confidence_intervals(
    y_true: Sequence[int], probabilities: Sequence[float],
    threshold: float = CLASSIFICATION_THRESHOLD,
    *, seed: int = CI_SEED, replicates: int = CI_REPLICATES,
) -> dict[str, float | int | str]:
    """Compute subject-bootstrap and Clopper-Pearson intervals."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    def exact(successes: int, total: int) -> tuple[float, float]:
        return (0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1)),
                1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes)))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == value) for value in (0, 1)]
    if not all(len(group) for group in groups):
        raise ValueError("Confidence intervals require both target classes")
    indices = np.concatenate([rng.choice(group, size=(replicates, len(group)), replace=True) for group in groups], axis=1)
    ys, ps = y[indices], p[indices]
    positives = ys.sum(axis=1)
    negatives = ys.shape[1] - positives
    boot_pred = ps >= threshold
    tp_boot = (boot_pred & (ys == 1)).sum(axis=1)
    f1_boot = np.divide(2 * tp_boot, boot_pred.sum(axis=1) + positives,
                        out=np.zeros(replicates), where=(boot_pred.sum(axis=1) + positives) != 0)
    valid = (positives > 0) & (negatives > 0)
    ranks = rankdata(ps[valid], axis=1)
    auc_boot = ((ranks * ys[valid]).sum(axis=1) - positives[valid] * (positives[valid] + 1) / 2) / (positives[valid] * negatives[valid])
    return {
        "ACC_lo": exact(tp + tn, len(y))[0], "ACC_hi": exact(tp + tn, len(y))[1],
        "SEN_lo": exact(tp, tp + fn)[0], "SEN_hi": exact(tp, tp + fn)[1],
        "SPE_lo": exact(tn, tn + fp)[0], "SPE_hi": exact(tn, tn + fp)[1],
        "AUC_lo": float(np.percentile(auc_boot, 2.5)), "AUC_hi": float(np.percentile(auc_boot, 97.5)),
        "F1_lo": float(np.percentile(f1_boot, 2.5)), "F1_hi": float(np.percentile(f1_boot, 97.5)),
        "bootstrap_replicates": int(replicates), "bootstrap_seed": int(seed),
        "ACC_ci_method": "exact Clopper-Pearson", "SEN_ci_method": "exact Clopper-Pearson",
        "SPE_ci_method": "exact Clopper-Pearson", "AUC_ci_method": f"subject bootstrap, {replicates} replicates",
        "F1_ci_method": f"subject bootstrap, {replicates} replicates",
    }


def make_prediction_table(
    subjects: Sequence[object],
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = CLASSIFICATION_THRESHOLD,
    model_label: str = "rf_top40",
) -> pd.DataFrame:
    """Build a per-subject prediction table separate from summary metrics."""

    p = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            "subject_id": list(subjects),
            "true_label": np.asarray(y_true, dtype=int),
            "probability": p.astype(float),
            "predicted_label": predict_labels_from_probability(p, threshold=threshold),
            "model": model_label,
        }
    )


def save_selected_feature_columns(feature_columns: Sequence[str], path: str | Path) -> None:
    """Save selected feature columns in exact model-input order."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Original_Feature_Key": list(feature_columns)}).to_csv(output, index=False)


def load_selected_feature_columns(path: str | Path) -> list[str]:
    """Load selected feature columns preserving CSV row order."""

    df = pd.read_csv(path)
    if "Original_Feature_Key" not in df.columns:
        raise ValueError("Selected feature CSV is missing Original_Feature_Key")
    features = df["Original_Feature_Key"].astype(str).tolist()
    if len(features) != len(set(features)):
        raise ValueError("Selected feature CSV contains duplicate feature keys")
    return features


def validate_feature_order(actual_columns: Sequence[str], expected_columns: Sequence[str]) -> None:
    """Raise if the feature order differs from the selected feature order."""

    actual = list(actual_columns)
    expected = list(expected_columns)
    if actual != expected:
        raise ValueError("Feature columns do not match the selected feature order")


def save_model_bundle(
    model,
    feature_columns: Sequence[str],
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Persist the complete preprocessing, RF, and calibration object."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(feature_columns),
            "metadata": dict(metadata or {}),
        },
        output,
    )


def load_model_bundle(path: str | Path) -> tuple[object, list[str], dict]:
    """Load a persisted calibrated model bundle."""

    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload or "feature_columns" not in payload:
        raise ValueError("Model bundle must contain model and feature_columns")
    return payload["model"], list(payload["feature_columns"]), dict(payload.get("metadata", {}))


def run_oof_protocol(
    feature_table: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    selection_config: SelectionConfig = SelectionConfig(),
    rf_config: RandomForestConfig = RandomForestConfig(),
    subject_column: str = "subject_id",
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[str]]]:
    """Run strict OOF prediction with per-fold feature selection and calibration."""

    prediction_rows = []
    per_fold_metrics = []
    selected_by_fold: dict[int, list[str]] = {}

    required_assignment_columns = {subject_column, "fold", "split"}
    missing = required_assignment_columns - set(fold_assignments.columns)
    if missing:
        raise ValueError(f"Fold assignments are missing columns: {', '.join(sorted(missing))}")
    if subject_column not in feature_table.columns or target_column not in feature_table.columns:
        raise ValueError("Feature table is missing the subject or target column")
    if feature_table[subject_column].isna().any() or fold_assignments[subject_column].isna().any():
        raise ValueError("OOF subject IDs must not be missing")
    if fold_assignments["fold"].isna().any():
        raise ValueError("OOF fold IDs must not be missing")

    normalized_split = fold_assignments["split"].astype(str).str.lower()
    if not normalized_split.isin({"train", "val"}).all():
        raise ValueError("OOF split must contain only train or val")
    validation_rows = fold_assignments[normalized_split == "val"]
    validation_counts = validation_rows[subject_column].value_counts()
    feature_subjects = set(feature_table[subject_column])
    validation_subjects = set(validation_counts.index)
    if validation_subjects != feature_subjects:
        raise ValueError("OOF validation assignments must cover every feature-table subject exactly")
    if (validation_counts != 1).any():
        raise ValueError("Each subject must appear in exactly one OOF validation fold")
    if fold_assignments["fold"].nunique() == 0:
        raise ValueError("OOF fold assignments are empty")

    for fold in sorted(fold_assignments["fold"].unique()):
        fold_rows = fold_assignments[fold_assignments["fold"] == fold].copy()
        train_ids = set(fold_rows.loc[fold_rows["split"].astype(str).str.lower() == "train", subject_column])
        val_ids = set(fold_rows.loc[fold_rows["split"].astype(str).str.lower() == "val", subject_column])
        if train_ids & val_ids:
            raise ValueError(f"Fold {fold} training and validation subjects overlap")
        if not train_ids or not val_ids:
            raise ValueError(f"Fold {fold} must contain both training and validation subjects")

        selected, _, _, _ = select_oof_fold_features(
            feature_table,
            fold_assignments,
            fold=int(fold),
            config=selection_config,
            subject_column=subject_column,
            target_column=target_column,
        )
        features = selected["Original_Feature_Key"].astype(str).tolist()
        selected_by_fold[int(fold)] = features

        train_df = feature_table[feature_table[subject_column].isin(train_ids)].copy()
        val_df = feature_table[feature_table[subject_column].isin(val_ids)].copy()
        model = fit_oof_calibrated_rf(train_df, features, target_column=target_column, config=rf_config)
        probabilities = predict_positive_probability(model, val_df, features)
        fold_predictions = make_prediction_table(
            val_df[subject_column],
            val_df[target_column],
            probabilities,
            threshold=rf_config.threshold,
            model_label=f"fold_{int(fold):02d}",
        )
        fold_predictions["fold"] = int(fold)
        prediction_rows.append(fold_predictions)
        per_fold_metrics.append(
            {
                "fold": int(fold),
                **compute_binary_metrics(
                    val_df[target_column],
                    probabilities,
                    threshold=rf_config.threshold,
                    label=f"fold_{int(fold):02d}",
                ),
            }
        )

    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    if len(predictions) != len(feature_table) or predictions["subject_id"].duplicated().any():
        raise ValueError("OOF predictions must contain exactly one row per subject")
    metrics = pd.DataFrame(per_fold_metrics)
    return predictions, metrics, selected_by_fold


def evaluate_independent_test(
    model,
    test_table: pd.DataFrame,
    feature_columns: Sequence[str],
    subject_column: str = "subject_id",
    target_column: str = "target",
    threshold: float = CLASSIFICATION_THRESHOLD,
    model_label: str = "rf_top40",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a final model on independent test rows only."""

    probabilities = predict_positive_probability(model, test_table, feature_columns)
    predictions = make_prediction_table(
        test_table[subject_column],
        test_table[target_column],
        probabilities,
        threshold=threshold,
        model_label=model_label,
    )
    metrics = pd.DataFrame(
        [
            compute_binary_metrics(
                test_table[target_column],
                probabilities,
                threshold=threshold,
                label=model_label,
            )
        ]
    )
    return predictions, metrics


def train_final_and_evaluate_test(
    train300_table: pd.DataFrame,
    test70_table: pd.DataFrame,
    feature_columns: Sequence[str],
    rf_config: RandomForestConfig = RandomForestConfig(),
    subject_column: str = "subject_id",
    target_column: str = "target",
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    """Fit Train300-only final model, then evaluate Test70 without training on it."""

    overlap = set(train300_table[subject_column]) & set(test70_table[subject_column])
    if overlap:
        raise ValueError("Train and independent-test subjects must not overlap")

    model = fit_final_train300_calibrated_rf(
        train300_table,
        feature_columns,
        target_column=target_column,
        config=rf_config,
    )
    predictions, metrics = evaluate_independent_test(
        model,
        test70_table,
        feature_columns,
        subject_column=subject_column,
        target_column=target_column,
        threshold=rf_config.threshold,
    )
    return model, predictions, metrics
