"""Random-forest training and sigmoid calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.50


@dataclass(frozen=True)
class RandomForestConfig:
    """RF and calibration parameters."""

    n_estimators: int = 200
    max_depth: int = 8
    min_samples_leaf: int = 5
    class_weight: str = "balanced"
    random_state: int = RANDOM_STATE
    calibration_method: str = "sigmoid"
    final_calibration_splits: int = 5
    threshold: float = CLASSIFICATION_THRESHOLD


def build_rf_pipeline(config: RandomForestConfig = RandomForestConfig()) -> Pipeline:
    """Build the median-impute, scale, random-forest pipeline."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    min_samples_leaf=config.min_samples_leaf,
                    class_weight=config.class_weight,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def make_sigmoid_calibrated_model(
    n_splits: int,
    config: RandomForestConfig = RandomForestConfig(),
) -> CalibratedClassifierCV:
    """Create the sigmoid-calibrated RF estimator."""

    return CalibratedClassifierCV(
        build_rf_pipeline(config),
        method=config.calibration_method,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state),
    )


def _feature_frame(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy with exactly the requested feature order."""

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input data is missing selected feature columns: {', '.join(missing)}")
    return frame.reindex(columns=list(feature_columns))


def _positive_class_column(model: CalibratedClassifierCV) -> int:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Fitted model classes do not include positive label 1: {classes}")
    return classes.index(1)


def fit_oof_calibrated_rf(
    outer_train: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = "target",
    config: RandomForestConfig = RandomForestConfig(),
) -> CalibratedClassifierCV:
    """Fit an OOF-fold model using only the outer-training partition."""

    y_train = outer_train[target_column].astype(int)
    class_counts = y_train.value_counts()
    if set(class_counts.index) != {0, 1}:
        raise ValueError("OOF calibration requires both target classes 0 and 1")
    min_class_n = int(class_counts.min())
    if min_class_n < 2:
        raise ValueError("OOF calibration requires at least 2 samples per class")
    n_splits = min(config.final_calibration_splits, max(2, min_class_n))
    model = make_sigmoid_calibrated_model(n_splits=n_splits, config=config)
    model.fit(_feature_frame(outer_train, feature_columns), y_train)
    return model


def fit_final_train300_calibrated_rf(
    train300: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = "target",
    config: RandomForestConfig = RandomForestConfig(),
) -> CalibratedClassifierCV:
    """Fit the final model on Train300 only with five-fold calibration."""

    y_train = train300[target_column].astype(int)
    class_counts = y_train.value_counts()
    if set(class_counts.index) != {0, 1}:
        raise ValueError("Final calibration requires both target classes 0 and 1")
    if int(class_counts.min()) < config.final_calibration_splits:
        raise ValueError(
            f"Final calibration requires at least {config.final_calibration_splits} samples per class"
        )
    model = make_sigmoid_calibrated_model(n_splits=config.final_calibration_splits, config=config)
    model.fit(_feature_frame(train300, feature_columns), y_train)
    return model


def predict_positive_probability(
    model: CalibratedClassifierCV,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    """Predict calibrated probability for class label 1 using fixed feature order."""

    probabilities = model.predict_proba(_feature_frame(frame, feature_columns))
    return probabilities[:, _positive_class_column(model)].astype(float)


def predict_labels_from_probability(
    probabilities: Sequence[float],
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> np.ndarray:
    """Apply the 0.50 decision threshold."""

    return (np.asarray(probabilities, dtype=float) >= threshold).astype(int)
