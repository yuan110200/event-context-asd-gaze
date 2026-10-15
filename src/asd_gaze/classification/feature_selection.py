"""Statistical feature selection for the gaze classifier.

The functions operate only on caller-provided training feature tables. They do
not load validation or independent-test data, which keeps OOF fold selection
and final Train300 selection separate from model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


MIN_VALID_N_PER_GROUP = 20
MIN_ABS_EFFECT_SIZE = 0.35
FDR_ALPHA = 0.05
CORR_THRESHOLD = 0.90
CORR_METHOD = "spearman"
TOP_N_FEATURES = 40
ID_COLUMNS = {"id", "subject_id"}


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for the statistical selection rule."""

    min_valid_n_per_group: int = MIN_VALID_N_PER_GROUP
    min_abs_effect_size: float = MIN_ABS_EFFECT_SIZE
    fdr_alpha: float = FDR_ALPHA
    corr_threshold: float = CORR_THRESHOLD
    corr_method: str = CORR_METHOD
    top_n_features: int = TOP_N_FEATURES


def compute_cohen_d(td_values: pd.Series, asd_values: pd.Series) -> float:
    """Return pooled-standard-deviation Cohen's d as TD minus ASD."""

    x = pd.Series(td_values).dropna().astype(float)
    y = pd.Series(asd_values).dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0
    return float((x.mean() - y.mean()) / pooled_std)


def compute_univariate_feature_stats(
    td_features: pd.DataFrame,
    asd_features: pd.DataFrame,
    min_valid_n_per_group: int = MIN_VALID_N_PER_GROUP,
) -> pd.DataFrame:
    """Compute Mann-Whitney U p values and effect sizes for candidate features."""

    metrics = [column for column in (set(td_features.columns) | set(asd_features.columns)) if column not in ID_COLUMNS]
    rows = []
    for metric in sorted(metrics):
        td_values = td_features[metric] if metric in td_features.columns else pd.Series(dtype=float)
        asd_values = asd_features[metric] if metric in asd_features.columns else pd.Series(dtype=float)
        td_values = pd.to_numeric(td_values, errors="coerce").dropna()
        asd_values = pd.to_numeric(asd_values, errors="coerce").dropna()
        if len(td_values) < min_valid_n_per_group or len(asd_values) < min_valid_n_per_group:
            continue

        try:
            _, p_value = mannwhitneyu(td_values, asd_values, alternative="two-sided")
        except ValueError:
            continue

        cohen_d = compute_cohen_d(td_values, asd_values)
        key = str(metric)
        parts = key.split(".")
        is_macro = key.startswith("Overall.")
        rows.append(
            {
                "Original_Feature_Key": key,
                "Feature_Name": parts[-1],
                "Event_ID": None if is_macro else parts[0],
                "Is_Macro_Metric": is_macro,
                "Cohen_D": cohen_d,
                "Abs_Cohen_D": abs(cohen_d) if pd.notna(cohen_d) else np.nan,
                "P_Value_Raw": float(p_value),
                "Mean_TD": float(np.mean(td_values)),
                "Mean_ASD": float(np.mean(asd_values)),
                "N_TD": int(len(td_values)),
                "N_ASD": int(len(asd_values)),
            }
        )
    return pd.DataFrame(rows)


def apply_bh_fdr_correction(stats: pd.DataFrame, alpha: float = FDR_ALPHA) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to raw p values."""

    corrected = stats.copy()
    if corrected.empty:
        corrected["P_Value_FDR"] = pd.Series(dtype=float)
        corrected["Is_Significant"] = pd.Series(dtype=bool)
        return corrected

    reject, pvals_fdr, _, _ = multipletests(corrected["P_Value_Raw"], alpha=alpha, method="fdr_bh")
    corrected["P_Value_FDR"] = pvals_fdr
    corrected["Is_Significant"] = reject
    return corrected


def rank_and_score_features(features: pd.DataFrame, top_n_features: int = TOP_N_FEATURES) -> pd.DataFrame:
    """Rank features by significance, absolute effect size, and raw p value."""

    ranked = features.copy()
    if ranked.empty:
        for column, dtype in [
            ("NegLog10P", float),
            ("Score_D", float),
            ("Score_P", float),
            ("Final_Score", float),
            ("Rank", int),
            ("Selected", int),
        ]:
            ranked[column] = pd.Series(dtype=dtype)
        return ranked

    ranked["P_Value_Raw"] = pd.to_numeric(ranked["P_Value_Raw"], errors="coerce")
    ranked["P_Value_FDR"] = pd.to_numeric(ranked.get("P_Value_FDR", np.nan), errors="coerce")
    ranked["Abs_Cohen_D"] = pd.to_numeric(ranked["Abs_Cohen_D"], errors="coerce").fillna(0.0)
    ranked["Is_Significant"] = ranked.get("Is_Significant", False).fillna(False).astype(bool)

    eps = 1e-12
    ranked["NegLog10P"] = -np.log10(ranked["P_Value_Raw"].clip(lower=eps))
    ranked["Score_D"] = ranked["Abs_Cohen_D"]
    ranked["Score_P"] = ranked["NegLog10P"]
    ranked["Final_Score"] = ranked["Abs_Cohen_D"] + 0.1 * ranked["NegLog10P"]
    ranked = ranked.sort_values(
        by=["Is_Significant", "Abs_Cohen_D", "P_Value_Raw"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    ranked["Selected"] = (ranked["Rank"] <= top_n_features).astype(int)
    return ranked


def filter_fdr_effect_candidates(
    corrected_stats: pd.DataFrame,
    min_abs_effect_size: float = MIN_ABS_EFFECT_SIZE,
) -> pd.DataFrame:
    """Keep features passing BH-FDR significance and the effect-size gate."""

    candidates = corrected_stats.copy()
    if candidates.empty:
        return candidates
    candidates["Abs_Cohen_D"] = pd.to_numeric(candidates["Abs_Cohen_D"], errors="coerce").fillna(0.0)
    selected = candidates[candidates["Is_Significant"] & (candidates["Abs_Cohen_D"] >= min_abs_effect_size)].copy()
    if not selected.empty:
        selected["Significance_Type"] = "FDR"
        selected["Feature_Category"] = np.where(selected["Is_Macro_Metric"], "Macro", "Event")
        selected["Entry_Route"] = "fdr_effect_size"
    return selected


def prune_correlated_features(
    ranked_candidates: pd.DataFrame,
    td_features: pd.DataFrame,
    asd_features: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
    method: str = CORR_METHOD,
) -> tuple[pd.DataFrame, list[tuple[str, str, float, str]]]:
    """Remove lower-ranked features with high absolute Spearman correlation."""

    if ranked_candidates.empty:
        return ranked_candidates.copy(), []

    threshold = pd.to_numeric(pd.Series([threshold]), errors="coerce").iloc[0]
    threshold = CORR_THRESHOLD if pd.isna(threshold) else float(threshold)
    merged = pd.concat([td_features, asd_features], ignore_index=True)
    feature_keys = [key for key in ranked_candidates["Original_Feature_Key"].tolist() if key in merged.columns]
    if len(feature_keys) <= 1:
        return ranked_candidates.copy(), []

    corr_df = merged[feature_keys].apply(pd.to_numeric, errors="coerce")
    corr_mat = corr_df.corr(method=method).abs()

    rank_cols = []
    ascending = []
    if "Final_Score" in ranked_candidates.columns:
        rank_cols.append("Final_Score")
        ascending.append(False)
    rank_cols.extend(["Abs_Cohen_D", "P_Value_Raw"])
    ascending.extend([False, True])
    ranking = ranked_candidates.set_index("Original_Feature_Key")[rank_cols].sort_values(
        by=rank_cols,
        ascending=ascending,
    )

    kept = []
    removed_pairs: list[tuple[str, str, float, str]] = []
    for feature in ranking.index:
        remove_feature = False
        for kept_feature in kept:
            corr_value = corr_mat.loc[feature, kept_feature] if feature in corr_mat.index and kept_feature in corr_mat.columns else np.nan
            corr_value = pd.to_numeric(pd.Series([corr_value]), errors="coerce").iloc[0]
            if pd.notna(corr_value) and float(corr_value) > threshold:
                remove_feature = True
                removed_pairs.append((feature, kept_feature, float(corr_value), method))
                break
        if not remove_feature:
            kept.append(feature)

    kept_df = ranked_candidates[ranked_candidates["Original_Feature_Key"].isin(kept)].copy()
    sort_cols = []
    sort_asc = []
    if "Final_Score" in kept_df.columns:
        sort_cols.append("Final_Score")
        sort_asc.append(False)
    sort_cols.extend(["Is_Macro_Metric", "Abs_Cohen_D", "P_Value_Raw"])
    sort_asc.extend([False, False, True])
    return kept_df.sort_values(by=sort_cols, ascending=sort_asc).reset_index(drop=True), removed_pairs


def select_features_from_training_groups(
    td_train_features: pd.DataFrame,
    asd_train_features: pd.DataFrame,
    config: SelectionConfig = SelectionConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str, float, str]], pd.DataFrame]:
    """Run the Train300/OOF statistical feature-selection sequence."""

    raw_stats = compute_univariate_feature_stats(
        td_train_features,
        asd_train_features,
        min_valid_n_per_group=config.min_valid_n_per_group,
    )
    corrected = apply_bh_fdr_correction(raw_stats, alpha=config.fdr_alpha)
    candidates = filter_fdr_effect_candidates(corrected, min_abs_effect_size=config.min_abs_effect_size)
    scored = rank_and_score_features(candidates, top_n_features=config.top_n_features)
    pruned, removed_pairs = prune_correlated_features(
        scored,
        td_train_features,
        asd_train_features,
        threshold=config.corr_threshold,
        method=config.corr_method,
    )
    ranked = rank_and_score_features(pruned, top_n_features=config.top_n_features)
    if not ranked.empty:
        ranked["Entry_Route"] = np.where(ranked["Is_Significant"], "fdr_effect_pass", "")
        ranked["Final_Selected_Reason"] = np.where(
            ranked["Selected"].astype(int) == 1,
            "top_n_after_fdr_effect_corr",
            "",
        )
    selected = ranked[ranked["Selected"].astype(int) == 1].copy() if not ranked.empty else ranked.copy()
    return selected, ranked, removed_pairs, raw_stats


def select_oof_fold_features(
    feature_table: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    fold: int,
    config: SelectionConfig = SelectionConfig(),
    subject_column: str = "subject_id",
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str, float, str]], pd.DataFrame]:
    """Select features using only the outer-training subjects for one OOF fold."""

    fold_rows = fold_assignments[fold_assignments["fold"] == fold].copy()
    train_ids = set(fold_rows.loc[fold_rows["split"].astype(str).str.lower() == "train", subject_column])
    val_ids = set(fold_rows.loc[fold_rows["split"].astype(str).str.lower() == "val", subject_column])
    if not train_ids:
        raise ValueError(f"Fold {fold} has no training subjects")
    if train_ids & val_ids:
        raise ValueError(f"Fold {fold} training and validation subjects overlap")

    train_table = feature_table[feature_table[subject_column].isin(train_ids)].copy()
    td_train = train_table[train_table[target_column].astype(int) == 0].drop(columns=[target_column], errors="ignore")
    asd_train = train_table[train_table[target_column].astype(int) == 1].drop(columns=[target_column], errors="ignore")
    return select_features_from_training_groups(td_train, asd_train, config=config)


def select_final_train_features(
    train300_features: pd.DataFrame,
    config: SelectionConfig = SelectionConfig(),
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str, float, str]], pd.DataFrame]:
    """Select final features from Train300 only; independent-test rows are not accepted."""

    td_train = train300_features[train300_features[target_column].astype(int) == 0].drop(columns=[target_column], errors="ignore")
    asd_train = train300_features[train300_features[target_column].astype(int) == 1].drop(columns=[target_column], errors="ignore")
    return select_features_from_training_groups(td_train, asd_train, config=config)
