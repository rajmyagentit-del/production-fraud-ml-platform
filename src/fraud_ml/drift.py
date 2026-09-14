from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class DriftMetric:
    feature: str
    metric: str
    value: float
    severity: str


def severity_from_value(value: float) -> str:
    if value >= 0.25:
        return "HIGH"
    if value >= 0.10:
        return "MEDIUM"
    return "LOW"


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)

    if reference.empty or current.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy())

    if len(edges) < 3:
        minimum = min(reference.min(), current.min())
        maximum = max(reference.max(), current.max())

        if minimum == maximum:
            return 0.0

        edges = np.linspace(minimum, maximum, bins + 1)

    edges = np.unique(edges)

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_total_variation(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    ref = reference.astype(str).value_counts(normalize=True)
    cur = current.astype(str).value_counts(normalize=True)

    categories = ref.index.union(cur.index)
    ref = ref.reindex(categories, fill_value=0.0)
    cur = cur.reindex(categories, fill_value=0.0)

    return float(0.5 * np.abs(ref - cur).sum())


def calculate_numeric_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Iterable[str],
) -> list[DriftMetric]:
    results: list[DriftMetric] = []

    for feature in features:
        value = population_stability_index(
            reference[feature],
            current[feature],
        )
        results.append(
            DriftMetric(
                feature=feature,
                metric="PSI",
                value=round(value, 6),
                severity=severity_from_value(value),
            )
        )

    return results


def calculate_categorical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature: str,
) -> DriftMetric:
    value = categorical_total_variation(
        reference[feature],
        current[feature],
    )

    return DriftMetric(
        feature=feature,
        metric="total_variation",
        value=round(value, 6),
        severity=severity_from_value(value),
    )


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: Iterable[str],
    categorical_feature: str = "type",
) -> dict:
    metrics = calculate_numeric_drift(
        reference,
        current,
        numeric_features,
    )

    metrics.append(
        calculate_categorical_drift(
            reference,
            current,
            categorical_feature,
        )
    )

    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    overall = max(
        (metric.severity for metric in metrics),
        key=lambda value: severity_rank[value],
        default="LOW",
    )

    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "overall_severity": overall,
        "metrics": [asdict(metric) for metric in metrics],
    }


def calculate_prediction_score_drift(
    reference_scores: pd.Series,
    current_scores: pd.Series,
) -> DriftMetric:
    value = population_stability_index(
        reference_scores,
        current_scores,
    )

    return DriftMetric(
        feature="model_fraud_probability",
        metric="PSI",
        value=round(value, 6),
        severity=severity_from_value(value),
    )


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
