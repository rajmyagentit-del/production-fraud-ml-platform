import numpy as np
import pandas as pd

from fraud_ml.drift import (
    build_drift_report,
    calculate_prediction_score_drift,
    categorical_total_variation,
    population_stability_index,
    severity_from_value,
)


def test_severity_thresholds():
    assert severity_from_value(0.00) == "LOW"
    assert severity_from_value(0.099) == "LOW"
    assert severity_from_value(0.10) == "MEDIUM"
    assert severity_from_value(0.249) == "MEDIUM"
    assert severity_from_value(0.25) == "HIGH"


def test_identical_numeric_distribution_has_no_drift():
    values = pd.Series(np.arange(1, 101, dtype=float))

    psi = population_stability_index(values, values.copy())

    assert psi == 0.0


def test_large_numeric_shift_detects_high_drift():
    reference = pd.Series(np.arange(0, 1000, dtype=float))
    current = pd.Series(np.arange(5000, 6000, dtype=float))

    psi = population_stability_index(reference, current)

    assert psi >= 0.25
    assert severity_from_value(psi) == "HIGH"


def test_categorical_total_variation():
    reference = pd.Series(["PAYMENT"] * 100)
    current = pd.Series(["TRANSFER"] * 100)

    value = categorical_total_variation(reference, current)

    assert value == 1.0
    assert severity_from_value(value) == "HIGH"


def test_drift_report_detects_high_overall_severity():
    reference = pd.DataFrame({
        "amount": np.arange(0, 1000, dtype=float),
        "type": ["PAYMENT"] * 1000,
    })

    current = pd.DataFrame({
        "amount": np.arange(5000, 6000, dtype=float),
        "type": ["TRANSFER"] * 1000,
    })

    report = build_drift_report(
        reference=reference,
        current=current,
        numeric_features=["amount"],
        categorical_feature="type",
    )

    assert report["reference_rows"] == 1000
    assert report["current_rows"] == 1000
    assert report["overall_severity"] == "HIGH"
    assert len(report["metrics"]) == 2


def test_prediction_score_drift_detects_shift():
    reference = pd.Series(
        np.linspace(0.00, 0.20, 1000)
    )
    current = pd.Series(
        np.linspace(0.80, 1.00, 1000)
    )

    metric = calculate_prediction_score_drift(
        reference,
        current,
    )

    assert metric.feature == "model_fraud_probability"
    assert metric.metric == "PSI"
    assert metric.value >= 0.25
    assert metric.severity == "HIGH"
