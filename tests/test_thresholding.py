import numpy as np

from fraud_ml.thresholding import evaluate_threshold


def test_threshold_confusion_matrix():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.9, 0.4])

    result = evaluate_threshold(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1


def test_threshold_precision_recall():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.9, 0.4])

    result = evaluate_threshold(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_business_cost():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.9, 0.4])

    result = evaluate_threshold(
        y_true,
        probabilities,
        threshold=0.5,
        false_positive_cost=1.0,
        false_negative_cost=100.0,
    )

    assert result["false_positive_cost"] == 1.0
    assert result["false_negative_cost"] == 100.0
    assert result["estimated_business_cost"] == 101.0
