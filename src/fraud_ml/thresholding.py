import numpy as np


def evaluate_threshold(
    y_true,
    probabilities,
    threshold: float,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 100.0,
):
    """
    Evaluate a fraud decision threshold.

    Business-cost values are expressed in RELATIVE COST UNITS,
    not currency.

    Default illustrative assumptions:
    - false positive cost = 1 unit
    - false negative cost = 100 units

    These costs are configurable and are used only to compare
    threshold tradeoffs.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    predictions = probabilities >= threshold

    actual_fraud = y_true == 1
    actual_legit = ~actual_fraud

    tp = int(np.sum(predictions & actual_fraud))
    fp = int(np.sum(predictions & actual_legit))
    fn = int(np.sum(~predictions & actual_fraud))
    tn = int(np.sum(~predictions & actual_legit))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    false_positive_cost_total = (
        fp * false_positive_cost
    )

    false_negative_cost_total = (
        fn * false_negative_cost
    )

    estimated_business_cost = (
        false_positive_cost_total
        + false_negative_cost_total
    )

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "false_positive_cost": (
            false_positive_cost_total
        ),
        "false_negative_cost": (
            false_negative_cost_total
        ),
        "estimated_business_cost": (
            estimated_business_cost
        ),
    }
