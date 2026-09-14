from fraud_ml.retraining import evaluate_retraining_policy


def test_retraining_recommended_for_high_score_drift():
    report = {
        "metrics": [
            {
                "feature": "model_fraud_probability",
                "severity": "HIGH",
            }
        ],
        "label_monitoring": {
            "fraud_rate_ratio": 1.0,
        },
        "heldout_performance": {
            "pr_auc": 0.97,
        },
    }

    decision = evaluate_retraining_policy(report)

    assert decision.retraining_recommended is True
    assert decision.auto_promotion_allowed is False
    assert decision.severity == "HIGH"


def test_retraining_recommended_for_label_shift():
    report = {
        "metrics": [],
        "label_monitoring": {
            "fraud_rate_ratio": 3.0,
        },
        "heldout_performance": {
            "pr_auc": 0.97,
        },
    }

    decision = evaluate_retraining_policy(report)

    assert decision.retraining_recommended is True
    assert decision.auto_promotion_allowed is False


def test_retraining_not_required_for_stable_system():
    report = {
        "metrics": [
            {
                "feature": "model_fraud_probability",
                "severity": "LOW",
            }
        ],
        "label_monitoring": {
            "fraud_rate_ratio": 1.1,
        },
        "heldout_performance": {
            "pr_auc": 0.97,
        },
    }

    decision = evaluate_retraining_policy(report)

    assert decision.retraining_recommended is False
    assert decision.auto_promotion_allowed is False
    assert decision.severity == "LOW"


from fraud_ml.retraining import evaluate_candidate_promotion


def test_candidate_promoted_when_metrics_improve():
    production = {
        "pr_auc": 0.95,
        "recall": 0.88,
    }

    candidate = {
        "pr_auc": 0.97,
        "recall": 0.89,
    }

    decision = evaluate_candidate_promotion(
        production,
        candidate,
    )

    assert decision.promotion_recommended is True
    assert decision.decision == "CANDIDATE_APPROVED_FOR_REVIEW"
    assert decision.reasons == []


def test_candidate_rejected_when_pr_auc_is_worse():
    production = {
        "pr_auc": 0.97,
        "recall": 0.88,
    }

    candidate = {
        "pr_auc": 0.95,
        "recall": 0.90,
    }

    decision = evaluate_candidate_promotion(
        production,
        candidate,
    )

    assert decision.promotion_recommended is False
    assert decision.decision == "KEEP_PRODUCTION"


def test_candidate_rejected_for_excessive_recall_drop():
    production = {
        "pr_auc": 0.97,
        "recall": 0.90,
    }

    candidate = {
        "pr_auc": 0.98,
        "recall": 0.85,
    }

    decision = evaluate_candidate_promotion(
        production,
        candidate,
    )

    assert decision.promotion_recommended is False
    assert decision.decision == "KEEP_PRODUCTION"
