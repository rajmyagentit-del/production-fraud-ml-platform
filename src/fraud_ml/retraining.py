from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RetrainingDecision:
    retraining_recommended: bool
    auto_promotion_allowed: bool
    severity: str
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_retraining_policy(
    drift_report: dict,
    fraud_rate_ratio_threshold: float = 2.0,
    min_pr_auc: float = 0.90,
) -> RetrainingDecision:
    reasons: list[str] = []

    metrics = {
        metric["feature"]: metric
        for metric in drift_report.get("metrics", [])
    }

    score_drift = metrics.get("model_fraud_probability", {})
    score_severity = score_drift.get("severity", "LOW")

    fraud_rate_ratio = (
        drift_report.get("label_monitoring", {})
        .get("fraud_rate_ratio", 1.0)
    )

    pr_auc = (
        drift_report.get("heldout_performance", {})
        .get("pr_auc", 0.0)
    )

    if score_severity == "HIGH":
        reasons.append(
            "High model-score distribution drift detected."
        )

    if fraud_rate_ratio >= fraud_rate_ratio_threshold:
        reasons.append(
            "Fraud prevalence shifted beyond the configured threshold."
        )

    if pr_auc < min_pr_auc:
        reasons.append(
            "Held-out PR-AUC fell below the configured minimum."
        )

    retraining_recommended = bool(reasons)

    severity = "HIGH" if retraining_recommended else "LOW"

    return RetrainingDecision(
        retraining_recommended=retraining_recommended,
        auto_promotion_allowed=False,
        severity=severity,
        reasons=reasons,
    )


@dataclass
class PromotionDecision:
    promotion_recommended: bool
    decision: str
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_candidate_promotion(
    production_metrics: dict,
    candidate_metrics: dict,
    min_pr_auc_improvement: float = 0.0,
    max_recall_drop: float = 0.02,
) -> PromotionDecision:
    reasons: list[str] = []

    production_pr_auc = float(
        production_metrics.get("pr_auc", 0.0)
    )
    candidate_pr_auc = float(
        candidate_metrics.get("pr_auc", 0.0)
    )

    production_recall = float(
        production_metrics.get("recall", 0.0)
    )
    candidate_recall = float(
        candidate_metrics.get("recall", 0.0)
    )

    pr_auc_improvement = (
        candidate_pr_auc - production_pr_auc
    )

    recall_drop = (
        production_recall - candidate_recall
    )

    if pr_auc_improvement < min_pr_auc_improvement:
        reasons.append(
            "Candidate PR-AUC does not meet the required improvement."
        )

    if recall_drop > max_recall_drop:
        reasons.append(
            "Candidate recall degradation exceeds the allowed limit."
        )

    promotion_recommended = not reasons

    return PromotionDecision(
        promotion_recommended=promotion_recommended,
        decision=(
            "CANDIDATE_APPROVED_FOR_REVIEW"
            if promotion_recommended
            else "KEEP_PRODUCTION"
        ),
        reasons=reasons,
    )
