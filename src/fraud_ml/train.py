from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from fraud_ml.data import load_transactions
from fraud_ml.features import create_features
from fraud_ml.validation import validate_schema, validate_target


DATA_PATH = Path("data/raw/transactions.csv")

FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "amount_log",
    "orig_balance_delta",
    "dest_balance_delta",
    "orig_balance_error",
    "dest_balance_error",
    "amount_to_orig_balance",
    "hour",
    "day",
]

TARGET = "isFraud"


def time_based_split(df, train_fraction: float = 0.8):
    """
    Split transactions chronologically.

    Earlier transactions are used for training,
    later transactions are used for evaluation.
    """
    split_index = int(len(df) * train_fraction)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def main() -> None:
    print("Loading PaySim dataset...")
    df = load_transactions(DATA_PATH)

    print("Validating dataset...")
    validate_schema(df)
    validate_target(df)

    print("Creating transaction features...")
    df = create_features(df)

    print("Creating chronological train/test split...")
    train_df, test_df = time_based_split(df)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")

    fraud_train = int(y_train.sum())
    fraud_test = int(y_test.sum())

    print(f"Training fraud cases: {fraud_train:,}")
    print(f"Testing fraud cases:  {fraud_test:,}")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / max(positive_count, 1)

    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    mlflow.set_experiment("production-fraud-ml-platform")

    print("Training XGBoost model...")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        threshold = 0.5
        predictions = (probabilities >= threshold).astype(int)

        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc = average_precision_score(y_test, probabilities)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        false_positive_rate = fp / max(fp + tn, 1)

        print("\n" + "=" * 60)
        print("FRAUD MODEL RESULTS")
        print("=" * 60)

        print(f"ROC-AUC:             {roc_auc:.6f}")
        print(f"PR-AUC:              {pr_auc:.6f}")
        print(f"Precision:           {precision:.6f}")
        print(f"Recall:              {recall:.6f}")
        print(f"F1:                  {f1:.6f}")
        print(f"False Positive Rate: {false_positive_rate:.6f}")

        print("\nConfusion Matrix")
        print(f"TN: {tn:,}")
        print(f"FP: {fp:,}")
        print(f"FN: {fn:,}")
        print(f"TP: {tp:,}")

        print("\nClassification Report")
        print(
            classification_report(
                y_test,
                predictions,
                digits=6,
                zero_division=0,
            )
        )

        mlflow.log_params(
            {
                "model": "XGBClassifier",
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
                "threshold": threshold,
                "split_strategy": "chronological_80_20",
                "isFlaggedFraud_used": False,
            }
        )

        mlflow.log_metrics(
            {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": false_positive_rate,
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        )

        mlflow.xgboost.log_model(
            model,
            name="fraud_xgboost_model",
        )

        print("\nMLflow run logged successfully.")


if __name__ == "__main__":
    main()
