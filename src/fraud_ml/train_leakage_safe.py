from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from fraud_ml.data import load_transactions
from fraud_ml.validation import validate_schema, validate_target


DATA_PATH = Path("data/raw/transactions.csv")
TARGET = "isFraud"


def create_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features using information available at transaction decision time.

    Deliberately excluded:
    - newbalanceOrig
    - newbalanceDest
    - isFlaggedFraud
    - features derived from post-transaction balances
    """
    data = pd.DataFrame(index=df.index)

    data["step"] = df["step"]
    data["amount"] = df["amount"]
    data["oldbalanceOrg"] = df["oldbalanceOrg"]
    data["oldbalanceDest"] = df["oldbalanceDest"]

    data["hour"] = df["step"] % 24
    data["day"] = df["step"] // 24

    data["amount_to_orig_balance"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1.0)
    )

    transaction_types = pd.get_dummies(
        df["type"],
        prefix="type",
        dtype=int,
    )

    data = pd.concat([data, transaction_types], axis=1)

    return data


def chronological_split(df: pd.DataFrame, fraction: float = 0.8):
    df = df.sort_values("step", kind="stable").reset_index(drop=True)

    split_index = int(len(df) * fraction)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def main() -> None:
    print("Loading PaySim...")
    df = load_transactions(DATA_PATH)

    validate_schema(df)
    validate_target(df)

    print("Creating chronological split...")
    train_df, test_df = chronological_split(df)

    print(
        f"Train steps: {train_df['step'].min()} -> "
        f"{train_df['step'].max()}"
    )
    print(
        f"Test steps:  {test_df['step'].min()} -> "
        f"{test_df['step'].max()}"
    )

    X_train = create_safe_features(train_df)
    X_test = create_safe_features(test_df)

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())

    scale_pos_weight = negatives / max(positives, 1)

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print(f"Training fraud: {positives:,}")
    print(f"Testing fraud:  {int(y_test.sum()):,}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    mlflow.set_experiment("production-fraud-ml-platform")

    print("Training leakage-safe XGBoost...")

    with mlflow.start_run(run_name="xgboost_leakage_safe"):
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc = average_precision_score(y_test, probabilities)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        false_positive_rate = fp / max(fp + tn, 1)

        print("\n" + "=" * 60)
        print("LEAKAGE-SAFE FRAUD MODEL")
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

        mlflow.log_params(
            {
                "model": "XGBClassifier",
                "experiment_type": "leakage_safe",
                "post_transaction_balances": False,
                "isFlaggedFraud_used": False,
                "split_strategy": "chronological",
                "threshold": 0.5,
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "scale_pos_weight": scale_pos_weight,
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
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            }
        )

        mlflow.xgboost.log_model(
            model,
            name="leakage_safe_fraud_model",
        )

        print("\nLeakage-safe MLflow run logged.")


if __name__ == "__main__":
    main()
