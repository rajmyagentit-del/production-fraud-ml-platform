# Model Card

This model card documents the behavioral fraud-detection model used by the Production Fraud ML Platform.

---

## Model Summary

**Model type:** XGBoost binary classifier

**Primary task:** classify transactions as fraud or legitimate

**Training data:** PaySim synthetic financial transactions

**Feature strategy:** current transaction attributes plus point-in-time behavioral aggregates

**Evaluation strategy:** strict chronological train/test split

**Serving framework:** FastAPI

**Deployed decision threshold:** 0.98

**Model artifact:**

```text
models/behavioral_xgboost.json
```

---

## Intended Use

This model is intended as an educational and portfolio demonstration of production-style fraud ML engineering.

It demonstrates:

- handling extreme class imbalance
- leakage-aware feature engineering
- chronological model evaluation
- behavioral feature construction
- threshold optimization
- model serving through an API
- automated testing and deployment

It can be used to study fraud-model architecture and ML engineering patterns.

---

## Out-of-Scope Use

This model should not be used directly to approve, block, or investigate real financial transactions.

It should not be treated as a validated banking, payments, credit, compliance, or anti-money-laundering system.

The model was trained on synthetic PaySim data rather than real customer transaction data.

The reported metrics therefore describe performance on this synthetic evaluation setup only.

---

## Training Dataset

The model was developed using the PaySim synthetic financial transaction dataset.

Dataset characteristics used during development:

| Property | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Fraud transactions | 8,213 |
| Legitimate transactions | 6,354,407 |
| Fraud rate | approximately 0.129% |
| Simulation steps | 1-743 |

PaySim is synthetic data and does not represent a live production customer population.

The extreme class imbalance makes metrics such as precision, recall, F1, and PR-AUC particularly important.

Accuracy is not used as the primary measure of model quality.

---

## Feature Design

The production modeling path combines transaction attributes with historical behavioral signals.

Examples include:

- transaction step
- transaction amount
- origin and destination balance information available to the model
- transaction type indicators
- prior origin transaction count
- prior origin transaction amount statistics
- time since the previous origin transaction
- origin account age
- amount relative to prior behavior
- prior origin-to-destination interaction count
- whether the destination was previously observed

Behavioral aggregates are constructed using only information from PRIOR simulation steps.

Transactions within the same simulation step are treated as simultaneous so they cannot become historical information for each other.

---

## Leakage Investigation

An early model benchmark used post-transaction balance-derived information and produced extremely strong results.

Early benchmark metrics included approximately:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.999977 |
| PR-AUC | 0.998693 |
| Precision | 0.973361 |
| Recall | 0.987776 |
| F1 | 0.980516 |

Instead of presenting these results as production performance, the project treats them as leakage-prone.

Balance-derived fields can encode simulator-specific information that would not necessarily be available in the same form at real-time decision time.

The production modeling path therefore emphasizes chronological evaluation and point-in-time behavioral features.

The `isFlaggedFraud` field is also excluded from the production feature set.

This distinction is important because unusually strong offline metrics should trigger leakage investigation rather than automatically being interpreted as superior model quality.

---

## Chronological Evaluation

The production modeling path uses a strict temporal holdout.

```text
Training: steps 1-354
Testing:  steps 355-743
```

Development split statistics:

| Split | Transactions | Fraud |
|---|---:|---:|
| Training | 5,069,097 | 3,955 |
| Held-out future test | 1,293,523 | 4,258 |

This evaluation asks a more realistic question than a random split:

**Can a model trained on past behavior identify fraud in future transactions?**

---

## Final Behavioral Model Performance

Performance on the held-out future period at the default 0.50 threshold:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.999906 |
| PR-AUC | 0.974028 |
| Precision | 0.770368 |
| Recall | 0.979333 |
| F1 | 0.862372 |
| False-positive rate | 0.000964 |

Confusion-matrix counts:

| Result | Count |
|---|---:|
| True negatives | 1,288,022 |
| False positives | 1,243 |
| False negatives | 88 |
| True positives | 4,170 |

PR-AUC is emphasized because the positive fraud class is extremely rare.

---

## Decision Threshold

Model probability and business decision threshold are treated as separate concerns.

Two useful operating points from development were:

| Threshold | Precision | Recall | F1 | False Positives | False Negatives |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.770368 | 0.979333 | 0.862372 | 1,243 | 88 |
| 0.98 | 0.929336 | 0.877172 | 0.902501 | 284 | 523 |

The public API currently uses a threshold of **0.98** because it produced the best F1 score in the evaluated threshold range.

This illustrates an important operational trade-off:

- lowering the threshold generally catches more fraud but creates more false positives
- raising the threshold generally reduces false positives but can miss more fraud

The correct production threshold would depend on business costs, investigation capacity, customer impact, and risk tolerance.

The repository includes an illustrative cost-sensitive analysis using relative units.

Those values are not real financial costs and should not be interpreted as dollars.

---

## Known Limitations

1. **Synthetic data** — PaySim cannot reproduce every behavioral pattern found in real payment systems.
2. **Generalization is unproven** — performance has not been validated on real customer transactions or a separate real-world fraud dataset.
3. **Online features are not automated yet** — behavioral aggregates are currently supplied to the inference API by the caller.
4. **Fraud patterns change** — a static model can degrade as transaction and attacker behavior evolves.
5. **Threshold selection is demonstration-oriented** — the deployed 0.98 threshold is based on the synthetic evaluation rather than a real business cost model.
6. **High offline metrics require caution** — simulator structure may make PaySim easier to model than real production fraud.

---

## Generalization and Risk Considerations

The model should not be assumed to generalize across banks, payment processors, countries, customer populations, transaction channels, or time periods.

Before real-world use, a fraud model would require evaluation on representative production data and analysis of false-positive and false-negative impact.

Human review, compliance requirements, security controls, privacy requirements, and business rules would also need to be considered in a real financial system.

---

## Monitoring Expectations

A deployed fraud model should be monitored after release rather than treated as permanently accurate.

Important monitoring signals include:

- feature distribution drift
- transaction-type distribution changes
- fraud-score distribution drift
- prediction-rate changes
- missing or invalid feature rates
- latency and API failures
- precision and recall when delayed labels become available

Drift monitoring is implemented using chronological reference/current windows, numerical PSI, transaction-type total variation, model-score PSI, label-shift monitoring, and held-out performance evaluation. Automated retraining remains a planned extension.

---

## Reproducibility

For instructions to reproduce feature generation, training, evaluation, threshold analysis, tests, and API serving, see:

[Local Setup Guide](LOCAL_SETUP.md)

For the complete system design, see:

[Architecture](ARCHITECTURE.md)

For verified deployment evidence, see:

[Project Evidence](PROJECT_EVIDENCE.md)
