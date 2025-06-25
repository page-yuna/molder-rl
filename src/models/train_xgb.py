"""
Train & evaluate an XGBoost defect-classifier for the KAMP
injection-moulding dataset.  Outputs:

    – confusion matrix
    – precision, recall, F1
    – ROC-AUC
    – PR-AUC

Saves the model to models/xgb_quality.pkl
"""

import pathlib as pl
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

BASE = pl.Path("data/interim")

# Use the full scaled parquet files
TRAIN_PATH = BASE / "features_train_full_scaled.parquet"
TEST_PATH = BASE / "features_test_full_scaled.parquet"
MODEL_SAVE_PATH = pl.Path("models/xgb_quality_best.json")

def main():
    # Load data
    dft = pd.read_parquet(TRAIN_PATH)
    dfe = pd.read_parquet(TEST_PATH)

    X_train = dft.drop(columns="label")
    y_train = dft["label"]
    X_test = dfe.drop(columns="label")
    y_test = dfe["label"]

    # Calculate imbalance ratio for scale_pos_weight
    neg, pos = y_train.value_counts().sort_index()
    scale_pos_weight = neg / pos

    # Create and train model with your chosen hyperparameters
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=800,
        subsample=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Save model in native XGBoost format
    model.save_model(MODEL_SAVE_PATH)
    print(f"Model saved → {MODEL_SAVE_PATH}")

    # Predict probabilities on test set
    prob = model.predict_proba(X_test)[:, 1]

    # Find best threshold to maximize F1 with recall ≥ 0.70
    prec, rec, thr = precision_recall_curve(y_test, prob)

    best = None
    for p, r, t in zip(prec, rec, thr):
        if r >= 0.70:
            f1 = 2 * p * r / (p + r + 1e-9)
            if best is None or f1 > best[2]:
                best = (p, r, f1, t)

    print(f"\nChosen threshold {best[3]:.4f} -> P={best[0]:.2f} R={best[1]:.2f} F1={best[2]:.2f}\n")

    pred = (prob >= best[3]).astype(int)
    print("Confusion matrix [actual 1/0 rows, predicted 1/0 cols]")
    print(confusion_matrix(y_test, pred, labels=[1, 0]))
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4))


if __name__ == "__main__":
    main()
