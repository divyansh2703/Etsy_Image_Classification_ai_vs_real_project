from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


PROJECT_ROOT = Path(".")
OOF_PATH = PROJECT_ROOT / "outputs" / "oof_predictions" / "efficientnetv2s_oof_predictions.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def main():
    df = pd.read_csv(OOF_PATH)

    if "label" not in df.columns or "oof_prob_ai" not in df.columns:
        raise ValueError("OOF file must contain 'label' and 'oof_prob_ai' columns")

    y_true = df["label"].astype(int).values
    y_prob = df["oof_prob_ai"].astype(float).values

    thresholds = np.arange(0.05, 0.951, 0.01)
    results = [evaluate_threshold(y_true, y_prob, thr) for thr in thresholds]
    results_df = pd.DataFrame(results)

    best_idx = results_df["f1"].idxmax()
    best_row = results_df.loc[best_idx]

    default_row = results_df[np.isclose(results_df["threshold"], 0.50)].iloc[0]

    best_threshold = float(best_row["threshold"])
    best_f1 = float(best_row["f1"])
    best_precision = float(best_row["precision"])
    best_recall = float(best_row["recall"])

    print("=" * 60)
    print("EFFICIENTNETV2 S THRESHOLD TUNING RESULTS")
    print("=" * 60)
    print(
        f"Default threshold 0.50 -> "
        f"F1: {default_row['f1']:.4f}, "
        f"Precision: {default_row['precision']:.4f}, "
        f"Recall: {default_row['recall']:.4f}"
    )
    print()
    print(f"Best threshold       -> {best_threshold:.2f}")
    print(f"Best F1              -> {best_f1:.4f}")
    print(f"Best Precision       -> {best_precision:.4f}")
    print(f"Best Recall          -> {best_recall:.4f}")
    print(f"F1 improvement       -> {best_f1 - float(default_row['f1']):.4f}")

    threshold_csv_path = REPORT_DIR / "efficientnetv2s_threshold_search.csv"
    results_df.to_csv(threshold_csv_path, index=False)

    tuned_df = df.copy()
    tuned_df["tuned_pred_label"] = (tuned_df["oof_prob_ai"] >= best_threshold).astype(int)
    tuned_oof_path = REPORT_DIR / "efficientnetv2s_oof_with_tuned_threshold.csv"
    tuned_df.to_csv(tuned_oof_path, index=False)

    summary = {
        "default_threshold": 0.50,
        "default_f1": float(default_row["f1"]),
        "default_precision": float(default_row["precision"]),
        "default_recall": float(default_row["recall"]),
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "f1_improvement": best_f1 - float(default_row["f1"]),
    }

    summary_json_path = REPORT_DIR / "efficientnetv2s_threshold_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("Saved threshold search to :", threshold_csv_path)
    print("Saved tuned OOF file to   :", tuned_oof_path)
    print("Saved threshold summary to:", summary_json_path)


if __name__ == "__main__":
    main()