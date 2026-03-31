from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


PROJECT_ROOT = Path(".")
CONVNEXT_TTA_PATH = PROJECT_ROOT / "outputs" / "oof_predictions" / "convnext_v2_tta_oof_predictions.csv"
EFFNET_TTA_PATH = PROJECT_ROOT / "outputs" / "oof_predictions" / "efficientnetv2s_tta_oof_predictions.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def main():
    conv_df = pd.read_csv(CONVNEXT_TTA_PATH)
    eff_df = pd.read_csv(EFFNET_TTA_PATH)

    required_cols = ["image_id", "label", "tta_prob_ai"]
    for col in required_cols:
        if col not in conv_df.columns or col not in eff_df.columns:
            raise ValueError(f"Both TTA OOF files must contain column: {col}")

    merged = conv_df[["image_id", "label", "tta_prob_ai"]].rename(
        columns={"tta_prob_ai": "conv_tta_prob"}
    ).merge(
        eff_df[["image_id", "label", "tta_prob_ai"]].rename(
            columns={"tta_prob_ai": "eff_tta_prob"}
        ),
        on=["image_id", "label"],
        how="inner"
    )

    if len(merged) == 0:
        raise ValueError("Merged TTA OOF file is empty. Check image_id alignment.")

    y_true = merged["label"].astype(int).values

    weight_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    results = []

    for w_conv in weight_grid:
        w_eff = 1.0 - w_conv
        ensemble_prob = (
            w_conv * merged["conv_tta_prob"].values +
            w_eff * merged["eff_tta_prob"].values
        )

        metrics = compute_metrics(y_true, ensemble_prob, threshold=0.50)
        metrics["conv_weight"] = float(w_conv)
        metrics["eff_weight"] = float(w_eff)
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
    best_row = results_df.iloc[0]

    print("=" * 60)
    print("CONVNEXT TTA + EFFICIENTNET TTA ENSEMBLE RESULTS")
    print("=" * 60)
    print(results_df[["conv_weight", "eff_weight", "f1", "precision", "recall"]])

    print()
    print("Best default-threshold TTA ensemble")
    print(f"ConvNeXt TTA weight -> {best_row['conv_weight']:.2f}")
    print(f"EffNet TTA weight   -> {best_row['eff_weight']:.2f}")
    print(f"F1                  -> {best_row['f1']:.4f}")
    print(f"Precision           -> {best_row['precision']:.4f}")
    print(f"Recall              -> {best_row['recall']:.4f}")

    best_conv_weight = float(best_row["conv_weight"])
    best_eff_weight = float(best_row["eff_weight"])

    merged["ensemble_tta_prob"] = (
        best_conv_weight * merged["conv_tta_prob"] +
        best_eff_weight * merged["eff_tta_prob"]
    )
    merged["ensemble_tta_pred_label"] = (merged["ensemble_tta_prob"] >= 0.50).astype(int)

    results_csv_path = REPORT_DIR / "convnext_effnet_tta_ensemble_weight_search.csv"
    ensemble_oof_path = REPORT_DIR / "convnext_effnet_tta_ensemble_oof.csv"
    summary_json_path = REPORT_DIR / "convnext_effnet_tta_ensemble_summary.json"

    results_df.to_csv(results_csv_path, index=False)
    merged.to_csv(ensemble_oof_path, index=False)

    summary = {
        "best_conv_weight": best_conv_weight,
        "best_eff_weight": best_eff_weight,
        "default_threshold": 0.50,
        "best_f1": float(best_row["f1"]),
        "best_precision": float(best_row["precision"]),
        "best_recall": float(best_row["recall"]),
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("Saved weight search to :", results_csv_path)
    print("Saved ensemble OOF to  :", ensemble_oof_path)
    print("Saved summary to       :", summary_json_path)


if __name__ == "__main__":
    main()