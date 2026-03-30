from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import ImageDataset
from src.transforms import (
    get_efficientnetv2s_train_transforms,
    get_efficientnetv2s_valid_transforms,
)
from src.models import get_efficientnetv2s_model
from src.engine import train_one_epoch, validate
from src.metrics import compute_f1, compute_precision, compute_recall


PROJECT_ROOT = Path(".")
FOLDS_CSV = PROJECT_ROOT / "outputs" / "folds" / "train_folds.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "efficientnetv2s"
OOF_DIR = PROJECT_ROOT / "outputs" / "oof_predictions"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OOF_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

N_FOLDS = 5
BATCH_SIZE = 16
EPOCHS = 8
LR = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42
LABEL_SMOOTHING = 0.05


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    seed_everything(SEED)

    print(f"Using device: {DEVICE}")

    df = pd.read_csv(FOLDS_CSV)
    df["label"] = df["label"].astype(int)
    df["fold"] = df["fold"].astype(int)

    print("Loaded folds data:", df.shape)

    oof_rows = []
    fold_results = []

    for fold in range(N_FOLDS):
        print("\n" + "=" * 60)
        print(f"FOLD {fold}")
        print("=" * 60)

        train_df = df[df["fold"] != fold].copy().reset_index(drop=True)
        valid_df = df[df["fold"] == fold].copy().reset_index(drop=True)

        print(f"Train rows: {len(train_df)}")
        print(f"Valid rows: {len(valid_df)}")

        train_dataset = ImageDataset(
            train_df,
            transform=get_efficientnetv2s_train_transforms(),
            is_test=False
        )
        valid_dataset = ImageDataset(
            valid_df,
            transform=get_efficientnetv2s_valid_transforms(),
            is_test=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        model = get_efficientnetv2s_model()
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )

        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        best_f1 = -1.0
        best_epoch = -1
        best_probs = None
        best_preds = None
        best_targets = None

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            train_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, DEVICE
            )

            val_loss, preds, probs, targets = validate(
                model, valid_loader, loss_fn, DEVICE
            )

            f1 = compute_f1(targets, preds)
            precision = compute_precision(targets, preds)
            recall = compute_recall(targets, preds)

            print(f"Train Loss : {train_loss:.4f}")
            print(f"Val Loss   : {val_loss:.4f}")
            print(f"F1 Score   : {f1:.4f}")
            print(f"Precision  : {precision:.4f}")
            print(f"Recall     : {recall:.4f}")

            scheduler.step()

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                best_probs = probs.copy()
                best_preds = preds.copy()
                best_targets = targets.copy()

                ckpt_path = CHECKPOINT_DIR / f"efficientnetv2s_fold{fold}_best.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved best checkpoint to: {ckpt_path}")

        fold_precision = compute_precision(best_targets, best_preds)
        fold_recall = compute_recall(best_targets, best_preds)

        print("\nBest fold result")
        print(f"Best Epoch : {best_epoch}")
        print(f"Best F1    : {best_f1:.4f}")
        print(f"Precision  : {fold_precision:.4f}")
        print(f"Recall     : {fold_recall:.4f}")

        valid_oof = valid_df.copy()
        valid_oof["oof_prob_ai"] = best_probs
        valid_oof["oof_pred_label"] = (best_probs >= 0.5).astype(int)
        valid_oof["best_epoch"] = best_epoch
        valid_oof["model_name"] = "efficientnet_v2_s"

        oof_rows.append(valid_oof)

        fold_results.append({
            "fold": fold,
            "best_epoch": int(best_epoch),
            "best_f1": float(best_f1),
            "precision": float(fold_precision),
            "recall": float(fold_recall),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
        })

    oof_df = pd.concat(oof_rows, axis=0).sort_index().reset_index(drop=True)

    overall_f1 = compute_f1(oof_df["label"], oof_df["oof_pred_label"])
    overall_precision = compute_precision(oof_df["label"], oof_df["oof_pred_label"])
    overall_recall = compute_recall(oof_df["label"], oof_df["oof_pred_label"])

    fold_results_df = pd.DataFrame(fold_results)

    print("\n" + "=" * 60)
    print("FINAL 5-FOLD SUMMARY")
    print("=" * 60)
    print(fold_results_df[["fold", "best_epoch", "best_f1", "precision", "recall"]])

    print("\nMean Fold F1      :", round(fold_results_df["best_f1"].mean(), 4))
    print("Std Fold F1       :", round(fold_results_df["best_f1"].std(), 4))
    print("Overall OOF F1    :", round(overall_f1, 4))
    print("Overall Precision :", round(overall_precision, 4))
    print("Overall Recall    :", round(overall_recall, 4))

    oof_path = OOF_DIR / "efficientnetv2s_oof_predictions.csv"
    report_csv_path = REPORT_DIR / "efficientnetv2s_fold_results.csv"
    report_json_path = REPORT_DIR / "efficientnetv2s_summary.json"

    oof_df.to_csv(oof_path, index=False)
    fold_results_df.to_csv(report_csv_path, index=False)

    summary = {
        "device": DEVICE,
        "n_folds": N_FOLDS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "mean_fold_f1": float(fold_results_df["best_f1"].mean()),
        "std_fold_f1": float(fold_results_df["best_f1"].std()),
        "overall_oof_f1": float(overall_f1),
        "overall_precision": float(overall_precision),
        "overall_recall": float(overall_recall),
        "fold_results": fold_results,
    }

    with open(report_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved OOF predictions to:", oof_path)
    print("Saved fold results to   :", report_csv_path)
    print("Saved summary json to   :", report_json_path)


if __name__ == "__main__":
    main()