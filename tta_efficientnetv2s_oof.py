from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score

from src.dataset import ImageDataset
from src.models import get_efficientnetv2s_model


PROJECT_ROOT = Path(".")
FOLDS_CSV = PROJECT_ROOT / "outputs" / "folds" / "train_folds.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "efficientnetv2s"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OOF_DIR = PROJECT_ROOT / "outputs" / "oof_predictions"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OOF_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

N_FOLDS = 5
BATCH_SIZE = 16
NUM_WORKERS = 0
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_tta_transforms():
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    tta_list = [
        ("orig", transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            norm,
        ])),
        ("hflip", transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            norm,
        ])),
        ("resize_400_crop_384", transforms.Compose([
            transforms.Resize(400),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            norm,
        ])),
    ]
    return tta_list


def predict_probs(model, loader, device):
    model.eval()
    probs = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1]
            probs.extend(batch_probs.cpu().numpy())

    return np.array(probs)


def compute_metrics(y_true, y_pred):
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def main():
    seed_everything(SEED)

    print(f"Using device: {DEVICE}")

    df = pd.read_csv(FOLDS_CSV)
    df["label"] = df["label"].astype(int)
    df["fold"] = df["fold"].astype(int)

    tta_transforms = get_tta_transforms()

    oof_rows = []
    fold_results = []

    for fold in range(N_FOLDS):
        print("\n" + "=" * 60)
        print(f"TTA OOF FOLD {fold}")
        print("=" * 60)

        valid_df = df[df["fold"] == fold].copy().reset_index(drop=True)
        y_true = valid_df["label"].values

        ckpt_path = CHECKPOINT_DIR / f"efficientnetv2s_fold{fold}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = get_efficientnetv2s_model()
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        tta_probs_all = []

        for tta_name, tta_transform in tta_transforms:
            print(f"Running TTA transform: {tta_name}")

            valid_dataset = ImageDataset(
                valid_df,
                transform=tta_transform,
                is_test=False
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS
            )

            probs = predict_probs(model, valid_loader, DEVICE)
            tta_probs_all.append(probs)

        tta_probs_all = np.stack(tta_probs_all, axis=0)
        mean_probs = tta_probs_all.mean(axis=0)
        pred_labels = (mean_probs >= 0.5).astype(int)

        metrics = compute_metrics(y_true, pred_labels)

        print(f"Fold F1        : {metrics['f1']:.4f}")
        print(f"Fold Precision : {metrics['precision']:.4f}")
        print(f"Fold Recall    : {metrics['recall']:.4f}")

        fold_results.append({
            "fold": fold,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })

        fold_oof = valid_df.copy()
        fold_oof["tta_prob_ai"] = mean_probs
        fold_oof["tta_pred_label"] = pred_labels
        fold_oof["model_name"] = "efficientnetv2s_tta"
        oof_rows.append(fold_oof)

    oof_df = pd.concat(oof_rows, axis=0).sort_index().reset_index(drop=True)

    overall_metrics = compute_metrics(
        oof_df["label"].values,
        oof_df["tta_pred_label"].values
    )

    fold_results_df = pd.DataFrame(fold_results)

    print("\n" + "=" * 60)
    print("FINAL EFFICIENTNETV2 S TTA OOF SUMMARY")
    print("=" * 60)
    print(fold_results_df)

    print("\nMean Fold F1      :", round(fold_results_df["f1"].mean(), 4))
    print("Std Fold F1       :", round(fold_results_df["f1"].std(), 4))
    print("Overall OOF F1    :", round(overall_metrics["f1"], 4))
    print("Overall Precision :", round(overall_metrics["precision"], 4))
    print("Overall Recall    :", round(overall_metrics["recall"], 4))

    oof_path = OOF_DIR / "efficientnetv2s_tta_oof_predictions.csv"
    report_csv_path = REPORT_DIR / "efficientnetv2s_tta_fold_results.csv"
    report_json_path = REPORT_DIR / "efficientnetv2s_tta_summary.json"

    oof_df.to_csv(oof_path, index=False)
    fold_results_df.to_csv(report_csv_path, index=False)

    summary = {
        "device": DEVICE,
        "n_folds": N_FOLDS,
        "batch_size": BATCH_SIZE,
        "tta_transforms": [name for name, _ in tta_transforms],
        "mean_fold_f1": float(fold_results_df["f1"].mean()),
        "std_fold_f1": float(fold_results_df["f1"].std()),
        "overall_oof_f1": float(overall_metrics["f1"]),
        "overall_precision": float(overall_metrics["precision"]),
        "overall_recall": float(overall_metrics["recall"]),
        "fold_results": fold_results,
    }

    with open(report_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved TTA OOF predictions to:", oof_path)
    print("Saved fold results to      :", report_csv_path)
    print("Saved summary json to      :", report_json_path)


if __name__ == "__main__":
    main()