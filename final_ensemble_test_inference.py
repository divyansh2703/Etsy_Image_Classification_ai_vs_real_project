from pathlib import Path
import random
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import ImageDataset
from src.models import get_convnext_model, get_efficientnetv2s_model


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(".")

TEST_CSV = PROJECT_ROOT / "outputs" / "cleaned_data" / "test_clean.csv"

CONVNEXT_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "convnext_v2"
EFFNET_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "efficientnetv2s"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

N_FOLDS = 5
BATCH_SIZE = 16
NUM_WORKERS = 0

FINAL_THRESHOLD = 0.48
CONVNEXT_WEIGHT = 0.50
EFFNET_WEIGHT = 0.50

SEED = 42


# =========================
# SEED
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =========================
# TTA TRANSFORMS
# =========================
def get_convnext_tta_transforms():
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return [
        ("orig", transforms.Compose([
            transforms.Resize(236),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            norm,
        ])),
        ("hflip", transforms.Compose([
            transforms.Resize(236),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            norm,
        ])),
        ("resize_248_crop_224", transforms.Compose([
            transforms.Resize(248),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            norm,
        ])),
    ]


def get_effnet_tta_transforms():
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return [
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


# =========================
# PREDICTION HELPER
# =========================
def predict_probs(model, loader, device):
    model.eval()
    probs = []

    with torch.no_grad():
        for batch in loader:
            images, image_ids = batch
            images = images.to(device)

            outputs = model(images)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1]

            probs.extend(batch_probs.cpu().numpy())

    return np.array(probs)


# =========================
# MODEL FAMILY INFERENCE
# =========================
def run_family_inference(
    test_df: pd.DataFrame,
    model_name: str,
    get_model_fn,
    ckpt_dir: Path,
    tta_transforms: list,
):
    print("\n" + "=" * 60)
    print(f"RUNNING FAMILY: {model_name}")
    print("=" * 60)

    fold_probs_all = []

    for fold in range(N_FOLDS):
        ckpt_path = ckpt_dir / f"{model_name}_fold{fold}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"\nLoading fold {fold} checkpoint: {ckpt_path.name}")

        model = get_model_fn()
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        tta_probs_all = []

        for tta_name, tta_transform in tta_transforms:
            print(f"  TTA: {tta_name}")

            test_dataset = ImageDataset(
                test_df,
                transform=tta_transform,
                is_test=True
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS
            )

            probs = predict_probs(model, test_loader, DEVICE)
            tta_probs_all.append(probs)

        tta_probs_all = np.stack(tta_probs_all, axis=0)
        mean_tta_probs = tta_probs_all.mean(axis=0)

        fold_probs_all.append(mean_tta_probs)

    fold_probs_all = np.stack(fold_probs_all, axis=0)
    mean_family_probs = fold_probs_all.mean(axis=0)

    return mean_family_probs


# =========================
# MAIN
# =========================
def main():
    seed_everything(SEED)

    print(f"Using device: {DEVICE}")

    test_df = pd.read_csv(TEST_CSV)
    print("Loaded test data:", test_df.shape)

    if "image_id" not in test_df.columns or "image_path" not in test_df.columns:
        raise ValueError("test_clean.csv must contain 'image_id' and 'image_path' columns")

    convnext_probs = run_family_inference(
        test_df=test_df,
        model_name="convnext_v2",
        get_model_fn=get_convnext_model,
        ckpt_dir=CONVNEXT_CKPT_DIR,
        tta_transforms=get_convnext_tta_transforms(),
    )

    effnet_probs = run_family_inference(
        test_df=test_df,
        model_name="efficientnetv2s",
        get_model_fn=get_efficientnetv2s_model,
        ckpt_dir=EFFNET_CKPT_DIR,
        tta_transforms=get_effnet_tta_transforms(),
    )

    final_prob = (
        CONVNEXT_WEIGHT * convnext_probs +
        EFFNET_WEIGHT * effnet_probs
    )

    final_pred = (final_prob >= FINAL_THRESHOLD).astype(int)

    output_df = pd.DataFrame({
        "image_id": test_df["image_id"].values,
        "pred_prob_ai": final_prob,
        "pred_label": final_pred,
    })

    submission_df = pd.DataFrame({
        "image_id": test_df["image_id"].values,
        "ground_truth": final_pred,
    })

    full_preds_path = OUTPUT_DIR / "final_ensemble_test_predictions.csv"
    submission_path = OUTPUT_DIR / "submission.csv"
    config_path = OUTPUT_DIR / "final_ensemble_config.json"

    output_df.to_csv(full_preds_path, index=False)
    submission_df.to_csv(submission_path, index=False)

    config = {
        "device": DEVICE,
        "n_folds": N_FOLDS,
        "convnext_weight": CONVNEXT_WEIGHT,
        "efficientnet_weight": EFFNET_WEIGHT,
        "final_threshold": FINAL_THRESHOLD,
        "convnext_tta": [name for name, _ in get_convnext_tta_transforms()],
        "efficientnet_tta": [name for name, _ in get_effnet_tta_transforms()],
        "test_rows": int(len(test_df)),
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("\n" + "=" * 60)
    print("FINAL INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Saved full predictions to: {full_preds_path}")
    print(f"Saved submission file to : {submission_path}")
    print(f"Saved config to          : {config_path}")
    print(f"Total test rows predicted: {len(test_df)}")


if __name__ == "__main__":
    main()